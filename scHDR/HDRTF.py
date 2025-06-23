import numpy as np
import dgl
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
import dgl.nn as dglnn
from dgl import function as fn
from sklearn.metrics import roc_curve, accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score

class HeteroRGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats[rel[0]], hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W = nn.Linear(in_dims * 2, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W(x)
        return {'score': y}

    def forward(self, graph, h):
        # h contains the node representations
        with graph.local_scope():
            graph.ndata['h'] = h   
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
        

class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    
    # Get edges that do not exist in the graph
    neg_src = []
    neg_dst = []
    edges_set = set(zip(src.tolist(), dst.tolist()))
    
    for i in range(len(src)):
        while True:
            # Generate random negative samples
            neg_dst_node = torch.randint(0, graph.number_of_nodes(vtype), (k,))
            for j in range(k):
                if (src[i], neg_dst_node[j].item()) not in edges_set:
                    neg_src.append(src[i].item())
                    neg_dst.append(neg_dst_node[j].item())
                    break
            if len(neg_src) == (i + 1) * k:
                break
    
    # Create negative graph
    return dgl.heterograph(
        {etype: (torch.tensor(neg_src), torch.tensor(neg_dst))},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})



def compute_loss1(pos_score, neg_score):
    # Determine the smaller size
    min_size = min(pos_score.size(0), neg_score.size(0))
    
    # Randomly sample to make sizes equal
    pos_indices = torch.randperm(pos_score.size(0))[:min_size]
    neg_indices = torch.randperm(neg_score.size(0))[:min_size]
    
    # Select the same size samples based on the indices
    pos_score = pos_score[pos_indices]
    neg_score = neg_score[neg_indices]
    # Compute margin loss
    margin_loss = 1 + neg_score.unsqueeze(1) - pos_score.unsqueeze(1)
    
    # Use clamp function to limit values less than 0 to 0
    margin_loss = torch.clamp(margin_loss, min=0)
    
    # Calculate the mean of the loss
    loss = torch.mean(margin_loss)
    
    return loss


def compute_loss2(scores, labels):
    criterion = nn.BCEWithLogitsLoss()
    # Ensure labels are in float type
    labels = labels.float()
    loss = criterion(scores, labels)
    return loss



def pca_subspace(data, n_components):
    """
    Use SVD to compute the principal component subspace of the data.
    :param data: Data matrix, shape (n_samples, n_features)
    :param n_components: Number of principal components to extract
    :return: Basis matrix of the subspace, shape (n_features, n_components)
    """
    data_mean = torch.mean(data, dim=0)
    data_centered = data - data_mean
    U, S, V = torch.svd(data_centered)
    return V[:, :n_components]


def subspace_alignment(X_source, X_target, num_components):
    """
    Align the subspaces of the source and target domains.
    :param X_source: Source domain data matrix, shape (n_samples_src, n_features)
    :param X_target: Target domain data matrix, shape (n_samples_tgt, n_features)
    :param num_components: Number of principal components to retain
    :return: Aligned feature matrices of source and target domains
    """

    # Compute the PCA subspaces of both domains
    source_subspace = pca_subspace(X_source, num_components)
    target_subspace = pca_subspace(X_target, num_components)
    
    # Compute the alignment matrix
    alignment_matrix = torch.mm(source_subspace.T, target_subspace)
    
    # Transform the source domain features
    transformed_source = torch.mm(X_source, torch.mm(source_subspace, alignment_matrix))
    
    # The target domain can directly use its PCA subspace for dimensionality reduction since we align to this subspace
    transformed_target = torch.mm(X_target, target_subspace)

    return transformed_source, transformed_target



def Edge_rename(graph, x):
    # Assume source_neg_graph is your original graph
    g = graph
    # Prepare a dictionary to store information of edges that do not need to be removed
    edges_data = {}
    # Iterate through all edge types in the original graph
    for canonical_etype in g.canonical_etypes:
        srctype, etype, dsttype = canonical_etype
        # Get the source and destination nodes for the edge type
        src, dst = g.edges(etype=canonical_etype)
        
        if etype == 'CD':  # If it is an edge type to be removed, change the type name
            new_etype = ('cell', 'CD'+x, 'drug')
        elif etype == 'DC':  # Same for another edge type
            new_etype = ('drug', 'DC'+x, 'cell')
        else:  # If it's not an edge type to be removed or changed, keep it as is
            new_etype = canonical_etype
            
        # Store edge information
        edges_data[new_etype] = (src, dst)
    
    g = dgl.heterograph({etype: (edges_data[etype][0], edges_data[etype][1]) for etype in edges_data})

    return g



def Graph(graph1, graph2):
    merged_edges = {}
    for etype in graph1.canonical_etypes:
        src, dst = graph1.all_edges(form='uv', etype=etype)
        merged_edges[etype] = (src, dst)
    for etype in graph2.canonical_etypes:
        src, dst = graph2.all_edges(form='uv', etype=etype)
        merged_edges[etype] = (src, dst)
        
    return merged_edges


def Fold(folds, fold):
    test_idx = folds[fold]
    train_idx = np.concatenate([folds[i] for i in range(5) if i != fold])
    
    return test_idx, train_idx



class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded


def safe_min_max_normalize(features):
    if features.size(0) == 1:
        # If there is only one sample, return the original features directly
        return features
    
    min_val = torch.min(features, dim=0)[0]
    max_val = torch.max(features, dim=0)[0]

    # Prevent division by zero, set a non-zero minimum range
    range = max_val - min_val
    range[range == 0] = 1  # Avoid division by zero error

    # Apply Min-Max normalization
    normalized_features = (features - min_val) / range
    return normalized_features



def maximum_mean_discrepancy(x, y, kernel_width=1.0):
    # Compute the Gaussian kernel matrices
    x_kernel = torch.exp(-0.5 * torch.cdist(x, x) ** 2 / (kernel_width ** 2))
    y_kernel = torch.exp(-0.5 * torch.cdist(y, y) ** 2 / (kernel_width ** 2))
    xy_kernel = torch.exp(-0.5 * torch.cdist(x, y) ** 2 / (kernel_width ** 2))

    # Compute the MMD loss
    mmd_loss = torch.mean(x_kernel) - 2 * torch.mean(xy_kernel) + torch.mean(y_kernel)
    return mmd_loss
