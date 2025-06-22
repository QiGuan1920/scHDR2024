# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:51:11 2024

@author: 78760
"""

from HDRTF import *
from dgl.data.utils import save_graphs, load_graphs

class process:
    
    def __init__(self, S):
        self.preprocess(S)
    
    def preprocess(self, S):

        seed = 14
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        source_basic_graph,_ = load_graphs("../../Graph-Drug/GDSC_"+S+"_basic.bin")
        source_pos_graph,_ = load_graphs("../../Graph-Drug/GDSC_"+S+"_pos.bin")
        source_neg_graph,_ = load_graphs("../../Graph-Drug/GDSC_"+S+"_neg.bin")


        target_basic_graph,_ = load_graphs('../../Graph/'+S+'_basic.bin')
        target_pos_graph,_ = load_graphs('../../Graph/'+S+'_pos.bin')
        target_neg_graph,_ = load_graphs('../../Graph/'+S+'_neg.bin')

        # 使用新的名字重新命名图
        source_basic_graph = source_basic_graph[0]
        source_pos_graph = source_pos_graph[0]
        source_neg_graph = source_neg_graph[0]

        target_basic_graph = target_basic_graph[0]
        target_pos_graph = target_pos_graph[0]
        target_neg_graph = target_neg_graph[0]

        self.source_pos_graph = Edge_rename(source_pos_graph, 'pos')
        self.source_neg_graph = Edge_rename(source_neg_graph, 'neg')
        self.target_pos_graph = Edge_rename(target_pos_graph, 'pos')
        self.target_neg_graph = Edge_rename(target_neg_graph, 'neg')
        
        merged_edges = Graph(self.source_pos_graph, self.source_neg_graph)
        source_X_graph = dgl.heterograph(merged_edges)


        dec_graph = source_X_graph['drug', :, 'cell']
        edge_label = dec_graph.edata[dgl.ETYPE]

        # 获取唯一的边类型Label
        unique_labels = torch.unique(edge_label)
        # 构建从0开始的映射
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        # 根据映射更新边类型Label
        edge_label2 = torch.tensor([label_map[label.item()] for label in edge_label])


        num_edges = len(edge_label2)
        # 生成边的索引
        idx = np.arange(num_edges)
        # 随机打乱索引
        np.random.shuffle(idx)
        # 将边等分成五份
        folds = np.array_split(idx, 5)
        
        self.dec_graph = dec_graph
        self.edge_label2 = edge_label2
        self.folds = folds
        
        self.source_basic_graph = source_basic_graph
        self.target_basic_graph = target_basic_graph


        self.source_features = {node_type: self.source_basic_graph.nodes[node_type].data['features'] for node_type in self.source_basic_graph.ntypes}
        self.target_features = {node_type: self.target_basic_graph.nodes[node_type].data['features'] for node_type in self.target_basic_graph.ntypes}

        self.source_features = {node_type: safe_min_max_normalize(features) for node_type, features in self.source_features.items()}
        self.target_features = {node_type: safe_min_max_normalize(features) for node_type, features in self.target_features.items()}


        self.rel_names = [('drug', 'DT', 'target'),('target', 'TD', 'drug'),('cell', 'CT', 'target'), ('target', 'TC', 'cell'), ('cell', 'CC1', 'cell'),('cell', 'CC2', 'cell'), ('target', 'TT1', 'target'), ('target', 'TT2', 'target')]


        self.in_feats = {'drug': 167, 'cell': 5000, 'target': 147}
        self.hidden_feats = 64
        self.out_feats = 64

        self.target_gcn_model = HeteroRGCN(self.in_feats, self.hidden_feats, self.out_feats, self.rel_names)
        self.linklink_predictor = HeteroMLPPredictor(32, 1)
        # linklink_predictor = HeteroMLPPredictor(64, 1)
        self.link_predictor = HeteroDotProductPredictor()
        self.AEmodel = Autoencoder(64, 32)

        self.optimizer = torch.optim.Adam(list(self.target_gcn_model.parameters())+ list(self.linklink_predictor.parameters()) + list(self.link_predictor.parameters()) + list(self.AEmodel.parameters()), lr=0.001, weight_decay=0.001)


        merged_edges = Graph(self.target_pos_graph, self.target_neg_graph)
        target_X_graph = dgl.heterograph(merged_edges)

        dec_graph_ = target_X_graph['drug', :, 'cell']
        edge_label_ = dec_graph_.edata[dgl.ETYPE]

        # 获取唯一的边类型Label
        unique_labels = torch.unique(edge_label_)
        # 构建从0开始的映射
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        # 根据映射更新边类型Label
        edge_label2_ = torch.tensor([label_map[label.item()] for label in edge_label_])
        
        self.dec_graph_ = dec_graph_
        self.edge_label2_ = edge_label2_




def lossloss(linkmodel, graph, k, embeddings, etype):
    
    negative_graph = construct_negative_graph(graph, k, etype)
    pos_score = linkmodel(graph,  embeddings, etype)
    neg_score = linkmodel(negative_graph, embeddings, etype)
    link_lossloss = compute_loss1(pos_score, neg_score)
    
    return link_lossloss


















