from HDRTF import *
import matplotlib.pyplot as plt

def evaluate(target_gcn_model, AEmodel, linklink_predictor, source_basic_graph, target_basic_graph, source_features, target_features, dec_graph_, edge_label2_):

    target_gcn_model.eval()
    AEmodel.eval()
    linklink_predictor.eval()
    
    
    # Get embeddings for the source and target graphs
    source_embeddings = target_gcn_model(source_basic_graph, source_features)
    target_embeddings = target_gcn_model(target_basic_graph, target_features)
    
    source_embeddings2, target_embeddings2 = {}, {} 
    
    for key in ['cell','target']:
        # Use the autoencoder model to get embeddings
        source_embeddings2[key], source_embeddings[key] = AEmodel(source_embeddings[key])
        target_embeddings2[key], target_embeddings[key] = AEmodel(target_embeddings[key])
        # Also apply the autoencoder for the 'drug' node type
        source_embeddings2['drug'], source_embeddings['drug'] = AEmodel(source_embeddings['drug'])
        target_embeddings2['drug'], target_embeddings['drug'] = AEmodel(target_embeddings['drug'])
    
        # Align the embeddings between source and target for each node type
        source_embeddings[key], target_embeddings[key] = subspace_alignment(source_embeddings[key], target_embeddings[key], 64)
    
    embeddings = {'drug': target_embeddings2['drug'], 'cell': target_embeddings2['cell']}
    # embeddings = {'drug': target_embeddings['drug'], 'cell': target_embeddings['cell']}
    
    # Predict the scores for the target graph
    target_pos_score = linklink_predictor(dec_graph_, embeddings)
    
    # Combine the predicted scores with the true labels
    CC = torch.cat([target_pos_score, edge_label2_.unsqueeze(1)], dim=1)
    CC = CC.detach().numpy()
    
    y_pred = CC[:, 0]
    y_true = CC[:, 1]
    
    # Calculate AUC
    target_auc = roc_auc_score(y_true, y_pred)
    # Calculate AUPR
    target_aupr = average_precision_score(y_true, y_pred)
    print("AUC:", target_auc)
    print("AUPR:", target_aupr)
    
    # Calculate TPR, FPR, and thresholds
    fpr, tpr, thresholds = roc_curve(CC[:, 1], CC[:, 0])
    # Find the threshold that maximizes the Youden's Index
    target_best_threshold = thresholds[(tpr - fpr).argmax()]
    # Calculate accuracy using the best threshold
    target_acc = accuracy_score(CC[:, 1], CC[:, 0] >= target_best_threshold)
    target_precision = precision_score(y_true, y_pred >= target_best_threshold)
    target_f1 = f1_score(y_true, y_pred >= target_best_threshold)
    print("Best Threshold:", target_best_threshold)
    print("Accuracy:", target_acc)
    print("Precision:", target_precision)
    print("F1 Score:", target_f1)
    
    # Separate the predicted scores and true labels
    scores = CC[:, 0]
    labels = CC[:, 1]
    # Use a loop to plot and set legends for each category
    for label in set(labels):
        indices = labels == label
        plt.scatter(range(len(scores[indices])), scores[indices], label=f'Class {int(label)}')
    
    # Set the title and labels for the plot
    plt.title('Predicted Scores by Class')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.legend()
    
    return target_auc, target_aupr, target_best_threshold, target_acc, target_precision, target_f1, y_pred, y_true
