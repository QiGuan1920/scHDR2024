# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:08:09 2024

@author: 78760
"""
from HDRTF import *
import matplotlib.pyplot as plt



def evaluate(target_gcn_model, AEmodel, linklink_predictor, source_basic_graph, target_basic_graph, source_features, target_features, dec_graph_, edge_label2_):

    target_gcn_model.eval()
    AEmodel.eval()
    linklink_predictor.eval()
    
    
    source_embeddings = target_gcn_model(source_basic_graph, source_features)
    target_embeddings = target_gcn_model(target_basic_graph, target_features)
    
    
    source_embeddings2, target_embeddings2 = {}, {} 
    
    for key in ['cell','target']:
       source_embeddings2[key], source_embeddings[key] = AEmodel(source_embeddings[key])
       target_embeddings2[key], target_embeddings[key] = AEmodel(target_embeddings[key])
       source_embeddings2['drug'],source_embeddings['drug'] = AEmodel(source_embeddings['drug'])
       target_embeddings2['drug'],target_embeddings['drug'] = AEmodel(target_embeddings['drug'])
    
       source_embeddings[key], target_embeddings[key] = subspace_alignment(source_embeddings[key], target_embeddings[key], 64)
    
    
               
    embeddings = {'drug': target_embeddings2['drug'], 'cell': target_embeddings2['cell']}
    # embeddings = {'drug': target_embeddings['drug'], 'cell': target_embeddings['cell']}
    
    target_pos_score = linklink_predictor(dec_graph_, embeddings)
    
    CC=torch.cat([target_pos_score, edge_label2_.unsqueeze(1)],dim=1)
    CC=CC.detach().numpy()
    
    y_pred = CC[:, 0]
    y_true = CC[:, 1]
    # Calculate AUC
    target_auc = roc_auc_score(y_true, y_pred)
    # Calculate AUPR
    target_aupr = average_precision_score(y_true, y_pred)
    print("AUC:", target_auc)
    print("AUPR:", target_aupr)
    # print("F1 Score:", f1)
    # Calculate TPR, FPR 和Threshold
    fpr, tpr, thresholds = roc_curve(CC[:, 1], CC[:, 0])
    # 找到最大约登指数的Threshold作为最佳Threshold
    target_best_threshold = thresholds[(tpr - fpr).argmax()]
    # 根据最佳ThresholdCalculateAccuracy
    target_acc = accuracy_score(CC[:, 1], CC[:, 0] >= target_best_threshold)
    target_precision = precision_score(y_true, y_pred>= target_best_threshold)
    target_f1 = f1_score(y_true, y_pred>= target_best_threshold)
    print("Best Threshold:", target_best_threshold)
    print("Accuracy:", target_acc)
    print("precision:", target_precision)
    print("F1 Score:", target_f1)
    
    
    
    # 分离预测Score和真实Label
    scores = CC[:, 0]
    labels = CC[:, 1]
    # 使用循环为每个类别绘制并设置图例
    for label in set(labels):
        indices = labels == label
        plt.scatter(range(len(scores[indices])), scores[indices], label=f'Class {int(label)}')
    
    plt.title('Predicted Scores by Class')
    plt.xlabel('Sample Index')
    plt.ylabel('Score')
    plt.legend()
    
    return target_auc, target_aupr, target_best_threshold, target_acc, target_precision, target_f1, y_pred, y_true


































