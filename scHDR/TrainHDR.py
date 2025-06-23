# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 12:51:11 2024

@author: 78760
"""
import pandas as pd
from HDRTF import *
from Preprocessing import lossloss, process
from Evaluate import evaluate



def Train(S):
    results = pd.DataFrame(columns=['N', 'AUC', 'AUPR', 'Threshold', 'Accuracy', 'Precision', 'F1'])
    i = 0
    
    while i < 3:  # 限制重试次数，避免无限循环
        try:
            Process = process(S)
            all_attributes = dir(Process)
            class_attributes = [attr for attr in all_attributes if not callable(getattr(Process, attr)) and not attr.startswith("__")]
            for attr in class_attributes:
                globals()[attr] = getattr(Process, attr)

            target_gcn_model = Process.target_gcn_model
            AEmodel = Process.AEmodel
            link_predictor = Process.link_predictor
            linklink_predictor = Process.linklink_predictor
            # 你现有的代码...
            
            for epoch in range(10):
                for fold in range(5):
                    
                    target_gcn_model.train()
                    AEmodel.train()
                    link_predictor.train()
                    linklink_predictor.train()
                    
                    source_embeddings = target_gcn_model(source_basic_graph, source_features)
                    target_embeddings = target_gcn_model(target_basic_graph, target_features)
            
                    
                    source_embeddings2, target_embeddings2 = {}, {} 
                    
                    for key in ['cell','target']:
                       source_embeddings2[key], source_embeddings[key] = AEmodel(source_embeddings[key])
                       target_embeddings2[key], target_embeddings[key] = AEmodel(target_embeddings[key])
                       source_embeddings2['drug'],source_embeddings['drug'] = AEmodel(source_embeddings['drug'])
                       target_embeddings2['drug'],target_embeddings['drug'] = AEmodel(target_embeddings['drug'])
            
                       source_embeddings[key], target_embeddings[key] = subspace_alignment(source_embeddings[key], target_embeddings[key], 64)
            
                    source_loss1 = lossloss(link_predictor, source_basic_graph, 1, source_embeddings, etype=('drug', 'DT', 'target'))
                    source_loss2 = lossloss(link_predictor, source_basic_graph, 1, source_embeddings, etype=('cell', 'CT', 'target'))
                    source_loss3 = lossloss(link_predictor, source_basic_graph, 1, source_embeddings, etype=('cell', 'CC1', 'cell'))
                    source_loss4 = lossloss(link_predictor, source_basic_graph, 1, source_embeddings, etype=('target', 'TT1', 'target'))
                    target_loss1 = lossloss(link_predictor, target_basic_graph, 1, target_embeddings, etype=('drug', 'DT', 'target'))
                    target_loss2 = lossloss(link_predictor, target_basic_graph, 1, target_embeddings, etype=('cell', 'CT', 'target'))
                    target_loss3 = lossloss(link_predictor, target_basic_graph, 1, target_embeddings, etype=('cell', 'CC1', 'cell'))
                    target_loss4 = lossloss(link_predictor, target_basic_graph, 1, target_embeddings, etype=('target', 'TT1', 'target'))
                    
                    linkloss = source_loss1+source_loss2+source_loss3+source_loss4+target_loss1+target_loss2+target_loss3+target_loss4
            
            
                    
                    test_mask, train_mask = Fold(folds, fold)
                    # train_g2, test_g2, train_labels2, test_labels2 = Fold2(fold)
                    
                    embeddings = {'drug': source_embeddings2['drug'], 'cell': source_embeddings2['cell']}
                    source_test = linklink_predictor(dec_graph, embeddings)[train_mask]
                    source_link_loss = compute_loss2(source_test.squeeze(), edge_label2[train_mask])
                    
                    total_loss = source_link_loss + linkloss
                
                    # 2
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    print(f"N {i}, Epoch {epoch+1}/{10}, fold {fold+1}/{5}, Link label Loss: {source_link_loss.item()}, Link link Loss: {linkloss.item()}")
                    
                    linklink_predictor.eval()
                    embeddings = {'drug': source_embeddings2['drug'], 'cell': source_embeddings2['cell']}
                    target_pos_score = linklink_predictor(dec_graph, embeddings)[test_mask]
            
                    CC=torch.cat([target_pos_score, edge_label2[test_mask].unsqueeze(1)],dim=1)
                    CC=CC.detach().numpy()
                    y_pred = CC[:, 0]
                    y_true = CC[:, 1]
                    # 计算 AUC
                    auc = roc_auc_score(y_true, y_pred)
                    # 计算 AUPR
                    aupr = average_precision_score(y_true, y_pred)
                    print("AUC:", auc)
                    print("AUPR:", aupr)
                    # print("F1 Score:", f1)
                    # 计算 TPR, FPR 和阈值
                    fpr, tpr, thresholds = roc_curve(CC[:, 1], CC[:, 0])
                    # 找到最大约登指数的阈值作为最佳阈值
                    best_threshold = thresholds[(tpr - fpr).argmax()]
                    # 根据最佳阈值计算准确率
                    acc = accuracy_score(CC[:, 1], CC[:, 0] >= best_threshold)
                    precision = precision_score(y_true, y_pred>= best_threshold)
                    f1 = f1_score(y_true, y_pred>= best_threshold)
                    print("Best Threshold:", best_threshold)
                    print("Accuracy:", acc)
                    print("precision:", precision)
                    print("F1 Score:", f1)
                    
                    
            target_auc, target_aupr, target_best_threshold, target_acc, target_precision, target_f1, y_pred, y_true = evaluate(target_gcn_model, AEmodel, linklink_predictor, source_basic_graph, target_basic_graph, source_features, target_features, dec_graph_, edge_label2_)
    
            if i==0:
                best_auc, best_aupr, best_best_threshold, best_acc, best_precision, best_f1 = target_auc, target_aupr, target_best_threshold, target_acc, target_precision, target_f1
                
                results = results._append({'N': i, 'AUC': target_auc,'AUPR': target_aupr,'Threshold': target_best_threshold,'Accuracy': target_acc,'Precision': target_precision,'F1': target_f1}, ignore_index=True)
                scores_df = pd.DataFrame({f'y_pred_{i}': y_pred, f'y_true_{i}': y_true})
                
                # torch.save(target_gcn_model.state_dict(), '../save_code14测试/'+S+'_target_gcn_model.pth')
                # torch.save(AEmodel.state_dict(), '../save_code14测试/'+S+'_AEmodel.pth')
                # torch.save(linklink_predictor.state_dict(), '../save_code14测试/'+S+'_linklink_predictor.pth')
                
                results.to_csv('结果1/'+S+'_results.csv', index=False)
                scores_df.to_csv('结果1/'+S+'_scores.csv', index=False)
                
            else:
                results = results._append({'N': i,'AUC': target_auc,'AUPR': target_aupr,'Threshold': target_best_threshold,'Accuracy': target_acc,'Precision': target_precision,'F1': target_f1}, ignore_index=True)
                score_df = pd.DataFrame({f'y_pred_{i}': y_pred, f'y_true_{i}': y_true})
                scores_df = pd.concat([scores_df, score_df], axis=1)
                
                if (target_auc > best_auc) & (target_aupr > best_aupr):
                    best_auc, best_aupr, best_best_threshold, best_acc, best_precision, best_f1 = target_auc, target_aupr, target_best_threshold, target_acc, target_precision, target_f1
                    
                    # torch.save(target_gcn_model.state_dict(), '../save_code14测试/'+S+'_target_gcn_model.pth')
                    # torch.save(AEmodel.state_dict(), '../save_code14测试/'+S+'_AEmodel.pth')
                    # torch.save(linklink_predictor.state_dict(), '../save_code14测试/'+S+'_linklink_predictor.pth')
                    
                results.to_csv('结果1/'+S+'_results.csv', index=False)
                scores_df.to_csv('结果1/'+S+'_scores.csv', index=False)

            i += 1  # 为下一次迭代增加 i
        except Exception as e:
            print(f"在第 {i + 1} 次迭代中发生错误：{e}")
            # 如果需要，可以在此处添加额外的错误处理或记录
            # 如果要重试，则不增加 i，以便重试相同的 i 值

    return results, scores_df  # 根据需要返回结果

















