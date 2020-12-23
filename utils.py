# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

@author: hyeongyuy
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, \
precision_score, f1_score
import copy
from functools import reduce  
import operator
import pickle
import pandas as pd
from collections import Counter
import re
from sklearn.metrics import confusion_matrix

def count_class(class_idx_att, n_class):
    return [sum(class_idx_att == i) for i in range(n_class)]

def setInDict(dataDict, mapList, value):
    """
    https://python-decompiler.com/article/2013-02/
    access-nested-dictionary-items-via-a-list-of-keys
    """
    def getFromDict(dataDict, mapList):
        return reduce(operator.getitem, mapList, dataDict)
    temp_dict = copy.deepcopy(dataDict)
    getFromDict(temp_dict, mapList[:-1])[mapList[-1]] = value
    return temp_dict

#model read & write
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_leaf_rule(tree_dict, rule_list, rule, leaf_info):
    tree = copy.deepcopy(tree_dict)
    k_list = list(tree.keys())

    if k_list[0] != 'Root_node':
        left, right = k_list[0], k_list[1]
        for direct in [left, right]:
            
            # isinstance(tree[direct], dict): tree[direct]가 dict형식인지 봄 -> True, False 
            if not isinstance(tree[direct], dict):
                
                # leaf_info: True
                if leaf_info:
                    rule_list.append(rule + [direct] + [tree[direct]])
                    
                # leaf_info: False
                else:
                    rule_list.append(rule + [direct])
            else:
                get_leaf_rule(tree[direct], rule_list, rule + [direct], leaf_info=leaf_info)
        return rule_list
    else:
        return [[k_list[0]] + [tree[k_list[0]]]]



def perform_check(real, pred, prob, n_class, val_idx_dict, average = 'macro'):
    """
    #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    #sklearn.metrics.roc_auc_score
    """
    result = []
    real = pd.DataFrame(real, columns=['target'])
    real = real.loc[pred.index, : ]
        
    try:
        # val_idx_dict: {0: 0, 1: 1}
        # real_v_num: 0 or 1 -> 1d array
        # pred_v_num: 0 or 1 -> 1d array
        real_v_num = np.array([val_idx_dict[v] for v in real.values.reshape(-1)])
        pred_v_num = np.array([val_idx_dict[v] for v in pred.values.reshape(-1)])
        
    except AttributeError:
        real_v_num = np.array([val_idx_dict[v] for v in real])
        pred_v_num = np.array([val_idx_dict[v] for v in pred])

    if n_class == 2:
        accr = accuracy_score(real_v_num, pred_v_num)
        recall = recall_score(real_v_num, pred_v_num)
        precision = precision_score(real_v_num, pred_v_num)
        f1 = f1_score(real_v_num, pred_v_num)
        
        if len(np.unique(real_v_num)) == 2:
            auc = roc_auc_score(real_v_num, pred)
            result = accr, recall, precision, f1, auc
        
        else:
            result = accr, recall, precision, f1
        
        confusion_mat = confusion_matrix(real_v_num, pred_v_num, labels=[0,1])
        
        correct_label_num = len(np.in1d(real_v_num,pred_v_num)[np.in1d(real_v_num,pred_v_num)])
        
        
    else:
        prob_v = prob.astype(float)
        accr = accuracy_score(real_v_num, pred_v_num)
        recall = recall_score(real_v_num, pred_v_num, average = average, \
                            labels = np.unique(real_v_num)) 
        precision = precision_score(real_v_num, pred_v_num, \
                            average = average, labels = np.unique(real_v_num))
        f1 = f1_score(real_v_num, pred_v_num, average = average, \
                            labels =  np.unique(real_v_num))
        real_v_dummy = []
        
        for v in real:
            zeros = np.zeros(n_class) 
            zeros[val_idx_dict[v]] = 1
            real_v_dummy.append(zeros)
        real_v_dummy = np.array(real_v_dummy)
        #auc = roc_auc_score(real_v_dummy, prob_v, average = average)
    
    return result, confusion_mat, correct_label_num
    
# 쓰인다 -> 변화없음 #
def recur_split(test, split_rule_list, idx=0, n_class=0):
    df= copy.deepcopy(test)
    cont_cond = ['>=', '<']
    cat_cond = ['==', '!=']
    if split_rule_list[0] == 'Root_node':
        print("""Untrained model.(Only root node.)
        The index for the input data and the ratio value
        for each class of the target variable are returned.""")
        if n_class == 0:
            return df.index
        
        else:
            cnt_list = np.array(count_class(split_rule_list[-1][-1], n_class))
            pred_value = cnt_list/sum(cnt_list)
            return df.index, pred_value
    
    if idx == len(split_rule_list) -1:
        if n_class == 0:
            return df.index
        
        else:
            cnt_list = np.array(count_class(split_rule_list[-1][-1], n_class))
            pred_value = cnt_list/sum(cnt_list)
            return df.index, pred_value
            
    else:
        att, cond, value = split_rule_list[idx].split()
        if cond == cont_cond[0]:
            sub_set = df.loc[df[att] >= float(value),:]
        elif cond == cont_cond[1] :
            sub_set = df.loc[df[att] < float(value),:]
        elif cond == cat_cond[0]:
            sub_set = df.loc[df[att] == value,:]
        else:
            sub_set = df.loc[df[att] != value,:]
        return recur_split(sub_set, split_rule_list, idx + 1, n_class=n_class)

def get_usrt_info(df, tree_ins, tree_model, cut_depth, target_att= 'target'):

    tree_rule = get_leaf_rule(tree_model, [], [], leaf_info = True)
    if cut_depth == -1:
        info_list =  []
        for s_rule in tree_rule:
            sidx = recur_split(df, s_rule)
            node_df = df.loc[sidx,:]
            depth = len(s_rule)
            if len(node_df) != 0:
                #If there is no value that satisfies this rule, it is ignored.
                simple_max_prob = max(Counter(node_df[target_att]).values())/len(node_df)
                sample_ratio = len(node_df)/len(df)
                info_list.append([depth, simple_max_prob, sample_ratio])
        return pd.DataFrame(info_list, \
                columns=['depth', 'max_prob', 'sample_ratio']).sort_values('depth')
    
    else:
        max_simple_rules = [i for i in tree_rule if len(i[:-1]) <= cut_depth]
        simple_idx = []
        for s_rule in max_simple_rules:
            sidx = recur_split(df, s_rule)
            simple_idx.extend(list(sidx))
        
        N_simple_rule = len(max_simple_rules)
        simple_df = df.loc[simple_idx,:]
        simple_max_prob = np.mean(np.max(tree_ins.predict(simple_df, tree_model)[1],1))
        if simple_max_prob is np.nan:
            simple_max_prob = 0
        
        simple_ratio = len(simple_df)/len(df)
        if tree_rule[0] == 'Root_node':
            return 0,0,0
        else:
            return simple_ratio, simple_max_prob, N_simple_rule
    
def sk_cart_prune(tree):
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]

    leaves = [(ls[i] == rs[i]) for i in nodes]

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
    return tree

def get_CART_info(estimator, df, cut_depth):
    df = df.reset_index(drop=True)
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    sample_node_id = estimator.apply(df)
    
    stack = [(0, -1)]  
    leaf_depth_dict = {}
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            leaf_depth_dict[node_id] = parent_depth+1
    
    info_list =  []
    for node_id in sample_node_id:
        depth = leaf_depth_dict[node_id]
        info_list.append([depth, node_id])
        
    info_df = pd.DataFrame(info_list, \
                           columns=['depth', 'node_id'])
    
    if cut_depth == -1:
        info_df['max_prob'] = np.max(estimator.predict_proba(df), 1)
        return info_df.drop_duplicates().loc[:,['depth', 'max_prob']].sort_values('depth')


    simple_info_df = info_df.loc[info_df.depth <= cut_depth,]
    simple_df = df.loc[simple_info_df.index,]
    
    if len(simple_df) == 0:
        return 0,0,0
    else:
        simple_ratio = len(simple_df) / len(df)
        simple_max_prob = np.mean(np.max(estimator.predict_proba(simple_df), 1))
        N_simple_rule = len(np.unique(simple_info_df.node_id))
        return simple_ratio, simple_max_prob, N_simple_rule


def mod_cart_graph(dot_data, n_data, value='gini'):
    dot_data = dot_data.replace(value, 'predict')
    predict_list = re.findall('predict = (.*)\\\\', dot_data)
    sample_list = [s.replace('\\n', ', ') for s in re.findall('value = \[(.*)\]"', dot_data)]
    num_sample_list = [[int(v) for v in samples.split(', ')] for samples in sample_list]  
    predict_v = [np.argmax(s) for s in num_sample_list]
    homog_v = [np.round(max(s)/sum(s), 3) for s in num_sample_list]
    
    for i, pred in enumerate(predict_list):
        dot_data=dot_data.replace(pred, str(predict_v[i]) \
              + '\\nhomogeneity = ' + str(homog_v[i]) + '\\' + '\\'.join(pred.split('\\')[1:]))
         
    for i in re.findall('label=(.*)\]"', dot_data):
        if i[1:8] != 'predict' :
            #print([i.split('\\')[1]] + i.split('\\')[2:])
            dot_data = dot_data.replace(i, '\\'.join([i.split('\\')[0]] + i.split('\\')[3:]))
        else:
            samp = i.split('\\nvalue')[0].split('\\n')[2]
            if samp[:7] == 'samples':
                dot_data = dot_data.replace(samp, 'coverage = ' + str(np.round(int(samp.split(' = ')[1])/n_data,3)))
    dot_data = dot_data.replace('value', 'samples/class')
    dot_data = dot_data.replace(', headlabel=\"True\"', '')
    dot_data = dot_data.replace(', headlabel=\"False\"', '')
    return dot_data


def get_cart_info(df, tree_model , target_att= 'target'):
    
    n_data=len(df)
    elements = np.unique(df[target_att])
    NUM_CLASSES = len(elements)
    
    # original y의 class 수 #
    uni_class = np.unique(df[target_att])

    class_number = {}
    for i in uni_class:    
        class_number[i] = len(df[df[target_att] == i])
    
    cnt_list_df = list(class_number.values())
    cnt_list_df = list(map(int, cnt_list_df))

    class_prior = [cnt_list_df[i]/n_data for i in range(len(cnt_list_df))]
    print(f'class prior: {class_prior}')
    
    tree_rule = get_leaf_rule(tree_model, [], [], leaf_info = True)
 
    info_list =  []
    for s_rule in tree_rule:

        sidx = recur_split(df, s_rule)
        
        ind_cla,_ = pd.factorize(df[target_att])
        del df[target_att]
        df[target_att] = ind_cla
        
        node_df = df.loc[sidx,:]
        
        node_df = node_df.sort_index()
        
        cnt_list = ut.count_class(node_df[target_att].values, NUM_CLASSES)
        print(f'leaf node별 class수: {cnt_list}')
        
        
        depth = len(s_rule)-1
        if len(node_df) != 0 and depth <= 5:
            
            #If there is no value that satisfies this rule, it is ignored.
            pred = np.argmax(cnt_list)
            homo = np.round(max(cnt_list)/sum(cnt_list),3)
            cover = np.round(sum(cnt_list)/n_data,3)

            lift = np.round(homo / class_prior[pred], 4)
            var_num = depth
            
            info_list.append([pred, depth, homo, lift, cover, var_num])

    fin = pd.DataFrame(info_list, columns=['pred','depth','homogeneity', 'lift', 'coverage', 'number_of_variable']).sort_values(['depth'])
    
    fin_df = pd.DataFrame()
    for depth in np.unique(fin['depth']):
        temp_df = fin[fin['depth'] == depth]
        temp_df = temp_df.sort_index()
        if len(fin_df) == 0:
            fin_df = temp_df
        else:
            fin_df = pd.concat([fin_df,temp_df], axis=0)
        
        fin_df = fin_df.reset_index(drop=True)
        
        
        
    return fin_df