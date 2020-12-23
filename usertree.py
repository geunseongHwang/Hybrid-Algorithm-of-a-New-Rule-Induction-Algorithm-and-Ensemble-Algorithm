# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

author: hyeongyuy
"""

import pandas as pd
import numpy as np
from functools import reduce
from collections import Counter

import utils as ut
import visgraph as vg
import splitcriterion as sc
import re
class userTree(object):
    def __init__(self, min_samples, max_depth, params, algorithm='paper', simplify = True):
        ### parameters

        #terminate criteria
        self.MIN_SAMPLES = min_samples
        self.MAX_DEPTH = max_depth

        '''
        adaptiveSplitCrit을 쓰는 건 변함이 없음
        '''
        
        # select algorithm(call instance)
        assert algorithm in ['paper', 'adaptive'], 'algorithm = paper or adpative'
        if algorithm == 'paper':
            self.crt = sc.newSplitCrit(self.MIN_SAMPLES, params)
        elif  algorithm == 'adaptive':
            self.crt = sc.adaptiveSplitCrit(self.MIN_SAMPLES, params)

        self.SIMPLIFY = simplify
        self.graph = vg.visGraph()
        
        
    #############################################################################################
    '''
    max_path 부분 추가가 됌
    '''
    def growing_tree(self, data, target_attribute_name, max_path, depth = 1):
        
        '''
        여기를 상세히 파보자
        '''
        
        target_values = data[target_attribute_name]
        ####################
        #해당 노드에 들어 있는 모든 class 값이 동일할 경우 그 값을 반환
        '''
        # cnt_list: 전체 data의 각각 class 수
        # leaf_node_class: 전체 class 중 더 수가 많은 class와 target data의 수 
        '''        
        cnt_list = ut.count_class(target_values.values, self.NUM_CLASSES)
        leaf_node_class = [np.argmax(cnt_list), target_values]
        

        if(depth > self.MAX_DEPTH) or (len(data)==0) or \
            (len(np.unique(target_values.values)) == 1):
            
            # data의 sample수가 일정이하이거나 depth도 일정이하이면 return
            return leaf_node_class, self.graph.node_info(cnt_list, self.N_DATA, max_path=max_path, root=False)

            '''
            max_path_srt 부분이 self.crt.best_split에 추가됌
            '''
        else:
            
            self.crt.DEPTH = depth
            [slt_dtype, best_cut, best_feature, left_sub_data, right_sub_data, max_path_srt] = \
                        self.crt.best_split(data, target_attribute_name)
            
            # 분기 할 변수 없으면 종료
            if best_feature =='':
                
                
                if depth == 1:
                    
                    
                    tree_org = {'Root_node' : leaf_node_class}  
                    
                    leaf_print = {'Root_node' : self.graph.node_info(cnt_list, \
                                                self.N_DATA, max_path='Root', root=True)}
                           
                    return tree_org, leaf_print
                
                else:
 
                    return leaf_node_class, self.graph.node_info(cnt_list, \
                                    self.N_DATA, max_path=max_path, root=False)
                        
            #split(수치 : 범위, 카테고리 : 값)
            condition = ['<', '>='] if slt_dtype =='n' else ['!=', '==']
            path_cond = [str(max_path_srt =='left'), str(max_path_srt == 'right')]
            

            left_subtree, graph_left_subtree = self.growing_tree(left_sub_data, \
                target_attribute_name, max_path= path_cond[0], depth= depth +1)
            
        
            right_subtree, graph_right_subtree = self.growing_tree(right_sub_data,\
            target_attribute_name, max_path= path_cond[1], depth= depth +1)
            
            
            #mk sub tree
            tree = {}
            tree['{} {} {}'.format(best_feature, condition[0], best_cut)] = left_subtree
            tree['{} {} {}'.format(best_feature, condition[1], best_cut)] = right_subtree
            
            
            '''
            max_path 추가
            '''
            
            # 이 부분이 
            

            
            graph_tree = self.graph.get_graph_tree(best_feature, best_cut, cnt_list, condition, \
                                        [graph_left_subtree, graph_right_subtree], max_path)
            

                
        return tree, graph_tree


    def recur_simplify(self):
        bf_rule_list = ut.get_leaf_rule(self.tree, [], [], leaf_info=False)
        tree_rule = ut.get_leaf_rule(self.tree, [], [], leaf_info=True)
        print_tree_rule =ut.get_leaf_rule(self.graph_tree, [], [], leaf_info=True)

        # 분기 후 양쪽 노드의 class가 동일한 규칙 list생성(leaf의 부모노드 제외한 규칙 + 예측 결과가 동일)
        all_rules = [tuple(i[:-2] + [i[-1][0]]) for i in tree_rule]
        all_print_rules = [tuple(i[:-2] + re.findall('label="(.*)\\\\nhomogeneity' , i[-1])) for i in print_tree_rule]
        all_rules_dict = {tuple(i[:-2] + [i[-1][0]]):i for i in tree_rule}
        all_print_rules_dict = {tuple(i[:-2] + re.findall('label="(.*)\\\\nhomogeneity' , i[-1])):i for i in print_tree_rule}
        dup_rule = [all_rules_dict[r] for r, c in  Counter(all_rules).items() if c >=2]
        dup_print_rule = [all_print_rules_dict[r] for r, c in  Counter(all_print_rules).items() if c >=2]
              
        for n, r in enumerate(dup_rule):
            """
            parent node에서 파생 된 두개의 leaf node 예측값이 같은 경우 
            원래 parent node가 leaf node가 되고, 그 상위 node가 parent node가 된다.
            """
            
            new_parent_rule = list(r)[:-2] 
            new_parent_print_rule = list(dup_print_rule[n])[:-2]
            org_parent_print_rule = list(dup_print_rule[n])[-2]
            
            sub_dict = reduce(dict.get, tuple(new_parent_rule), self.tree)
            
            if  isinstance(sub_dict, dict):

                concat_child_df = pd.concat([i[1] for i in sub_dict.values()]) #부모노드까지의 rule과 예측값이 동일한 rule들의 sub set concat
                cnt_list  = ut.count_class(concat_child_df.values, self.NUM_CLASSES)

                if len(new_parent_rule) ==0: #상위노드가 없을 경우 simplify 할 수 없음
                    self.tree = {'Root_node' : [np.argmax(cnt_list), concat_child_df]} # self.setInDict(self.tree , ['Root Node'], [np.argmax(cnt_list),cnt_list])
                    leaf_print= self.graph.node_info(cnt_list, self.N_DATA, max_path='Root', root=True)
                    self.graph_tree = leaf_print 
                    return self.tree, self.graph_tree
                else:
                    
                    self.tree = ut.setInDict(self.tree , new_parent_rule, [np.argmax(cnt_list),concat_child_df])
                    leaf_print= self.graph.node_info(cnt_list, self.N_DATA, re.findall('max_path = (.*)\"' , org_parent_print_rule)[0], root=False)
                    self.graph_tree = ut.setInDict(self.graph_tree, new_parent_print_rule, leaf_print)

                #print(f'leaf_print: {leaf_print}')


        #splify 하고 난 이후에 남은 rule list  ///bf_rule_list와 비교하여 같을 경우 simpliify 안함
        aft_rule_list = ut.get_leaf_rule(self.tree, [], [], leaf_info=False)

        if bf_rule_list  == aft_rule_list :
            return self.tree, self.graph_tree
        else:
            return self.recur_simplify()

    #############################################################################################
    def fit(self, data, target_attribute_name, depth = 1):
        data = data.copy()
        target_values = data[target_attribute_name].values
        elements = np.unique(target_values)
        self.NUM_CLASSES = len(elements)
        self.CLASS_DICT = {i:v for i, v in enumerate(elements)}
        self.CLASS_DICT_ = {v:i for i, v in enumerate(elements)}
        self.N_DATA = len(data)
        data[target_attribute_name] = [self.CLASS_DICT_[v] for v in target_values]
        
        '''
        여기 밑 부분 부터 다름
        '''
        
        self.tree, self.graph_tree = self.growing_tree(data, target_attribute_name, max_path='Root', depth=depth)
        
        if list(self.tree.keys())[0] == 'Root_node':
            return self.tree, self.graph_tree

        if self.SIMPLIFY:
            self.recur_simplify()
        
        return self.tree, self.graph_tree

    ############################################################################################
    def predict(self, test, target_attribute_name='target'):
        test = test.copy()
        target_values = test[target_attribute_name]
        test[target_attribute_name] = [self.CLASS_DICT_[v] for v in target_values]
        
        rule_list = ut.get_leaf_rule(self.tree, [], [], leaf_info=True)
        
        predict_class = pd.DataFrame(columns=["class"], index=test.index)
        predict_prob = pd.DataFrame(columns=[str(i) \
                            for i in range(self.NUM_CLASSES)], index=test.index)

        for rule in rule_list:
            idx, pred = ut.recur_split(test, rule, n_class=self.NUM_CLASSES)
            if len(idx)!=0:
                predict_class.loc[idx, 'class'] = [self.CLASS_DICT[np.argmax(pred)]]
                predict_prob.loc[idx, [str(i) \
                    for i in range(self.NUM_CLASSES)]] = [pred] * len(idx)

        return predict_class, predict_prob