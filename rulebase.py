# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

author: hyeongyuy
"""

import pandas as pd
import numpy as np
from functools import reduce
#from collections import Counter
import graphviz
import re

#import copy
import utils as ut
#import visgraph as vg
#import splitcriterion as sc
from usertree import userTree as utr


class ruleBase(object):
    def __init__(self, min_samples, max_depth, params, algorithm='paper', simplify = True):
        #terminate criteria
        '''
        MIN_SAMPLES:  종료 기준, 분기 전 시점 최소 샘플 수
        MAX_DEPTH: 논문에 나와있는 조건을 만족하는 depth
        ALGORITHM: 'paper' or 'adaptive'
        SIMPLIFY: 자식노드가 부모노드의 예측값과 같을 때 합칠 것인지 말 것인지 결정
        params: paper version - lambda값, adative version - lambda 범위 값(array형태)
        '''
        
        self.MIN_SAMPLES = min_samples
        self.MAX_DEPTH = max_depth
        self.ALGORITHM = algorithm
        self.SIMPLIFY = simplify
        self.params = params
        
    ############################################################################################
    def fit(self, df_org, df_g, target_attribute_name='target', output_graph = False, save_dir='save_dir', d_set='datanm', rule_rate=0.7):
        df = df_org.copy()
        df_g= df_g.copy()
        
        elements = np.unique(df[target_attribute_name].values)
        self.NUM_CLASSES = len(elements)
        self.CLASS_DICT = {i:v for i, v in enumerate(elements)}
        self.CLASS_DICT_ = {v:i for i, v in enumerate(elements)}
        self.N_DATA = len(df_org)
        self.df_total_inf_base = pd.DataFrame()
        
        self.rule_model= {}
        self.analy_sample = {}
        
        rule_idx = 0
        rule_idx_ = 1
        used_idx = []
        
        # train set 수 모으기 #
        self.rule_tra_ind_num = 0 
        
        # rule 말고 나머지 데이터 뽑기용 #
        train_data = df_org.copy()
        self.others_df = pd.DataFrame()
        
        # rule 생성 rate #
        self.rule_rate = rule_rate
        testtest = []
        while True:
            ''' 
            ※ 핵심 ※
            # not_used_idx: max_depth=True인 sample을 제거하고 남은 것들(index)
            # used_idx: max_depth로 쓰여진 애들
            
            '''
            '''
            class_number = {}
            for i in uni_class:    
                class_number[i] = len(df[df[target_att] == i])
                '''
                
            not_used_idx = list(set(df.index) - set(used_idx))
            df  = df.loc[not_used_idx, ]      
            df_g  = df_g.loc[not_used_idx, ]  
        
            tree_ins = utr(self.MIN_SAMPLES, self.MAX_DEPTH, params=self.params, \
                           algorithm=self.ALGORITHM, simplify = self.SIMPLIFY)
            
            if output_graph:    
                
                tree, _ = tree_ins.fit(df, target_attribute_name = "target")
                tree_ins_ = utr(self.MIN_SAMPLES, self.MAX_DEPTH, params=self.params, \
                           algorithm=self.ALGORITHM, simplify = self.SIMPLIFY)
                
                ###
                _, graph_tree = tree_ins_.fit(df_g, target_attribute_name = "target")
                dot_data_gr= tree_ins.graph.tree_to_graph(tree_ins_.graph_tree)
                tree_graph_ = graphviz.Source(dot_data_gr)
                
                tree_graph_.render('{}/{}/DT_graph_{}_rule_{}_{}'.format(save_dir,f'rule_graph_{rule_rate}',d_set,rule_idx,rule_rate))

            else:
                
                tree, graph_tree = tree_ins.fit(df, target_attribute_name = "target")
                #dot_data= tree_ins.graph.tree_to_graph(tree_ins.graph_tree)
                #tree_graph = graphviz.Source(dot_data)
                
            if list(tree.keys())[0] == 'Root_node':
                
                target_values= np.array([self.CLASS_DICT_[v] for v in df[target_attribute_name].values])
                cnt_list = np.array(ut.count_class(target_values, self.NUM_CLASSES))
                pred_prob = cnt_list/np.sum(cnt_list)
                self.rule_model[-1] = [(np.argmax(pred_prob), pred_prob, len(not_used_idx)/self.N_DATA)]
                break
            
            '''
            기존 분석용
            '''
            tree_rule = ut.get_leaf_rule(tree, [], [], leaf_info=True)
            graph_tree_rule = ut.get_leaf_rule(graph_tree, [], [], leaf_info=True)
            
            #print(f'tree_rule: {tree_rule}')
            #print(f'graph_tree_rule: {graph_tree_rule}')
            
            each_rule_list =[]
#            if (list(tree.keys())[0] == 'Root_node') or (len(df)< self.MIN_SAMPLES):
#                target_values= np.array([self.CLASS_DICT_[v] for v in df[target_attribute_name].values])
#                cnt_list = np.array(ut.count_class(target_values, self.NUM_CLASSES))
#                pred_prob = cnt_list/np.sum(cnt_list)
#                self.rule_model[-1] = [(np.argmax(pred_prob), pred_prob, len(not_used_idx)/self.N_DATA)]
#                break
            
            for tr_list, gtr_list in zip(tree_rule, graph_tree_rule):
                
                temp= []
                each_rule_list=[]
                max_path_list = []
                
                for tr, gtr in zip(tr_list[:-1], gtr_list[1:]):
                    
                    # re.findall: string에서 pattern을 만족하는 문자열을 리스트로 반환

                    mxp_str = re.findall('max_path = (.*)\"' , gtr)[0]
                    #print(f'gtr: {gtr}')
                    
                    max_path = True if mxp_str in ['True', 'Root'] else False
                    max_path_list.append(max_path)
                    temp.append((max_path,mxp_str ))
                    
                    if not max_path:
                        each_rule_list  = []
                        break
                    
                    direction = False if  tr.split()[1] in ['<', '!='] else True
                    tr = tr.replace('<', '>=').replace('!=', '==')
                    
                    # round 작업 #
                    tr_spt = tr.split()                    
                    tr_spt_val = np.round(float(tr_spt[-1]), 4)
                    tr_spt[-1] = tr_spt_val
                    tr = " ".join(map(str, tr_spt))
                    
                    each_rule_list.append((tr, direction))
                    
                # 기존대로 돌리되 class 1에 대한 룰을 생성하면
                # 그 rule에 대한 index만 제거하고 다시 기존대로 모델을 생성 -> 여기에 대한 코드1
                # class 1에 대한 비율이 어느정도 다 걸려졌다면 -> 여기에 대한 코드2
                # class 0에 대한 룰을 생성함
                
                if len(max_path_list) == sum(max_path_list):
                    #print(f'max_path_list: {max_path_list}')
                    #print(f'each_rule_list: {each_rule_list}')

                    used_idx = tr_list[-1][-1].index
                    cnt_list = np.array(ut.count_class(tr_list[-1][-1].values, self.NUM_CLASSES))
                    pred_prob = cnt_list/sum(cnt_list)
                    
                    homogeneity = np.round(max(cnt_list)/sum(cnt_list),3)
                    train_index = len(tr_list[-1][-1])

                    #homogeneity >= 0.8 -> 뺌
                    if np.abs(rule_idx - rule_idx_) == 1 or rule_idx == 0:
                        print(homogeneity)
                        
                        if self.rule_tra_ind_num < np.round(self.N_DATA * rule_rate):
                            
                            
                            each_rule_list.append((tr_list[-1][0], pred_prob, round(sum(cnt_list)/self.N_DATA, 3)))
                            self.rule_model[rule_idx] = each_rule_list 
                            
                            
                            #train_coverage = each_rule_list[1][2]
                            
                            rule_idx_ += 1
                            
                            ############################################
                            # homogeneity와 train index를 저장하는 부분 #
                            ############################################
                            
                            homo_val = pd.DataFrame(data=homogeneity, columns = ["rule_%d"%rule_idx],  index = ['Homogeneity'])
                            tra_ind = pd.DataFrame(data=train_index, columns = ["rule_%d"%rule_idx],  index = ['The_number_of_train_index'])
                            #tra_cov = pd.DataFrame(data=train_coverage, columns = ["rule_%d"%rule_idx],  index = ['Train_coverage'])
                            
                            self.rule_tra_ind_num += train_index
                            
                            data_frames = [homo_val, tra_ind.astype(float)]
                            
                            df_merged = reduce(lambda left,right: pd.merge(left,right,on=["rule_%d"%rule_idx],
                                                how='outer', left_index=True, right_index=True), data_frames)
    
                            self.df_total_inf_base = pd.concat([self.df_total_inf_base, df_merged], axis=1)
                            
                            ###############################
                            # rule로 쓰이지 못한 data 추출 #                            
                            ###############################
                            
                            rule_label = tr_list[-1][-1]
                            temp_df = train_data[np.in1d(train_data.index, rule_label.index) == True]
                            self.others_df = pd.concat([self.others_df, temp_df])
                            
                            #########################
                            # rule별 attribute 추출 #
                            #########################
                                                        
                    else:
                        rule_idx_ += 100
                        
                    '''
                    each_rule_list.append((tr_list[-1][0], pred_prob, round(sum(cnt_list)/self.N_DATA, 3)))
                    self.rule_model[rule_idx] = each_rule_list
                    print(self.rule_model[rule_idx])
                    '''
             
            rule_idx +=1
            del tree_ins    
        
        '''
        rule로 못 만든 나머지 data
        '''
        
        self.others_df = train_data[np.in1d(train_data.index, self.others_df.index)== False]
        
        ##############################
        # 마지막 룰이 아닌 것을 제외한 #
        ##############################
            
        # 1. 룰의 attribute 수
        # 2. rule의 predict class
        
        rule_rule = pd.DataFrame(index=['Rule_predict','The_number_of_rule_attribute','Train_coverage', 'Train_cumulative_coverage'])
        
        cov_cum_list = {}
            
        for key, value in self.rule_model.items():
            
            if key != -1:
                
                # coverage_cumlative 과정
                val = float(value[-1][2])
                cov_cum_list[key] = val
                cov_cum = sum(list(cov_cum_list.values()))
                
                rule_rule.loc['Rule_predict','rule_%d'%key] = value[-1][0]
                rule_rule.loc['The_number_of_rule_attribute','rule_%d'%key] = len(value[:-1])
                rule_rule.loc['Train_coverage','rule_%d'%key] = value[-1][2]
                rule_rule.loc['Train_cumulative_coverage','rule_%d'%key] = cov_cum  
                
        self.df_total_inf_base = pd.concat([self.df_total_inf_base, rule_rule], axis=0)
        
        return self.rule_model
        
    ############################################################################################
    def predict(self, org_df, method='average'):
        assert method in ['average', 'priority'], 'method : average or priority'
        org_df = org_df.copy()
        org_df_idx = list(org_df.index)
        assert len(self.rule_model.keys()) > 1 , 'rule_model is not defined, rule_model : {} '.format(self.rule_model)
        predict_class = pd.DataFrame(columns=["class"], index=org_df.index)
        predict_prob = pd.DataFrame(columns=[str(i) \
                                             for i in range(self.NUM_CLASSES)], index=org_df.index).fillna(0)
        
        predict_info =  pd.DataFrame(columns=["pred_rule", 'coverage', 'count'], index=org_df.index).fillna('')
        predict_info['count'] = 0
            
        assigned_idx = []
        rule_index = {}

        # 추가 3
        df_sub_set_idx_num = pd.DataFrame(data=np.nan, columns=self.df_total_inf_base.columns, index=['The_number_of_test_index'])
        
        df_other = pd.DataFrame(data=np.nan, columns=['others'], index=['The_number_of_test_index'])
        
        # rule_nm: rule_number ex) 0,1,2...
        # sorted(self.rule_model.keys()) : [-1, 0, 1, 2, 3, 4, 5, 6]
        for rule_nm in sorted(self.rule_model.keys())[1:]:
            
            #print(self.rule_model.keys())
            rule_list = self.rule_model[rule_nm]
            
            df = org_df
            
            # rule: max_depth=True인 split criterion들
            # max_path: True or False(split criterion 조건)
            # rule_list[:-1]: ex) [('Uniformity_of_Cell_Size >= 3.5', False), ('Bare_Nuclei == 1', True)]
            for rule, max_path in rule_list[:-1]: 
                
                # att: attribute, cond: condition, value: numeric or category
                # ex)['Uniformity_of_Cell_Size', '>=', '3.5']
                att, cond, value = rule.split()
                
                if cond == '>=':
                    # numeric data - train data로 생성된 rule들에 test data를 
                    #                넣고 조건에 맞는 data의 index만 뽑음 #
                    if max_path:
                        sub_set_idx = df.loc[df[att] >= float(value),:].index
                        
                    else:
                        sub_set_idx = df.loc[df[att] < float(value),:].index
                    
                    #print(sub_set_idx, len(sub_set_idx))
                elif cond == '==':
                    # categorical data #
                    if max_path:
                        sub_set_idx = df.loc[df[att] == value,:].index
                    else:
                        sub_set_idx = df.loc[df[att] != value,:].index
                        
                    #print(sub_set_idx, len(sub_set_idx))
                else:
                    # 위와 같은 조건이 아니면 잘못된 것을 알려줌 #
                    assert False, 'rule format is wrong'
                
                df = df.loc[sub_set_idx,:]
                
                
                
                
            if method=='priority':
                
                # sub_set_idx_: 기존 data에서 분기된 data를 성능을 측정하고 그대로 뺀 index #
                sub_set_idx_ = list(set(list(sub_set_idx)) - set(list(assigned_idx)))
                
                ''' 
                ionosphere데이터는 아래 조건문을 통과못하는 index때문에 문제 발생
                rule마다 index가 겹치는 문제를 해결하기 위해 기존에 사용되었던 인덱스를
                계속 제거하고 rule index가 중복이 되지 않게끔 설계.
                그래서 중복된 rule은 제거하고 분석
                '''
                
                if len(sub_set_idx_) == 0:
                    continue
                
                # predict_prob: 전체 데이터를 0,1의 확률값으로 표현#
                predict_prob.loc[sub_set_idx_, [str(i) for i in range(self.NUM_CLASSES)]] = [rule_list[-1][1]] * len(sub_set_idx_)
                #print(predict_prob)
                
                # predict_class: 전체 데이터를 예측된 class로 표현#
                predict_class.loc[sub_set_idx_,:] = [rule_list[-1][0]]
                #print(predict_class)
                
            else:
                # 이건 average일 때임 #
                sub_set_idx_ = sub_set_idx
                predict_prob.loc[sub_set_idx_, [str(i) for i in range(self.NUM_CLASSES)]] += np.array([rule_list[-1][1]] * len(sub_set_idx_))
                
            #print(len(sub_set_idx_))
            # predict_info: data index마다 pred_rule, coverage, count를 매겨 
            #               어디 rule에 속하는지 보여줌 #
            predict_info.loc[sub_set_idx_, "pred_rule"] +=' rule_{} '.format(rule_nm)
            predict_info.loc[sub_set_idx_, "coverage"] +=' {} '.format(len(sub_set_idx_)/len(org_df))
            predict_info.loc[sub_set_idx_, "count"] += 1
            #print(predict_info)
            print(len(sub_set_idx_))
            
            
            df_sub_set_idx_num.loc['The_number_of_test_index',"rule_%d"%rule_nm] = int(len(sub_set_idx_))
          
            
            rule_index[rule_nm]= list(sub_set_idx_)
            assigned_idx += list(sub_set_idx) 
        
        self.df_total_inf_base = pd.concat([self.df_total_inf_base,df_sub_set_idx_num])
        #self.df_total_inf_base.loc['The_number_of_rule_index'] = self.df_total_inf_base.loc['The_number_of_rule_index'].astype('int')
        
        
        ## 여기서 추가작업을 하면 될거 같다 ##
        # na_idx: rule에 속하지 못한 나머지 data인듯 
        na_idx = list(set(org_df_idx) - set(assigned_idx))
        n_na_val = len(na_idx)
        rule_index[-1] = na_idx
        #print(len(na_idx))
        print(n_na_val) 
        # - ex) [240, 363, 235] 3
        
        self.df_total_inf_base = pd.merge(self.df_total_inf_base,df_other, left_index=True, right_index=True, how='outer', on = ['others'])    
        self.df_total_inf_base.loc['The_number_of_test_index','others'] = n_na_val
        
        if method =='average':
            # average일 때이니 pass함 
            predict_prob.loc[list(set(assigned_idx)),:] = \
                predict_prob.loc[list(set(assigned_idx)),:].values / \
                predict_info.loc[list(set(assigned_idx)),'count'] .values.reshape((-1,1))
            #예측 확률이 완전히 동일할 경우 어떤 값을 선택할 것인지 결정해야함(argmax는 index가 작은 class를 선택함.)
            predict_class.loc[list(set(assigned_idx)),:] = \
                np.argmax(predict_prob.loc[list(set(assigned_idx)),:].values,1).reshape((-1,1))
        
        ## 여기서 추가작업을 하면 될거 같다 ##
        
        # 밑의 코드는 rule에 속하지못하고 남은 data를 각각 확률값으로 변환 #
        predict_prob.loc[na_idx ,:] = [self.rule_model[-1][0][1]] * n_na_val 
        #print(self.rule_model[-1])
        #print([self.rule_model[-1][0][1]] * n_na_val)
        
        # 밑의 코드는 rule에 속하지못하고 남은 data를 각각 예측값으로 변환 -> 다 0이네 #
        predict_class.loc[na_idx ,:] = self.rule_model[-1][0][0]
        #print(predict_class.loc[na_idx ,:])
        
        # 밑의 코드는 rule에 속하지못하고 남은 data를 
        # predict_info에 pred_rule(others), coverage(남은data/전체data), count(몇개가 속한지)
        # 각 data에 동일하게 부여됨  #
        predict_info.loc[na_idx , "pred_rule"] +=' others '
        predict_info.loc[na_idx , "coverage"] +=' {} '.format(n_na_val/len(org_df))
        predict_info.loc[na_idx, "count"] += 1
        #print(predict_info.loc[na_idx , "pred_rule"])
        #print(predict_info.loc[na_idx , "coverage"])
        #print(predict_info.loc[na_idx, "count"])
        
        # predict_class: 위 과정들을 통해 완성되고 예측된 class만 index별로 뽑음 #
        predict_class.loc[:,:] =  np.array([self.CLASS_DICT[i] for i in predict_class.values.reshape(-1)]).reshape(-1,1)
        #print(predict_class.loc[:,:])
        
        rule_predict_info = {}
        rule_predict_prob = {}
        rule_predict_class = {}
        
        for i in rule_index:
            rule_predict_info[i] = predict_info.loc[rule_index[i], :]
            rule_predict_prob[i] = predict_prob.loc[rule_index[i], :]
            rule_predict_class[i] = predict_class.loc[rule_index[i], :]
        
            
        return rule_predict_class, rule_predict_prob, rule_predict_info
    

















