# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:07:23 2020

@author: User
"""

import collections
import pandas as pd
import numpy as np
#import heapq
from typing import *

import os
workingdir = r'C:\Users\User\Desktop\tutorial_rulebase\working'
os.chdir(os.path.join(workingdir, 'modules')) 

from Rule_test import Data_Setting as DS
import numpy as np
import pandas as pd
from glob import glob

from sklearn.model_selection import train_test_split




class Rule_info_dic(object):
    
    
    def __init__(self, d_set, range_num):
        
        self.d_set = d_set
        self.range_num = range_num
        
        
    def working(self):
        
        target_att = 'target'
        test_ratio = 0.2
        sample_ratio = 0.03
        
        dataset_dir = glob(os.path.join(workingdir, 'dataset', '*.csv'))
        data_diction = {dset.split('\\')[-1][:-4]: pd.read_csv(dset) for dset in dataset_dir}
        #dataset_list = sorted(list(data_diction.keys()), reverse=True)
        
        rule_rate = 1
        
        data = data_diction[self.d_set]
        data = data.reset_index(drop=True)
        colnm = data.columns
        data.info()
        
        fin_num = self.range_num
        
        
        self.rule_dic_list = []
        self.train_list = []
        for num in range(0, fin_num):
        
            ind = DS(self.d_set, test_ratio, workingdir)
            rule_dic, rule_dic_ex, rule_info = ind.data_set(rule = 'rule1', 
                                                            sample_ratio=sample_ratio, 
                                                            random_state = num,  
                                                            output_graph=False, 
                                                            rule_rate=rule_rate)
            
            X = data.loc[:,colnm [colnm != target_att]]
            y = data.loc[:, target_att]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, \
                                   random_state=num)
            
            train_idx, test_idx = X_train.index, X_test.index  
            train, _ = data.loc[train_idx,:], data.loc[test_idx,:]
            
            
            self.train_list.append(train)
            self.rule_dic_list.append(rule_dic_ex)
            
        return self.train_list, self.rule_dic_list
            

class Determine_Rule_Num(Rule_info_dic) :
    
    #def __init__(self, train_data, rule_dic_ex):
        
    #    self.train = self.train
    #    rule_dic_ex = rule_dic_ex
    
    
    def random_data(self):
        
        self.train_list, self.rule_dic_list = self.working()
        
        list_df_total_cum = []
        data_final_rule_store_list = {}
        data_num = 0
        for df, rule_dic_ex in zip(self.train_list, self.rule_dic_list):
            
            #df = self.train
            df_total_cum = pd.DataFrame()
            
            cum_rule_ind = []
    
            df_base_test = df
            df_adv_test = df
    
            # indepen depen 파악용 
            final_rule_store = 1
            
            final_rule_store_list = []
            
            det = 'Independant'
            
            for rule_nm in sorted(rule_dic_ex.keys())[1:]:
                
                test_num = 0
                df_total = pd.DataFrame()
                print(f'{rule_nm}번째 규칙')
                
                rule_list = rule_dic_ex[rule_nm][:-1]
                
                rule_length = len(rule_dic_ex[rule_nm][:-1])
                
                while rule_list != []:
                    
                    rule_list_temp = rule_list.pop(0)
                    
                    for rule, max_path in [rule_list_temp]: 
        
                        att, cond, value = rule.split()
    
                        
                        if rule_nm == 0:
                        
                            if cond == '>=':
                                
                                if max_path:
                                    
                                    cond_adv = '>='
                            
                                    sc_adv = ' '.join([att, cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] >= float(value),:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] < float(value),:].index
                                    
                                else:
                                    
                                    cond_adv = '<'
                            
                                    sc_adv = ' '.join([att, cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] < float(value),:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] >= float(value),:].index
                                
                            elif cond == '==':
                    
                                if max_path:
                                
                                    cond_adv = '=='
                            
                                    sc_adv = ' '.join([att, cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] == value,:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] != value,:].index
                                    
                                else:
                                    
                                    cond_adv = '!='
                            
                                    sc_adv = ' '.join([att, cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] != value,:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] == value,:].index
                                
                            df_base_test = df_base_test.loc[base_sub_set_idx,:]
                            df_adv_test = df_adv_test.loc[adv_sub_set_idx,:]
    
                            #cum_rule_ind.extend(sorted(base_sub_set_idx))
                            
                            #cum_rule_ind = list(set(cum_rule_ind))
                            
                        elif rule_nm != 0:
                            
                            if cond == '>=':
                                
                                if max_path:
                                    
                                    cond_adv = '>='
                            
                                    sc_adv = ' '.join([att,cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] >= float(value),:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] >= float(value),:].index
                                    
                                    temp_base_sub_set_idx = df_adv_test.loc[df_adv_test[att] < float(value),:].index
                                    
                                else:
                                    
                                    cond_adv = '<'
                            
                                    sc_adv = ' '.join([att,cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] < float(value),:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] < float(value),:].index
                                
                                    temp_base_sub_set_idx = df_adv_test.loc[df_adv_test[att] >= float(value),:].index
                                
                            elif cond == '==':
                    
                                if max_path:
                                    
                                    cond_adv = '=='
                            
                                    sc_adv = ' '.join([att,cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] == value,:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] == value,:].index
                                    
                                    temp_base_sub_set_idx = df_adv_test.loc[df_adv_test[att] != value,:].index
                                    
                                else:
                                    
                                    cond_adv = '!='
                            
                                    sc_adv = ' '.join([att,cond_adv, value])
                                    
                                    base_sub_set_idx = df_base_test.loc[df_base_test[att] != value,:].index
                                    adv_sub_set_idx = df_adv_test.loc[df_adv_test[att] != value,:].index
                                    
                                    temp_base_sub_set_idx = df_adv_test.loc[df_adv_test[att] == value,:].index
                        
                        #print(f'base_sub_set_idx: {sorted(base_sub_set_idx)}')
                        
                        print(f'sc_adv: {sc_adv}')
                        
                        print(f'df_base_test: {len(df_base_test)}')
                        print(f'df_adv_test: {len(df_adv_test)}')
                        
                        test_num += 1
 
                        
                        if rule_nm == 0 and test_num == rule_length:
                            
                            cum_rule_ind.extend(sorted(base_sub_set_idx))
                            cum_rule_ind = list(set(cum_rule_ind))
                            #base_sub_set_idx = base_sub_set_idx.values
                            
                            df_base_test = df_base_test.loc[cum_rule_ind,:]
                            df_adv_test = df.loc[df.index.difference(cum_rule_ind),:]
                            
                            #print(f'df_adv_test: {len(df_adv_test)}')
                        elif rule_nm != 0 and test_num != rule_length:
                            
                            #cum_rule_ind.extend(sorted(base_sub_set_idx))
                            #cum_rule_ind = list(set(cum_rule_ind))
                            df_temp = df_adv_test
                            
                            df_base_test = df_base_test.loc[base_sub_set_idx,:]
                            df_adv_test = df_adv_test.loc[adv_sub_set_idx,:]
                            
                            
                            
                        elif rule_nm != 0 and rule_length >= 2 and test_num == rule_length:
                            
                            
                            temp_base_sub_set_idx = df_temp.loc[df_temp.index.difference(adv_sub_set_idx),:].index
                
                        #rule_store.append(sc_adv)
                        
                print(f'base_sub_set_idx: {len(base_sub_set_idx)}')
                print(f'adv_sub_set_idx: {len(adv_sub_set_idx)}')
                # adv가 기존 rule, 엑셀용은 반대
                
                if rule_nm != 0 and len(base_sub_set_idx) > 0: 
    
                    
                    # 기존 rule의 index를 더함
                    #print(f'rule_store: {rule_store}')
                    print(f'cum_rule_ind: {len(cum_rule_ind)}')
                    
                    # 해당 rule class num 과정
                    df_base = df.loc[base_sub_set_idx,:] 
                    df_adv = df.loc[adv_sub_set_idx,:] 
                    
                    a,b = collections.defaultdict(list), collections.defaultdict(list)
    
                    a['base'] = df_base.groupby(by=df['target'])['target'].count()
                    a['adv'] = df_adv.groupby(by=df['target'])['target'].count()
                    
                    for i,j in a.items():
                        
                        length = len(j)
                        
                        if length == 1 and j.index == 0:
                        
                            b[i] = [j[0],0]
                        
                        elif length == 1 and j.index == 1:
                            
                            b[i] = [0,j[1]]
                            
                        elif length == 0:
                            
                            b[i] = [0,0]
                        
                        else:
                            
                            b[i] = [j[0],j[1]]
                            
                    # class num - 필 
                    base_cls0, base_cls1 = b['base']
                    adv_cls0, adv_cls1 = b['adv']
                    
                    # 0과 1중 pred - 필
                    base_pred = np.argmax(b['base'])
                    adv_pred = np.argmax(b['adv'])
        
                    # class 중 max
                    base_num_max = max(base_cls0, base_cls1)
                    adv_num_max = max(adv_cls0,adv_cls1)
                    
                    # rule class total - 필
                    base_num_total = base_cls0 + base_cls1
                    adv_num_total = adv_cls0 + adv_cls1
                    
                    # homogeneity - 필
                    
                    try:
                        base_homo = base_num_max / base_num_total
                        adv_homo = adv_num_max / adv_num_total
                        
                        base_homo = np.round(base_homo, 4)
                        adv_homo = np.round(adv_homo, 4)
                        
                    except ZeroDivisionError:
                        adv_homo = 0
                    
                    # coverage - 필
                    try:
                        #base_cover_total = (base_cls0 + base_cls1) / len(self.train)
                        #adv_cover_total = (adv_cls0 + adv_cls1) / len(self.train)
                        
                        base_cover_total = (base_cls0 + base_cls1) / len(df)
                        adv_cover_total = (adv_cls0 + adv_cls1) / len(df)
                        
                        
                        base_cover_total = np.round(base_cover_total, 4)
                        adv_cover_total = np.round(adv_cover_total, 4)
                        
                    except ZeroDivisionError:
                        adv_cover_total = 0
                    
                    '''
                    # 이 조건을 만족하면 독립적이다 #
                    # 만약 인덱스가 0이면 독립이라고 봄.
                    '''  
                    
                    # [설명]
                    # 1.645: 단측검정(0.05)
                    # adv_homo: rule의 homogeneity
                    # base_num_total: rule과 다른 분기기준을 적용했을 때의 샘플 수
                    # [정리]
                    # base_num_total 수가 작을수록 adv_homo가 작을수록
                    # epsilon의 값은 커짐 - 커질수록 좋음
                    
                    epsilon = 1.645 * np.sqrt( (adv_homo * (1- adv_homo)) / base_num_total)
                    
                    epsilon = np.round(epsilon, 4)
                    print(f'epsilon: {epsilon}')
                    
                    adv_homo_ep = adv_homo - epsilon
                    adv_homo_ep = np.round(adv_homo_ep, 4)
                    
                    print(f'comp: {base_homo, adv_homo_ep}')
                
                    # adv_homo_ep가 작을수록 조건을 만족할 확률이 높아짐
                    if base_homo >= adv_homo_ep:
                        print(f'{rule_nm}번째 규칙은 Independant')
                        
                        det = 'Independant'
                        
                        #del rule_store[:]
                        #rule_store.append(sc_adv)
                        #final_rule_store = dep_rule_store + rule_store
                        
                        #독립이므로 이전 룰들의 인덱스와 현재 룰의 인덱스를 더함  
                        cum_rule_ind.extend(list(sorted(set(adv_sub_set_idx))))
                        cum_rule_ind = list(set(cum_rule_ind))
                        cum_adv_sub_set_idx = cum_rule_ind
                        df_base_test = df.loc[cum_adv_sub_set_idx, :]
                        print(f'df_base_test: {len(df_base_test)}')
                    
                    # 이 조건을 만족하지 못하면 Dependent
                    elif base_homo < adv_homo_ep:
                        print(f'{rule_nm}번째 규칙은 dependant')
                        
                        det = 'Dependant'
                        
                        final_rule_store += 1
                        
                        
                        #dep_rule_store.append(sc_adv)
                        #final_rule_store = dep_rule_store
                        
                        cum_rule_ind = list(set(sorted(adv_sub_set_idx)))
                        df_base_test = df.loc[cum_rule_ind, :]
                        
                    df_adv_test = df.loc[temp_base_sub_set_idx,:]
    
                    
                    ##############
                    # infomation #
                    ##############
    
                    sort_ind = ['Sub_set_idx', 'Class_0', 'Class_1', 'Class_predict', \
                                'Total_class_num', 'Homogeneity', 'Epsilon', \
                                'Homogeneity-E', 'Coverage', 'Cum_index', 'Rule_determination', \
                                'Rule_attrs']
                
                    sort_col = [f'base_rule_{rule_nm}', f'adv_rule_{rule_nm}']
                      
    
                    
                    # base_homo 대신 adv_homo_ep 들어감
                    data = \
                    {sort_col[0] : [len(adv_sub_set_idx), adv_cls0, adv_cls1, adv_pred, \
                                    adv_num_total, adv_homo, epsilon, adv_homo_ep, \
                                    adv_cover_total, '-', det, final_rule_store],
                     sort_col[1] : [len(base_sub_set_idx), base_cls0, base_cls1, base_pred, \
                                    base_num_total, base_homo, '-',  '-', \
                                    base_cover_total, len(cum_rule_ind), '-', '-']
                    }
                    
    
    
                    df_total = pd.DataFrame(data, index=sort_ind)
                    
                    #df_total_cum = pd.concat([df_total_cum, df_total], axis=1)  
                    
                elif rule_nm != 0 and len(base_sub_set_idx) < 1:
                    
                    df_adv = df.loc[adv_sub_set_idx,:] 
                    cum_rule_ind.extend(list(sorted(adv_sub_set_idx)))
                    
                    a,b = collections.defaultdict(list), collections.defaultdict(list)
                    
                    a['adv'] = df_adv.groupby(by=df['target'])['target'].count()
                    
                    for i,j in a.items():
                        
                        length = len(j)
                        
                        if length == 1 and j.index == 0:
                        
                            b[i] = [j[0],0]
                        
                        elif length == 1 and j.index == 1:
                            
                            b[i] = [0,j[1]]
                            
                        else:
                            
                            b[i] = [j[0],j[1]]
                            
                    # class num - 필 
                    base_cls0, base_cls1 = np.nan, np.nan
                    adv_cls0, adv_cls1 = b['adv']
                    
                    # 0과 1중 pred - 필
                    base_pred = np.nan
                    adv_pred = np.argmax(b['adv'])
        
                    # class 중 max
                    base_num_max = np.nan
                    adv_num_max = max(adv_cls0,adv_cls1)
                    
                    # rule class total - 필
                    base_num_total = np.nan
                    adv_num_total = adv_cls0 + adv_cls1
                    
                    # homogeneity - 필
                    base_homo = np.nan
                    adv_homo = adv_num_max / adv_num_total
                    adv_homo = np.round(adv_homo, 4)
                    
                    # coverage - 필
                    base_cover_total = np.nan
                    #adv_cover_total = (adv_cls0 + adv_cls1) / len(self.train)
                    
                    adv_cover_total = (adv_cls0 + adv_cls1) / len(df)
                    adv_cover_total = np.round(adv_cover_total, 4)
                    
                                        
                    # 비교 - 필
                    #base_weight = base_homo * base_cover_total
                    #adv_weight = adv_homo * adv_cover_total
                    
                    #if base_homo <= adv_homo and base_cover_total < adv_cover_total:
                    
                    print(f'{rule_nm}번째 규칙은 Independant')
                    det = 'Independant'
                    #adv_sub_set_idx = cum_rule_ind
                    print(f'df_base_test: {len(adv_sub_set_idx)}')
                    
                    df_base_test = df.loc[cum_rule_ind, :]
                    df_adv_test = df.loc[temp_base_sub_set_idx,:]
                        
                    sort_ind = ['Sub_set_idx', 'Class_0', 'Class_1', 'Class_predict', \
                                'Total_class_num', 'Homogeneity', 'Coverage', 'Rule_determination', \
                                'Rule_attrs']
                
                    sort_col = [f'base_rule_{rule_nm}', f'adv_rule_{rule_nm}']
                
                    data = \
                    {sort_col[0] : [len(adv_sub_set_idx), adv_cls0, adv_cls1, adv_pred, \
                                    adv_num_total, adv_homo, adv_cover_total, det, final_rule_store],
                     sort_col[1] : [len(base_sub_set_idx), base_cls0, base_cls1, base_pred, \
                                    base_num_total, base_homo, base_cover_total, '-', '-']
                    }
                        
                    df_total = pd.DataFrame(data, index=sort_ind)
                    
                final_rule_store_list.append(final_rule_store)

                df_total_cum = pd.concat([df_total_cum, df_total], axis=1)
                
            data_num += 1
            df_total_cum.columns.name = f'{data_num}th_data'
                
            list_df_total_cum.append(df_total_cum)
            data_final_rule_store_list[f'{data_num}th_data'] = final_rule_store_list
            
        return list_df_total_cum, data_final_rule_store_list

        