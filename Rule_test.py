# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:25:15 2020

@author: GeunSeong
"""

import numpy as np
import pandas as pd

from glob import glob
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.getcwd(),'DT_model'))

# Rule 경로
workingdir = r'C:\Users\User\Desktop\tutorial_rulebase\working'
os.chdir(os.path.join(workingdir, 'modules' )) 
from rulebase import ruleBase as ubr
#from usertree import userTree as utr
import utils

from rulebase_excel import ruleBase as ubr_ex

# DT 경로
from usertree_cart import userTree as utr_cart

# return path
from sklearn.tree import DecisionTreeClassifier
os.chdir(os.path.join(workingdir, 'modules' )) 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import utils_cart
import graphviz
from typing import *


class Data_Setting(object):
    
    def __init__(self, d_set, test_ratio, workingdir):
        self.d_set = d_set
        self.test_ratio = test_ratio
        self.workingdir = workingdir
        
        
    def data_set(self, rule, sample_ratio, random_state, output_graph=False, rule_rate=0.7):
        self.random_state = random_state
        self.rule = rule
        self.output_graph = output_graph
        self.rule_rate = rule_rate
        
        dataset_dir = glob(os.path.join(self.workingdir, 'dataset', '*.csv'))
        data_diction = {dset.split('\\')[-1][:-4]: pd.read_csv(dset) for dset in dataset_dir}
        #dataset_list = sorted(list(data_diction.keys()), reverse=True)
        
        lambda_range = sorted([1-np.log10(i) for i in np.arange(1,10,1)])
        #lambda_range = sorted([1-np.log10(i) for i in np.arange(1,10,0.5)])

        target_att = 'target'
        
        MAX_DEPTH = 1000
        self.sample_ratio = sample_ratio
        
        data = data_diction[self.d_set]
        
        print(data)
        colnm = data.columns
        
        X = data.loc[:,colnm [colnm != target_att]]
        y = data.loc[:, target_att]
        #target_elements = np.unique(y)
            
        in_feature = list(data.columns [data.columns != target_att])
        
        cate_col = [col for col in in_feature \
                            if not np.issubdtype(X[col ].dtype, \
                    np.number)]
                
        X_dummies = pd.get_dummies(data.loc[:,in_feature], columns=cate_col)

        print(f'The_number_of_dummies_columns: {len(X_dummies.columns)}')

        #1.1 graph용 suffle (x)
        self.X_train_ori, self.X_test_ori, self.y_train_ori, self.y_test_ori = train_test_split(X, y,
                                                                                                test_size=self.test_ratio, \
                                                                                                    random_state=self.random_state)
        #1.2 graph용 suffle (o)
        #self.X_train_ori, self.X_test_ori, self.y_train_ori, self.y_test_ori = train_test_split(X, y, test_size=self.test_ratio, \
        #    random_state=self.random_state, stratify=y, shuffle = True)
    
        ori_data = pd.concat([X, y], axis=1)
        
        train_idx_ori, test_idx_ori = self.X_train_ori.index, self.X_test_ori.index
        self.train_ori, self.test_ori = ori_data.loc[train_idx_ori,:], data.loc[test_idx_ori,:]
    
        #2.1 analy용 suffle (x)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_dummies, y, test_size=self.test_ratio, \
                   random_state=self.random_state)
        
        #2.2 analy용 suffle (x)
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_dummies, y, test_size=self.test_ratio, \
        #   random_state=self.random_state, stratify=y, shuffle = True)
            
        dummie_data = pd.concat([X_dummies, y], axis=1)
            
        train_idx, test_idx = self.X_train.index, self.X_test.index
        self.train, self.test = dummie_data.loc[train_idx,:], dummie_data.loc[test_idx,:]
          
        print(len(self.train))
        print(len(self.test))
        
        n_samples = round(self.sample_ratio * len(self.train))
        print(n_samples)
            
        # base #
        self.ADAPTIVE_rule_ins = ubr(n_samples, MAX_DEPTH, params=lambda_range, algorithm='adaptive', simplify = True)
        self.rule_fit = self.ADAPTIVE_rule_ins.fit(df_org=self.train, df_g=self.train_ori, target_attribute_name='target', \
                                               output_graph = self.output_graph, save_dir='{}_{}(0.03)'.format(self.d_set, self.random_state), \
                                               d_set=self.d_set, rule_rate=rule_rate)
        
        # excel #
        self.ADAPTIVE_rule_ins_ex = ubr_ex(n_samples, MAX_DEPTH, params=lambda_range, algorithm='adaptive', simplify = True)
        self.rule_fit_ex, rule_info = self.ADAPTIVE_rule_ins_ex.fit(df_org=self.train_ori, target_attribute_name='target', \
                                               output_graph = self.output_graph, save_dir='{}_{}(0.03)'.format(self.d_set, self.random_state), \
                                               d_set=self.d_set, rule_rate=rule_rate) 
        
                     

        ###############
        ## lift 수치 ## 
        ###############
        
        prior_0 = len(self.train[self.train['target'] == 0]) / len(self.train)
        prior_1 = len(self.train[self.train['target'] == 1]) / len(self.train)
        
        print(f'prior_0_ratio: {np.round(prior_0,3)}')
        print(f'prior_1_ratio: {np.round(prior_1,3)}')
        
        ranges = len(self.ADAPTIVE_rule_ins.df_total_inf_base.columns)
        
        for rule_num in range(ranges):
            
            
            posterior = self.ADAPTIVE_rule_ins.df_total_inf_base.loc['Homogeneity','rule_%d'%rule_num]
            
            rule_pred = self.ADAPTIVE_rule_ins.df_total_inf_base.loc['Rule_predict','rule_%d'%rule_num]
            
            if rule_pred == 0:
                lift = posterior / prior_0
                lift = np.round(lift, 3)
                
                
            elif rule_pred == 1:
                lift = posterior / prior_1
                lift = np.round(lift, 3)
                
            self.ADAPTIVE_rule_ins.df_total_inf_base.loc['Lift','rule_%d'%rule_num] = lift

            #############################
            # rule index를 제외한 나머지 #
            #############################
            
        others_index = len(self.train) - self.ADAPTIVE_rule_ins.rule_tra_ind_num
        others_coverage = np.round(others_index / len(self.train), 3)
        
        others_ind = pd.DataFrame(data=[others_index,others_coverage], columns = ['others'],  index = ['The_number_of_train_index','Train_coverage'])
        
        self.ADAPTIVE_rule_ins.df_total_inf_base = pd.merge(self.ADAPTIVE_rule_ins.df_total_inf_base,others_ind, \
                                                            how = 'outer', left_index=True, right_index=True)
        
        return self.rule_fit, self.rule_fit_ex, rule_info
        
    def perfomace(self):
        self.rule_fit
         
        df_perform = pd.DataFrame(data=np.nan, columns=self.ADAPTIVE_rule_ins.df_total_inf_base.columns, index=['Accuracy'])
            
        # 여기 부분이 train data로 모델을 생성 했을 때 test data를 집어 넣고 예측값을 구하기 위한 것
        ADAPTIVE_rule_prior_predict_class, ADAPTIVE_rule_prior_predict_prob, ADAPTIVE_rule_prior_coverage_info = \
        self.ADAPTIVE_rule_ins.predict(self.test, method='priority')
        
        dic_cover = {}
        
        # coverage # 

        if self.rule_rate != 1:
            for num in list(ADAPTIVE_rule_prior_coverage_info.keys()):
                
                key = ADAPTIVE_rule_prior_coverage_info[num]['pred_rule'].unique()
                key = key[0].strip()
                
                val = ADAPTIVE_rule_prior_coverage_info[num]['coverage'].unique()
                val = float(val[0].strip())            
                val = round(val, 3)
                
                dic_cover[key] = val
                
        elif self.rule_rate == 1:
            for num in list(ADAPTIVE_rule_prior_coverage_info.keys())[:-1]:
                
                key = ADAPTIVE_rule_prior_coverage_info[num]['pred_rule'].unique()
                key = key[0].strip()
                
                val = ADAPTIVE_rule_prior_coverage_info[num]['coverage'].unique()
                val = float(val[0].strip())            
                val = round(val, 3)
                
                dic_cover[key] = val
            
        df_cover = pd.DataFrame(dic_cover, index=['Test_coverage'])

        self.ADAPTIVE_rule_ins.df_total_inf_base = pd.concat([self.ADAPTIVE_rule_ins.df_total_inf_base,df_cover]) 
   
        ad_rule_prior_result = []
        ad_confu_matrix = []
        correct_label_total = []

        df_p = pd.DataFrame()
        df_test_label = pd.DataFrame(self.test['target'])
        
        for i in list(ADAPTIVE_rule_prior_predict_class.keys())[:-1]:
            
            ADAPTIVE_rule_prior_result, confu_matrix_raw, correct_label = utils.perform_check(self.test['target'], \
                                 ADAPTIVE_rule_prior_predict_class[i], ADAPTIVE_rule_prior_predict_prob[i], \
                                 self.ADAPTIVE_rule_ins.NUM_CLASSES, self.ADAPTIVE_rule_ins.CLASS_DICT_)
            
                
            ad_rule_prior_result.append(ADAPTIVE_rule_prior_result)
            ad_confu_matrix.append(confu_matrix_raw)
            correct_label_total.append(correct_label)
            
            df_p = pd.concat([df_p, ADAPTIVE_rule_prior_predict_class[i]])
            
        # f1, auc 구하기 # 
        df_rule_labels = pd.concat([df_test_label, df_p], axis=1)
        
        df_rule_labels.dropna(axis=0, how='any',inplace=True)
        
        # rule label 가져가기 #
        self.df_rule_labels = df_rule_labels
        
        try:
            rule_acc = np.round(metrics.accuracy_score(df_rule_labels['target'], df_rule_labels['class']),4)
            rule_f1 = np.round(metrics.f1_score(df_rule_labels['target'], df_rule_labels['class']),4)
            rule_auc = np.round(metrics.roc_auc_score(df_rule_labels['target'], df_rule_labels['class']),4)
        
        except ValueError:
            rule_auc = np.nan
            
        df_rule_perf = pd.DataFrame({'Rule':[rule_acc,rule_f1,rule_auc]}, index=['Accuracy','F1_score','Auc']) 

        self.ADAPTIVE_rule_ins.df_total_inf_base = pd.concat([self.ADAPTIVE_rule_ins.df_total_inf_base, df_rule_perf], \
                                                                 axis=1)    
            
        df_others_pred = ADAPTIVE_rule_prior_predict_class[-1]
        
        
        if self.rule_rate == 1 and df_others_pred.empty == False:
            
            df_others_index = self.test.loc[np.in1d(self.test.index, df_rule_labels.index) == False]
            df_others_target = ADAPTIVE_rule_prior_predict_class[-1]
            
            print(f'df_others_target: {df_others_index}')
            print(f'df_others_target: {df_others_target}')
            
            df_total_rule_true = pd.concat([df_rule_labels['target'], df_others_index['target']]).sort_index()
            df_total_rule_pred = pd.concat([df_rule_labels['class'], df_others_target['class']]).sort_index()
        
            total_rule_acc = np.round(metrics.accuracy_score(df_total_rule_true, df_total_rule_pred),4)
            total_rule_f1 = np.round(metrics.f1_score(df_total_rule_true, df_total_rule_pred),4)
            total_rule_auc = np.round(metrics.roc_auc_score(df_total_rule_true, df_total_rule_pred),4)
        
            df_total_rule_perf = pd.DataFrame({'Total_Rule':[total_rule_acc,total_rule_f1,total_rule_auc]}, index=['Accuracy','F1_score','Auc'])
                    
            self.ADAPTIVE_rule_ins.df_total_inf_base = pd.concat([self.ADAPTIVE_rule_ins.df_total_inf_base, df_total_rule_perf], \
                                                         axis=1)
        
        elif self.rule_rate == 1 and df_others_pred.empty == True:
            
            df_total_rule_perf = pd.DataFrame({'Total_Rule':[rule_acc,rule_f1,rule_auc]}, index=['Accuracy','F1_score','Auc'])
                    
            self.ADAPTIVE_rule_ins.df_total_inf_base = pd.concat([self.ADAPTIVE_rule_ins.df_total_inf_base, df_total_rule_perf], \
                                                         axis=1)
            
        
        for k,j in enumerate(list(ADAPTIVE_rule_prior_predict_class.keys())[:-1]):
            perform_base_str = '{} : ACCURACY :{}, RECALL :{}, PRECISION : {}, F1 : {}'
            print(perform_base_str.format('\nADAPTIVE_rule_result_base [rule_{}]\n '.format(j), \
                        *np.round(np.array(ad_rule_prior_result[k]), 3)))
                
            df_perform.loc['Accuracy',"rule_%d"%j] = np.round(ad_rule_prior_result[k][0],3)    
            
            
            #self.ADAPTIVE_rule_ins.df_total_inf_base = pd.concat([self.ADAPTIVE_rule_ins.df_total_inf_base, df_perform])
                                                                
        self.ADAPTIVE_rule_ins.df_total_inf_base.loc['Accuracy',df_perform.columns] = \
            self.ADAPTIVE_rule_ins.df_total_inf_base.loc['Accuracy',df_perform.columns]=df_perform.loc['Accuracy']
            
        if self.rule_rate != 1:
            
            # correct_label_total_rule_ACC
            others_index_number = len(ADAPTIVE_rule_prior_predict_class[-1])
            print(f'others_index_number: {others_index_number}')
            
            #total_rule_ACC = round(sum(correct_label_total) / (len(self.test) - others_index_number), 4)
            
            print("총 {}개 중 ".format(len(self.test) - others_index_number))
            print("rule: 맞춘 갯수: ", sum(correct_label_total))
        
            others_predict_index = ADAPTIVE_rule_prior_predict_class[-1].index
            
            others_real_index = pd.DataFrame(self.test, columns=self.test.columns)
            
            others_real_index = others_real_index.loc[others_predict_index, :]
            
            XX = others_real_index.iloc[:,:-1]
            yy = others_real_index.iloc[:,-1]     
            
            ##############################
            # Others index -> Ranfo 적용 #
            ##############################
            rf_clf_bs=RandomForestClassifier(n_estimators=1000, random_state=0)
            
            rf_clf_bs.fit(self.X_train,self.y_train)
    
            y_pred_bs = rf_clf_bs.predict(XX)
            
            test_y_true = np.array(yy)
            
            #############################
            # others ranfo label 가져감 #
            #############################
            others_label = {'target': test_y_true, 'class': y_pred_bs}
            df_others_label = pd.DataFrame(data=others_label, index=others_predict_index)
      
            print(f'others_ranfo_true_label: {test_y_true}')
            print(f'others_ranfo_pred_label: {y_pred_bs}')
            
            print("******************************")
            print('Randomforest Accuracy :', np.round(metrics.accuracy_score(test_y_true, y_pred_bs),4))
            print("******************************")
            
            try:
                ranf_acc = np.round(metrics.accuracy_score(test_y_true, y_pred_bs),4)
                ranf_f1 = np.round(metrics.f1_score(test_y_true, y_pred_bs),4)
                ranf_auc = np.round(metrics.roc_auc_score(test_y_true, y_pred_bs),4)
                
            except ValueError:
                ranf_auc = np.nan
            
            
            df_ranf_perf = pd.DataFrame({'Others_Randomforest':[ranf_acc,ranf_f1,ranf_auc]}, index=['Accuracy','F1_score','Auc'])
            

            self.ADAPTIVE_rule_ins.df_total_inf_base = pd.merge(self.ADAPTIVE_rule_ins.df_total_inf_base, df_ranf_perf, \
                                                                 how = 'outer', left_index=True, right_index=True)
        

            ####################
            # ranfo Total perf #
            ####################
            ranfo_rig_num = len(test_y_true[np.equal(test_y_true,y_pred_bs) == True])
            #rule_rig_number = sum(correct_label_total)
            
            print("총 {}개 중 ".format(len(y_pred_bs)))
            print("ranfo: 맞춘 갯수: ", ranfo_rig_num)
            
            ###########################
            # Others index -> DT 적용 #
            ###########################
            
            target_name = 'target'
            MAX_DEPTH = 1000
            params = 'gini'
            simplify = True
            n_samples = round(self.sample_ratio * len(self.train))
            #n_samples = 1
            
            CART_ins = utr_cart(n_samples, MAX_DEPTH, params, simplify)
            
            gini_tree, gini_pprint_tree= CART_ins.fit(self.train, target_attribute_name = target_name)
            
            CART_tree, CART_graph_tree = \
                           CART_ins.tree, CART_ins.graph_tree 
                            
            ## utils.get_usrt_info 코드 설명 ##
            # train: data의 train data
            # CART_gini_tree: tree에서 분리된 노드들의 분기기준과 data의 index, target(=class)가 출력 -> 성능측정용
            # target_att: data의 y이름
            
            #DT_info = utils_cart.get_cart_info(self.train ,CART_tree, target_att = target_name)

            ## test predict ## 
            test_CART_all_pred, test_CART_all_pred_prob \
                = CART_ins.predict(others_real_index, CART_tree)
            
            # y의 name 넣기 
            classes = np.unique(self.train[target_name])
            
            # test metric
            test_CART_all_met = utils_cart.perform_check(others_real_index['target'], \
                     test_CART_all_pred, \
                     test_CART_all_pred_prob, \
                     len(classes), CART_ins.CLASS_DICT_)
                
            # 확인용
            perform_base_str = '{} : ACCURACY :{}, RECALL :{}, PRECISION : {}, F1 : {}, AUC : {}'
            print(perform_base_str.format('\nMetric\n ', \
                        *np.round(np.array(test_CART_all_met), 4)))
            
            #####################
            ## Others_DT_perfo ##
            #####################
            Acc_dt = np.round(test_CART_all_met[0],4)
            F1_dt = np.round(test_CART_all_met[3],4)
            Auc_dt = np.round(test_CART_all_met[4],4)
            
            others_dt_metric_sel = [Acc_dt, F1_dt, Auc_dt]
            
            others_dt_metric_sel_df = pd.DataFrame(data=others_dt_metric_sel, columns=['Others_Decision_tree'], index=['Accuracy', 'F1_score', 'Auc'])
    

            self.ADAPTIVE_rule_ins.df_total_inf_base = pd.merge(self.ADAPTIVE_rule_ins.df_total_inf_base, others_dt_metric_sel_df, \
                                                                 how = 'outer', left_index=True, right_index=True)
        
            # rule+ranfo 성능 #
            total_rule_ranfo = pd.concat([df_others_label,df_rule_labels], axis=0)
    
            # total true
            y_t = total_rule_ranfo['target']
            
            # total pred
            y_p = total_rule_ranfo['class']
        
            
            Total_Acc_ranfo = np.round(metrics.accuracy_score(y_t, y_p),4)
            Total_F1_ranfo = np.round(metrics.f1_score(y_t, y_p),4)
            Total_Auc_ranfo = np.round(metrics.roc_auc_score(y_t, y_p),4)   
            
            df_total_perf_ranfo = pd.DataFrame({'Total_performance(20%_ranfo)':[Total_Acc_ranfo,Total_F1_ranfo,Total_Auc_ranfo]}, index=['Accuracy','F1_score','Auc'])
            

            self.ADAPTIVE_rule_ins.df_total_inf_base = pd.merge(self.ADAPTIVE_rule_ins.df_total_inf_base, df_total_perf_ranfo, \
                                                                 how = 'outer', left_index=True, right_index=True)
        

            
            #############################
            ### others DT label 가져감 ###
            #############################
            others_label = {'target': test_y_true, 'class': test_CART_all_pred['class']}
            df_others_dt_label = pd.DataFrame(data=others_label, index=others_predict_index)       
            
            # rule+ranfo 성능 #
            total_rule_dt = pd.concat([df_others_dt_label,df_rule_labels], axis=0)
    
            # total true
            dt_y_t = total_rule_dt['target']
            # total pred
            dt_y_p = total_rule_dt['class'].apply(lambda x: int(x))
            
            Total_Acc_dt = np.round(metrics.accuracy_score(dt_y_t, dt_y_p),4)
            Total_F1_dt = np.round(metrics.f1_score(dt_y_t, dt_y_p),4)
            Total_Auc_dt = np.round(metrics.roc_auc_score(dt_y_t, dt_y_p),4)   
            
            df_total_perf_dt = pd.DataFrame({'Total_performance(20%_DT)':[Total_Acc_dt,Total_F1_dt,Total_Auc_dt]}, index=['Accuracy','F1_score','Auc'])
            

            self.ADAPTIVE_rule_ins.df_total_inf_base = pd.merge(self.ADAPTIVE_rule_ins.df_total_inf_base, df_total_perf_dt, \
                                                                 how = 'outer', left_index=True, right_index=True)
        

        
        ################# Dataframe 정리 ###################
        sort_ind = ['Accuracy', 'Auc', 'F1_score', 'Lift', 'Rule_predict',
       'Homogeneity', 'Train_coverage', 'Train_cumulative_coverage', 'Test_coverage', 'The_number_of_rule_attribute',
       'The_number_of_train_index', 'The_number_of_test_index']
        

        self.ADAPTIVE_rule_ins.df_total_inf_base = self.ADAPTIVE_rule_ins.df_total_inf_base.loc[sort_ind,:]
        
        self.ADAPTIVE_rule_ins.df_total_inf_base.append(pd.Series(), ignore_index=True)
        
        return self.ADAPTIVE_rule_ins.df_total_inf_base
        

        
        
    def DT_CART(self, target_name, MAX_DEPTH, params, simplify, graph=False):
        
        train = self.train.copy()
        train = train.sort_index()
        n_samples = round(self.sample_ratio * len(train))
        
        # 실제 train 데이터의 class별 각 갯수 #
        tra_uni_class = np.unique(train[target_name])
        tra_class_number = {}
        for cla_num in tra_uni_class:    
            tra_class_number[cla_num] = len(train[train[target_name] == cla_num])
        
        
        print(f'the_number_of_train_class: {tra_class_number}')
        
        CART_gini_ins = utr_cart(n_samples, MAX_DEPTH, params, simplify)
        
        # 모델 피팅 # 
        gini_tree, gini_pprint_tree= CART_gini_ins.fit(train, target_attribute_name = target_name)  
        
        ## CART_gini_tree: tree에서 분리된 노드들의 분기기준과 data의 index, target(=class)가 출력 -> 성능측정용
        ## CART_gini_graph_tree: tree 그래프를 생성하기 위한 값 저장 
        CART_gini_tree, CART_gini_graph_tree = \
                        CART_gini_ins.tree, CART_gini_ins.graph_tree 
                        
        ## utils.get_usrt_info 코드 설명 ##
        # train: data의 train data
        # CART_gini_tree: tree에서 분리된 노드들의 분기기준과 data의 index, target(=class)가 출력 -> 성능측정용
        # target_att: data의 y이름
        
        DT_info = utils_cart.get_cart_info(train ,CART_gini_tree, target_att = target_name)
        
        ## test predict ## 
        test_CART_gini_all_pred, test_CART_gini_all_pred_prob \
            = CART_gini_ins.predict(self.test, CART_gini_tree)
        
        # y의 name 넣기 
        classes = np.unique(train[target_name])
        
        # test metric
        test_CART_gini_all_met = utils_cart.perform_check(self.test[target_name], \
                 test_CART_gini_all_pred, \
                 test_CART_gini_all_pred_prob, \
                 len(classes), CART_gini_ins.CLASS_DICT_)

        # 확인용
        perform_base_str = '{} : ACCURACY :{}, RECALL :{}, PRECISION : {}, F1 : {}, AUC : {}'
        print(perform_base_str.format('\nMetric\n ', \
                    *np.round(np.array(test_CART_gini_all_met), 4)))
        
        #############
        ##수치 정리##
        #############
        Acc = np.round(test_CART_gini_all_met[0],4)
        F1 = np.round(test_CART_gini_all_met[3],4)
        Auc = np.round(test_CART_gini_all_met[4],4)
        
        metric_sel = [Acc, F1, Auc]
        
        metric_sel_df = pd.DataFrame(data=metric_sel, columns=['Decision_tree'], index=['Accuracy', 'F1_score', 'Auc'])

        #########################
        # Depth에 따른 수치 변화 #
        #########################
            
        Depth_columns = ['DT_Depth_%d'%x for x in DT_info['depth'].unique()]

        DT_df = pd.DataFrame(data=np.nan, columns=Depth_columns, \
                             index=['Homogeneity','Lift','Train_coverage', \
                                    'Number_of_leaves_by_depth', 'Cumulative_number_of_leaves_by_depth'])
        
        Depth_lists = [x for x in DT_info['depth'].unique()]
        
        temp = pd.DataFrame()
        cumul_depth_leaf = 0
        
        for depth_num in Depth_lists:
        
            depth_numeric = DT_info[DT_info.loc[:,'depth'] == depth_num]
            
            temp = pd.concat([temp, depth_numeric])
            
            homo = np.round(temp['homogeneity'].mean(), 4)
            lif = np.round(temp['lift'].mean(), 4)
            cov = np.round(temp['coverage'].sum(), 4)
            depth_leaf = len(depth_numeric)
            cumul_depth_leaf += depth_leaf
            
            total = [homo, lif, cov, depth_leaf, cumul_depth_leaf]
        
            DT_df.loc[ :, 'DT_Depth_%d'%depth_num] = total
        
        if graph == True:
            
            graph_dir = 'DT_graph'
            node, edge = CART_gini_ins.graph.tree_to_graph(CART_gini_graph_tree)
            tree_graph = graphviz.Source(node + edge+'\n}')
            
            # PDF 파일 출력 #
            tree_graph.render('{}/{}/CART_{}_{}_test_{}'.format(graph_dir, f'DT_graph_{self.d_set}_{self.rule_rate}', \
                                                                params, self.d_set, self.random_state))
        
        fin_DT = pd.concat([DT_df, metric_sel_df], axis=1)
        
        return fin_DT, DT_info
        
    def Ranfo(self):
        
        # n_estimators만 조정, 나머지 파라미터는 default
        
        X_test = self.X_test
        y_test = self.y_test

       
        rf_clf_bs=RandomForestClassifier(n_estimators=1000, random_state=0)
   
        rf_clf_bs.fit(self.X_train,self.y_train)
        
        y_pred_bs = rf_clf_bs.predict(X_test)
        
        ranf_acc = np.round(metrics.accuracy_score(y_test, y_pred_bs),4)
        ranf_f1 = np.round(metrics.f1_score(y_test, y_pred_bs),4)
        ranf_auc = np.round(metrics.roc_auc_score(y_test, y_pred_bs),4)
        
        df_ranf_perf = pd.DataFrame({'Randomforest':[ranf_acc,ranf_f1,ranf_auc]}, index=['Accuracy','F1_score','Auc'])
 
        return df_ranf_perf
            
        
        
    def Gradient_boosting(self, param='before'):
        '''

        Parameters
        ----------
        test : TYPE, optional
            DESCRIPTION. The default is 'total'.

        Returns
        -------
        df_ranf_perf : TYPE
            DESCRIPTION.

        '''
        
        # 나머지 데이터를 돌리는 작업 -> 버림
        # others_df = self.ADAPTIVE_rule_ins.others_df
        
        X_test = self.X_test
        y_test = self.y_test
        
        ## default parameter value ##
        # learning_rate=0.1, criterion='friedman_mse', max_depth=3
        # min_samples_leaf=1, min_samples_split=2, max_leaf_nodes=None
        
        if param=='before':
        
            gb_clf=GradientBoostingClassifier(n_estimators=1000, random_state=0)
        
            gb_clf.fit(self.X_train,self.y_train)
        
            y_pred = gb_clf.predict(X_test)
            
            grad_acc = np.round(metrics.accuracy_score(y_test, y_pred),4)
            grad_f1 = np.round(metrics.f1_score(y_test, y_pred),4)
            grad_auc = np.round(metrics.roc_auc_score(y_test, y_pred),4)
            
            df_grad_perf = pd.DataFrame({'GradientBoosting_before':[grad_acc,grad_f1,grad_auc]}, index=['Accuracy','F1_score','Auc'])
            
            return df_grad_perf
        
        elif param=='after':
            
            gb_clf=GradientBoostingClassifier(n_estimators=1000, random_state=0)
            
            '''
            # 데이터 작업 #
            X_train = others_df.loc[:, others_df.columns != 'target']
            y_train = others_df.loc[:, others_df.columns == 'target']
            '''
            
            # n_estimators: tree 갯수, max_features: 
            param_grid = [{'learning_rate': np.linspace(0.01, 0.1, 10)}]
            
            #  cv=5
            gs = GridSearchCV(estimator=gb_clf, param_grid=param_grid, scoring='accuracy', verbose=1, n_jobs=-1)

            gs.fit(self.X_train, np.array(self.y_train))
            
            # 모든 모델 중 가장 성능이 좋은 것을 뽑는 best estimator # n_estimators가 제일 높은 모델이 선정되는 것이 대부분
            best_gsb_estimator = gs.best_estimator_
            
            # best parameter
            #n_tree = best_gsb_estimator.n_estimators 
            #depth = best_gsb_estimator.max_depth
            lr = best_gsb_estimator.learning_rate
            #leaf = best_gsb_estimator.min_samples_leaf
            
            y_pred = best_gsb_estimator.predict(X_test)
        
            grad_acc = np.round(metrics.accuracy_score(y_test, y_pred),4)
            grad_f1 = np.round(metrics.f1_score(y_test, y_pred),4)
            grad_auc = np.round(metrics.roc_auc_score(y_test, y_pred),4)
            
            df_grad_perf = pd.DataFrame({'GradientBoosting_after':[grad_acc,grad_f1,grad_auc, lr]}, \
                                        index=['Accuracy', 'F1_score', 'Auc', \
                                               'learning_rate'])
        
            return df_grad_perf
              
        
    def rule_att(self, rule_dic):
        
        # 1. 모든 rule의 attribute를 각각 누적시켜서 비교
        # 2. 그 다음 attribute와 분기 기준이 같으면 rule이 같다고 보고 수를 누적시키지 않음
        # 3. 로우는 해당 rule의 attribute, 비교한 attribute의 누적 수
        #    컬럼은 해당 rule 
        
        # 분석할 것만 골라내기 #
        rule_tr = {}
        
        for key, values in rule_dic.items():
                    
            if key != -1:
                rule_tr[key] = list(values[:-1])
        
        # tuple -> list #
        for num, val_tr in enumerate(list(rule_tr.values())):
            
            list_covert = [list(i) for i in list(rule_tr.values())[num]]
            rule_tr[num] = list_covert

        test_df = pd.DataFrame(columns=[f'rule_{i}' for i in range(len(list(rule_tr.values())))])
        
        each_rule_tr = {}
        each_rule_direction = {}
        
        for num, tr_dir in enumerate(list(rule_tr.values())):
            
            Rule_attribute_ad = []
            
            # 각 rule별 attribute 추출 #
            if len(tr_dir) == 1:
                
                tr = tr_dir[0][0]
                direction = tr_dir[0][1]
                
                att, cond, value = tr.split()
                
                each_rule_tr[num] = att
                each_rule_direction[num] = direction
                
                test_df.loc[f'Rule_attribute_tr', f'rule_{num}'] = tr
                test_df.loc[f'Rule_attribute_direction', f'rule_{num}'] = direction
                
            else:
                
                for num_tr in range(len(tr_dir)):
                    
                    tr = tr_dir[num_tr][0]
                    direction = tr_dir[num_tr][1]
                    
                    att, cond, value = tr.split()
                    
                    each_rule_tr.setdefault(num, [])
                    each_rule_direction.setdefault(num, [])
                    
                    each_rule_tr[num].append(att)
                    each_rule_direction[num].append(direction)
                    
                    Rule_attribute_ad.append(tr)
                    
                test_df.loc['Rule_attribute_tr',f'rule_{num}'] = \
                    np.array(Rule_attribute_ad)
                    
                test_df.loc['Rule_attribute_direction',f'rule_{num}'] = \
                    np.array(each_rule_direction[num])
        
        att_num_base = 0
        
        # 같은 attribute 추출 #
        for att_num, att_val in enumerate(list(each_rule_tr.values())):
            
            appen = []
            
            if att_num == 0:       
                
                att_num_base += 1
                test_df.loc[f'Attribute_cumulative_num', f'rule_{att_num}'] = att_num_base
                
                pass
            
            else:
                
                for other_att_num in range(att_num):
                    base_att = att_val
                    comparison_att = list(each_rule_tr.values())[other_att_num]
                    
                    if type(base_att) == type([]):
                    
                        if comparison_att in base_att:
                            appen.append(True)
                            
                        else:
                            appen.append(False)
                            
                    else: 
        
                        if base_att in comparison_att:
                            appen.append(True)
        
                        else:
                            appen.append(False)
                            
            if True in appen:
                test_df.loc[f'Attribute_cumulative_num', f'rule_{att_num}'] = att_num_base
                    
            elif False in appen:
                att_num_base += 1
                
                test_df.loc[f'Attribute_cumulative_num', f'rule_{att_num}'] = att_num_base

        return test_df
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        