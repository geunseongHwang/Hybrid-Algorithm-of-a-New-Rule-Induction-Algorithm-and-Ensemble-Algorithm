# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:58:43 2020

@author: User
"""
import pandas as pd
import numpy as np

class Total_comparison(object):
    
    def rule_dt_comparison(df, rule):
        
        df = df.copy()
        
        # rule 작업
        rule_col = df.filter(like='rule',axis=1).columns
        df_rule_col = df.loc[:,rule_col]
        df_rule_cover = pd.DataFrame(columns=df_rule_col.columns)
        
        # 총 data 각 rule Homogeneity 평균 
        df_rule_homo = df_rule_col.loc['Homogeneity',:].mean()
        df_rule_homo = np.round(df_rule_homo,4)
        df_rule_cover.loc['homogeneity_mean',:] = df_rule_homo
        
        # 개별 data 각 rule별 누적 Train_coverage 평균 
        df_rule_cov_m = df_rule_col.loc['Train_cumulative_coverage',:].mean()
        df_rule_cov_m = np.round(df_rule_cov_m,4)
        df_rule_cover.loc['Coverage_Mean',:] = df_rule_cov_m
        
        df_rule_cov_dian = df_rule_col.loc['Train_cumulative_coverage',:].median()
        df_rule_cov_dian = np.round(df_rule_cov_dian,4)
        df_rule_cover.loc['Coverage_Median',:] = df_rule_cov_dian
        
        # 각 랜덤 데이터별 생성된 rule 수
        rule_count = df_rule_col.loc['Train_cumulative_coverage', :].count(axis='index')
        df_rule_cover.loc['rule_count',:] = rule_count
        
        # others 작업
        df_others_col = df.loc[:,'others']
        df_others_cover = pd.DataFrame(columns=['Others_Mean', 'Others_Median'])
        
        # others의 평균 coverage 비율 
        df_others_cover.loc['Train_coverage', 'Others_Mean'] = df_others_col['Train_coverage'].mean()
        df_others_cover.loc['Test_coverage', 'Others_Mean'] = df_others_col['Test_coverage'].mean()
        
        df_others_cover.loc['Train_coverage', 'Others_Median'] = df_others_col['Train_coverage'].median()
        df_others_cover.loc['Test_coverage', 'Others_Median'] = df_others_col['Test_coverage'].median()
        
        
        df_others_cover = np.round(df_others_cover,4)


        # dt 작업
        dt_col = df.filter(like='DT_Depth',axis=1).columns
        dt_col = np.sort(dt_col)
        df_dt_col = df.loc[:,dt_col]
        df_dt_cover = pd.DataFrame(columns=df_dt_col.columns)
        
        # 총 data 각 dt Homogeneity 평균 
        df_dt_homo = df_dt_col.loc['Homogeneity',:].mean()
        df_dt_homo = np.round(df_dt_homo,4)
        df_dt_cover.loc['homogeneity_mean',:] = df_dt_homo
        
        # 개별 data 각 DT별 누적 Train_coverage 평균
        df_dt_cov_m = df_dt_col.loc['Train_coverage',:].mean()
        df_dt_cov_m = np.round(df_dt_cov_m,4)
        df_dt_cover.loc['Coverage_Mean',:] = df_dt_cov_m
        
        df_dt_cov_dian = df_dt_col.loc['Train_coverage',:].median()
        df_dt_cov_dian = np.round(df_dt_cov_dian,4)
        df_dt_cover.loc['Coverage_Median',:] = df_dt_cov_dian
        
        # Model별 Metric
        if rule != 1:
        
            columns = ['Decision_tree','Total_performance(20%_DT)','Total_performance(20%_ranfo)', \
                       'Randomforest',  \
                       'GradientBoosting_before']
                
        else:
                    
            columns = ['Decision_tree','Total_Rule', 'Randomforest',  \
                       'GradientBoosting_before']
            
        rows = ['Accuracy', 'Auc', 'F1_score']
            
        df_metric_av = pd.DataFrame(index=rows, columns=columns)
        
        acc_av = np.round(df.loc['Accuracy',columns].mean(), 4)
        auc_av = np.round(df.loc['Auc',columns].mean(), 4)
        f1_av = np.round(df.loc['F1_score',columns].mean(), 4)
        
        df_metric_av.loc['Accuracy',columns] = acc_av
        df_metric_av.loc['Auc',columns] = auc_av
        df_metric_av.loc['F1_score',columns] = f1_av
        
        df_metric_dian = pd.DataFrame(index=rows, columns=columns)
        
        acc_dian = np.round(df.loc['Accuracy',columns].median(), 4)
        auc_dian = np.round(df.loc['Auc',columns].median(), 4)
        f1_dian = np.round(df.loc['F1_score',columns].median(), 4)
        
        df_metric_dian.loc['Accuracy',columns] = acc_dian
        df_metric_dian.loc['Auc',columns] = auc_dian
        df_metric_dian.loc['F1_score',columns] = f1_dian
        
        df_total_sum = pd.concat([df_rule_cover, df_dt_cover], axis=1)
        
        df_final = [df_total_sum, df_others_cover, df_metric_av, df_metric_dian]
        
        return df_final
        
    