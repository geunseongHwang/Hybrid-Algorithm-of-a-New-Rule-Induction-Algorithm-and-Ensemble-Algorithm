# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 02:51:29 2020

@author: User
"""
import ast
import numpy as np
import pandas as pd
from typing import *

class main_model_info(object):
    
    def __init__(self, data_info):
        
        self.data_info = data_info
    
    
    def model_collection_info(self, cum_coverage_rate: int):
        
        data = self.data_info 
        
        data.index = data.iloc[:,0]
        data = data.iloc[:,1:]
        
        rule_ind = data.filter(like='번째',axis=0).index
        
        final_model_info_list = []
        
        #1. 30개를 나눠서 정리
        model_info_list = []
        for num, _ in enumerate(rule_ind):
            
            if num != 29:
                info = data.loc[rule_ind[num]:rule_ind[num+1],:].iloc[:-1,:]
                
            else:
                info = data.loc[rule_ind[num]:,:]
                
            model_info_list.append(info)
            
            
        #2. 정리된 데이터마다 원하는 데로 정리
        for info_num in range(len(model_info_list)):
            
            print(f'{info_num}번째')
            
            #2.1 row nan 제거
            model_info_list[info_num].dropna(axis=0, inplace=True)
            
            #2.2 model별 자르기전 인덱스 구별
            NewSC_ind = model_info_list[info_num].filter(like='New Splitting Creterion',axis=0).index[0]
            Rule_ind = model_info_list[info_num].filter(like='Rule Induction',axis=0).index[0]
            CART_ind = model_info_list[info_num].filter(like='Decision Tree (CART-gini)',axis=0).index[0]
            
            #2.2.1 model별로 구분1
            NewSC = model_info_list[info_num].loc[NewSC_ind:Rule_ind,:][:-1]
            Rule = model_info_list[info_num].loc[Rule_ind:CART_ind,:][:-1]
            CART = model_info_list[info_num].loc[CART_ind:,:]
            
            #2.2.2 model별로 구분2(전처리)
            NewSC.columns = NewSC.iloc[0,:]
            NewSC = NewSC.iloc[1:,:]
            NewSC.index.name = None
            
            Rule.columns = Rule.iloc[0,:]
            Rule = Rule.iloc[1:,:]
            Rule.index.name = None
            
            CART.columns = CART.iloc[0,:]
            CART = CART.iloc[1:,:]
            CART.index.name = None
            
            #2.3 모델별 homogeneity기준 내림차순 정렬 
            NewSC_sort = NewSC.sort_values(by='homogeneity' ,ascending=False)
            Rule_sort = Rule.sort_values(by='homogeneity' ,ascending=False)
            CART_sort = CART.sort_values(by='homogeneity' ,ascending=False)
            
            # cart index 변환 (분석하기 까다로워서 변경함)
            #CART_sort.index = [f'rule_{i}' for i in CART_sort.index]
            
            #2.4 정렬 후 70% 이상 coverage끼리 묶기
            
            #2.4.1 coverage 누적 
            NewSC_cum = NewSC_sort.loc[:,'coverage'].cumsum()
            Rule_cum = Rule_sort.loc[:,'coverage'].cumsum()
            CART_cum = CART_sort.loc[:,'coverage'].cumsum()
            
            #2.4.2 누적 coverage 중 rule 선택

            
            if len(NewSC_cum.index) > 2:
            
                NewSC_rule_sel = \
                    [ind for ind, val in zip(NewSC_cum.index, NewSC_cum.values) \
                     if val > cum_coverage_rate][:-1][0]
                        
            elif len(NewSC_cum.index) == 2:
                NewSC_rule_sel = \
                    list(NewSC_cum.index)[0]
                    
            else:
                
                NewSC_rule_sel = \
                    list(NewSC_cum.index)[-1]
                        
            if len(Rule_cum.index) > 1:
                
                Rule_rule_sel = \
                    [ind for ind, val in zip(Rule_cum.index, Rule_cum.values) \
                     if val > cum_coverage_rate]
            
            else:
                Rule_rule_sel = \
                    list(Rule_cum.index)[-1]
                    
            if len(CART_cum.index) > 2:
                CART_rule_sel = \
                    [ind for ind, val in zip(CART_cum.index, CART_cum.values) \
                     if val > cum_coverage_rate][:-1][0]
            
            
            elif len(CART_cum.index) == 2:
                CART_rule_sel = \
                    list(CART_cum.index)[0]
                    
            else:
                CART_rule_sel = \
                    list(CART_cum.index)[-1]

            # rule 중 누적값이 전체 규칙으로 생성되면 뒤에 규칙이 없으므로 오류가 생김 
            # 그래서 조건을 붙임(2개 이상이 아니라 1개만 생성되면)
            
            if type(Rule_rule_sel) == list:
                if len(Rule_rule_sel) != 1:
                    Rule_rule_sel = Rule_rule_sel[:-1][0]
                    
                else:
                    Rule_rule_sel = Rule_rule_sel[0]
        
                
            NewSC_cover = NewSC_sort.loc[:NewSC_rule_sel,:]
            Rule_cover = Rule_sort.loc[:Rule_rule_sel,:]
            CART_cover = CART_sort.loc[:CART_rule_sel,:]
            
            #2.4.3 type 변경
            NewSC_cover.iloc[:,:-1] = NewSC_cover.iloc[:,:-1].astype('float')
            Rule_cover.iloc[:,:-1] = Rule_cover.iloc[:,:-1].astype('float')
            CART_cover.iloc[:,:-1] = CART_cover.iloc[:,:-1].astype('float')
            
            #2.5 model별 수치 정리
            ''' 
            순서대로 
            pred: 그대로
            depth: 그대로
            homogeneity : 평균 / 중위수
            lift : 평균 / 중위수
            coverage : 누적합
            homo*cover : 누적합(제외)
            '''
            
            model_list = [NewSC_cover, Rule_cover, CART_cover]
            
            for model in model_list:
                
                # homogeneity, lift 평균 작업 #
                homo, lift = model[['homogeneity','lift']].mean()
                
                # coverage, homo*cover 누적합 후 마지막꺼를 기준으로 추출 #
                row_cums = model['coverage'].cumsum()
                coverage = row_cums.iloc[-1,]
                
                # string 형태의 lists를 다시 list 형태로 원상복구 후 #
                # 모델의 규칙별 갯수의 평균 #
                
                sc_num = len(model['split_criterion'])
                
                row_sc_sum = [len(ast.literal_eval(model['split_criterion'][sc_n])) \
                     for sc_n in range(sc_num)]
                    
                sc_mean = np.mean(row_sc_sum)
                
                # 모델별 총 info 정리 #
                
                info_rw = [np.nan, np.nan, homo, lift, coverage, sc_mean]
                
                info_rw = np.round(info_rw,4)
                
                model.loc['statistic' , : ] = info_rw
            
            #2.6 마무리된 모델의 데이터 프레임 리스트를 하나로 합치기 #
            final_model_info = pd.concat([NewSC_cover,Rule_cover,CART_cover], \
                          keys=['New Splitting Creterion', 'Rule Induction', 'Decision Tree (CART-gini)'], \
                          names = [f'{info_num}번째'])
                
            final_model_info_list.append(final_model_info)
            
            #print(f'final_model_info: {final_model_info}')
            
            
        return final_model_info_list
    
    def reflect_concise_rule(self, data_model_info: list, df_concise_rule: list):
        
        New_data_model_info = []
        
        for data_num in range(0, len(data_model_info)):
            data_rule_info_ind =  data_model_info[data_num].loc['Rule Induction'].index[:-1]
            base_rule = df_concise_rule[data_num].filter(like='base_rule',axis=1)
            base_rule.columns = [f'rule_{i}' for i in range(1, len(base_rule.columns)+1)]
            
            # base_rule에서 적용된 Rule_attrs를 선택 #
            sel_cover_rule = base_rule.loc['Rule_attrs', :][data_rule_info_ind]
            
            # rule 0가 있다면 1을 넣음 #
            for rul in sel_cover_rule.index:
                if rul == 'rule_0':
                    sel_cover_rule[rul] = 1
            
            # Rule_static_mean의 평균을 data_model_info의 Rule의 평균으로 다시 적용 #
            Rule_static_mean = np.mean(sel_cover_rule)
            
            data_model_info[data_num].loc['Rule Induction'].loc['statistic','split_criterion'] = \
                Rule_static_mean
            
            New_data_model_info.append(data_model_info[data_num])
            
        return New_data_model_info
        
    def total_model_info(self, final_model_info_list: list):
        '''
        # model별 통계량 평균을 구하기 위한 코드 #
        '''
        #self.total_model_info
        
        df_NewSC = pd.DataFrame()
        df_Rule = pd.DataFrame()
        df_CART = pd.DataFrame()
        
        for model_num in range(len(final_model_info_list)):
        
            NewSC_static = final_model_info_list[model_num].loc['New Splitting Creterion', 'statistic']
            Rule_static = final_model_info_list[model_num].loc['Rule Induction', 'statistic']
            CART_static = final_model_info_list[model_num].loc['Decision Tree (CART-gini)', 'statistic']
            
            #data_model_info[model_num].index.names[0]
            
            df_NewSC = pd.concat([df_NewSC, NewSC_static], axis=1)
            df_Rule = pd.concat([df_Rule, Rule_static], axis=1)
            df_CART = pd.concat([df_CART, CART_static], axis=1)
            
        #3. 모델별 총 평균 #
        list_model = [df_NewSC, df_Rule, df_CART]
        
        info_mean = [np.mean(model, axis=1) for model in list_model]
        
        final_total_model_info = pd.DataFrame(info_mean, \
                                              index=[f'New Splitting Creterion', \
                                                     'Rule Induction', \
                                                     'Decision Tree (CART-gini)'])
        
        final_total_model_info = np.round(final_total_model_info, 4)
        
        return final_total_model_info
            
            

        
        
        
        
        