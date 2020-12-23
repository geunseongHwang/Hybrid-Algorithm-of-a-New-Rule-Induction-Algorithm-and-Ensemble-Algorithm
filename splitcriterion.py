# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

@author: hyeongyuy
"""

import pandas as pd
import numpy as np

class splitCrit(object):
    
    def __init__(self, min_samples, criterion):
        '''
        MIN_SAMPLES: node의 최소 샘플 수
        CRITERION: 분기기준
        CRITERION_LIST: 분기방법
        '''
        # 이전 것: self.MIN_SAMPLES = min_samples
        
        self.MIN_SAMPLES = min_samples
        self.CRITERION = criterion
        self.CRITERION_LIST = ['gini', 'entropy']
        
    def mk_p_list(self, values):
        
        '''
        노드당 클래스별 갯수를 비율로 나타냄.
        '''
        
        elements, counts = np.unique(values, return_counts = True)
        sum_c = np.sum(counts)
        return [counts[i]/sum_c for i in range(len(elements))]

    def homogeneity(self, p_list):

        #elements, counts = np.unique(target_col,return_counts = True)
        if 'gini' in self.CRITERION:
            homogeneity_ =  1 - np.sum([p**2 for p in p_list])
            return homogeneity_
        elif 'entropy' in self.CRITERION:
            homogeneity_ = -np.sum([p * np.log2(p) for p in p_list])
            return homogeneity_

    def split_criteria(self, left, right, target_values):
        '''
        target_values: data target values
        left_ratio: target_values 비율
        
        
        '''
        bf_split = self.homogeneity(self.mk_p_list(target_values))

        left_ratio = np.sum(left) /len(target_values)
        right_ratio = 1 - left_ratio
        left_node_homog = (left_ratio) * \
            self.homogeneity(self.mk_p_list(target_values[left]))
        right_node_homog = (right_ratio) * \
            self.homogeneity(self.mk_p_list(target_values[right]))
        aft_split = np.nansum([left_node_homog, right_node_homog])

        return bf_split - aft_split

    def get_feature_info(self, data, target_attribute_name):
        feature = data.columns[data.columns != target_attribute_name]
        dtype_dict = {}
        value_dict = {}        
        cand = []
        for f in feature:
            if np.issubdtype(data.loc[:, f].dtype, np.number):
                dtype_dict[f] = 'n'
                value_dict[f] = data.loc[:,f].values
                pre = np.unique(value_dict[f])[1:]
                post = np.unique(value_dict[f])[:-1]
                c_values = (pre + post)/2
                
                for c in c_values:
                    cand.append((f, c))
                    
            else:
                dtype_dict[f] = 'c'
                value_dict[f] = data.loc[:,f].values
                for c in np.unique(value_dict[f]):
                    cand.append((f, c))

                

        return dtype_dict, value_dict, cand
    
    def best_split(self, data, target_attribute_name):
        base_gain=0
        slt_dtype=''
        best_cut=None
        best_feature=''
        left_node_sub_data, right_node_sub_data = \
            pd.DataFrame(columns = data.columns), pd.DataFrame(columns = data.columns)

        target_values =data[target_attribute_name].values
        dtype_dict, value_dict, cand = \
            self.get_feature_info(data, target_attribute_name)
        
        for c in cand:
            dtype = dtype_dict[c[0]]
            feature_value = value_dict[c[0]]
            if dtype =='n':
                left_condtion , right_condtion = \
                    feature_value < c[1], feature_value >= c[1]
            else:
                left_condtion , right_condtion = \
                    feature_value != c[1], feature_value == c[1]

            if (np.sum(left_condtion) >= self.MIN_SAMPLES) \
                    and (np.sum(right_condtion) >= self.MIN_SAMPLES):
                
                gain = self.split_criteria(left_condtion, right_condtion, target_values)

                if (gain > base_gain):
                    base_gain = gain
                    slt_dtype = dtype
                    best_cut = round(c[1], 3)
                    best_feature = c[0]
                    left_node_sub_data = data.loc[left_condtion, : ]
                    right_node_sub_data = data.loc[right_condtion, : ]

        return slt_dtype, best_cut, best_feature, left_node_sub_data, \
             right_node_sub_data

class newSplitCrit(splitCrit):
    def __init__(self, min_samples, lambda_):
        super(newSplitCrit, self).__init__(min_samples, 'inv-gini')
        self.NODE_ID = 0
        self.lambda_ = lambda_
        self.SC_lambda_info=[]
        self.best_SC_lambda_info=[]
        self.DEPTH = 0
        
    def homogeneity(self, p_list):
        homogeneity_ = np.sum([p**2 for p in p_list])
        return homogeneity_

    def split_criteria(self, left_cond, right_cond, target_values, lambda_):
        bf_split = self.homogeneity(self.mk_p_list(target_values))

        left_ratio = np.sum(left_cond) /len(target_values)
        right_ratio = 1 - left_ratio
        
        l_w_homogeneity = ((left_ratio)**lambda_) * \
            self.homogeneity(self.mk_p_list(target_values[left_cond]))
        r_w_homogeneity = ((right_ratio)**lambda_) * \
            self.homogeneity(self.mk_p_list(target_values[right_cond]))
        
        aft_split, max_path = [l_w_homogeneity, 'left'] \
                            if l_w_homogeneity >= r_w_homogeneity else [r_w_homogeneity, 'right']
        return aft_split - bf_split, max_path

    def best_split(self, data, target_attribute_name):
        self.NODE_ID +=1
        dtype_dict, value_dict, cand = \
            super(newSplitCrit, self).get_feature_info(data, target_attribute_name)
        
        target_values = data[target_attribute_name].values
        right_node_sub_data = pd.DataFrame(columns = data.columns)
        left_node_sub_data = pd.DataFrame(columns = data.columns)
        max_path = 'Root'

        base_SC = 0
        slt_dtype =''
        best_cut = None
        best_feature = ''

        for c in cand:
            dtype = dtype_dict[c[0]]
            feature_value = value_dict[c[0]]
            if dtype =='n':
                left_condtion , right_condtion = \
                    feature_value < c[1], feature_value >= c[1]
            else:
                left_condtion , right_condtion = \
                    feature_value != c[1], feature_value == c[1]

            #분기 후에 샘플 수가 MIN_SAMPLES 보다 클 때 각 요소 별 homogeneity 계산
            if (np.sum(left_condtion) >= self.MIN_SAMPLES) and \
                    (np.sum(right_condtion) >= self.MIN_SAMPLES):
    
                aft_SC, mxp = self.split_criteria(left_condtion, right_condtion, target_values, self.lambda_)
    
            
                # aft_SC: temp_SC / SC: base_SC 
                if (aft_SC >= base_SC):
                    base_SC = aft_SC
                    slt_dtype = dtype
                    best_cut = c[1]
                    best_feature = c[0]
                    left_node_sub_data = data.loc[left_condtion, : ]
                    right_node_sub_data = data.loc[right_condtion, : ]
                    max_path = mxp

        return [slt_dtype, best_cut, best_feature, left_node_sub_data, \
                right_node_sub_data, max_path]

                
class adaptiveSplitCrit(splitCrit):
    def __init__(self, min_samples, params):
        super(adaptiveSplitCrit, self).__init__(min_samples, 'inv-gini')
        self.NODE_ID = 0
        self.LAMBDA_RANGE = params
        self.SC_lambda_info=[]
        self.best_SC_lambda_info=[]
        self.DEPTH = 0
        
    def homogeneity(self, p_list):
        homogeneity_ = np.sum([p**2 for p in p_list])
        return homogeneity_
    
    '''
    # 변경 전 
    def split_criteria(self, left_cond, right_cond, target_values, lambda_):

        bf_split = self.homogeneity(self.mk_p_list(target_values))
        
        left_ratio = np.sum(left_cond) /len(target_values)
        right_ratio = 1 - left_ratio

        l_w_homogeneity = ((left_ratio)**lambda_) * \
            self.homogeneity(self.mk_p_list(target_values[left_cond]))
        r_w_homogeneity = ((right_ratio)**lambda_) * \
            self.homogeneity(self.mk_p_list(target_values[right_cond]))
            

        aft_split, max_path = [l_w_homogeneity, 'left'] \
                           if l_w_homogeneity >= r_w_homogeneity else [r_w_homogeneity, 'right']        
        return aft_split - bf_split, max_path
    '''
    
    # 변경 후 
    def split_criteria(self, left_cond, right_cond, target_values, lambda_):

        left_ratio = np.sum(left_cond) /len(target_values)
        right_ratio = 1 - left_ratio
        
        '''
        *left_ratio. right_ratio: 
            해당 분기기준으로 나눴을 때,
            왼쪽과 오른쪽에 해당되는 class의 비율?
            [1을 기준으로 나눠진 형태]
        '''

        l_w_homogeneity = ((left_ratio)**lambda_) * \
            self.homogeneity(self.mk_p_list(target_values[left_cond]))
        r_w_homogeneity = ((right_ratio)**lambda_) * \
            self.homogeneity(self.mk_p_list(target_values[right_cond]))

        '''
        *l_w_homogeneity. r_w_homogeneity: 
            분기기준으로 나눠진 class의 비율에 각각 lambda값을 제곱하고
            거기에 왼쪽과 오른쪽에 해당되는 class의 homogeneity를 곱한 값
            *lambda가 커질수록 homo는 작아짐
        '''

        aft_split, max_path = [l_w_homogeneity, 'left'] \
                           if l_w_homogeneity >= r_w_homogeneity else [r_w_homogeneity, 'right']        
        
        return aft_split, max_path

    def best_split(self, data, target_attribute_name):
        self.NODE_ID +=1
        
        '''
        *dtype_dict: feature별 type
        *value_dict: Tree 생성 시 feature마다의 데이터  (array형태 ) 
        *cand: 기존 데이터의 feature type마다 분기기준 여러개를 후보로 생성
        
        '''
        dtype_dict, value_dict, cand = \
            super(adaptiveSplitCrit, self).get_feature_info(data, target_attribute_name)
    
        
        target_values = data[target_attribute_name].values
        right_node_sub_data = pd.DataFrame(columns = data.columns)
        left_node_sub_data = pd.DataFrame(columns = data.columns)
        max_path = 'Root'
        
        '''
        *target_values: 분기마다 남은 target value 
        '''
        
        best_lambda_dict = {}
        for lambda_ in self.LAMBDA_RANGE:
            base_SC = 0
            slt_dtype =''
            best_cut = None
            best_feature = ''
            
            '''
            *lambda_: 주어진 lambda마다 적용되는게 달라짐
            cand: 1~10까지 값이있는 feature가 있으면 1.5, 2.5 단위로 값 사이값을 
                  분기기준값으로 넣음 (범주형이면 그대로)
            '''

            for c in cand:
                dtype = dtype_dict[c[0]]
                feature_value = value_dict[c[0]]
                
                '''
                *c: cand에서 저장된 attribute와 해당 분기기준 값
                *dtype: 해당 분기기준 속성타입
                *feature_value: array에서 나온 데이터(feature 그대로의 값) ?
                
                '''
                      
                if dtype =='n':
                    left_condtion , right_condtion = \
                        feature_value < c[1], feature_value >= c[1]
                else:
                    left_condtion , right_condtion = \
                        feature_value != c[1], feature_value == c[1]
                
                
                '''
                *left_condtion. right_condtion: 
                    feature_value의 데이터 그대로 분기기준을 적용했을 때,
                    해당되면 True 아니면 False
                '''
            
                
                # 조건 아래 최소 샘플을 만족하면 트리의 노드를 생성함
                if (np.sum(left_condtion) >= self.MIN_SAMPLES) and \
                        (np.sum(right_condtion) >= self.MIN_SAMPLES):
        
                    aft_SC, mxp= self.split_criteria(left_condtion, right_condtion, target_values, lambda_)

                    '''
                    *aft_SC: 위에서 l_w_homogeneity. r_w_homogeneity 중 가장 높은
                             w_homo선정
                    *mxp: left or right
                        
                    '''
        
                    # 만약 이전에 저장된 base_SC보다 크면
                    # aft_SC 저장
                    if (aft_SC >= base_SC):
                        
                        base_SC = aft_SC
                        slt_dtype = dtype
                        best_cut = c[1]
                        best_feature = c[0]
                        left_node_sub_data = data.loc[left_condtion, : ]
                        right_node_sub_data = data.loc[right_condtion, : ]
                        max_path = mxp
                        
                    '''
                    # best information #
                    *slt_dtype: dtype data type
                    *best_feature: 분기기준 attribute
                    *best_cut: 분기기준 value
                    *left_node_sub_data: 왼쪽 분기기준으로 나눈 instance
                    *right_node_sub_data: 오른쪽 분기기준으로 나눈 instance
                    *max_path: *mxp: left or right
                    '''
       
            best_lambda_dict[lambda_] = \
                [slt_dtype, best_cut, best_feature, left_node_sub_data, \
                right_node_sub_data, max_path, base_SC]
            
  
            '''
            *best_lambda_dict: 
                best lambda로 선택된 best information
            '''
            
            if best_feature !='':
                self.SC_lambda_info.append([self.NODE_ID, self.DEPTH-1, lambda_, slt_dtype, \
                                            best_cut, best_feature, base_SC])
                        
        if best_feature =='':

            return [slt_dtype, best_cut, best_feature, left_node_sub_data, \
                    right_node_sub_data, max_path]

            '''
        # 변경전
        else:
            best_sc_list = np.array([splt[-1] \
                for splt in best_lambda_dict.values()])
            diff = best_sc_list[:-1] - best_sc_list[1:]
            diff_check = 0
            idx = 0
            for i, d in enumerate(diff):
                if d >= diff_check:
                    diff_check = d
                    idx = i

            best_lambda = self.LAMBDA_RANGE[idx]
            
            self.best_SC_lambda_info.append(\
                    [self.NODE_ID, self.DEPTH-1, best_lambda, best_sc_list[idx]])

            return best_lambda_dict[best_lambda][:-1]
            
        
            '''
        #변경후
        else:
            
            '''
            *best_sc_list: 
                best_lambda_dict들의 마지막에 저장된 value(W_homogeneity)를
                하나의 array로 저장 (왜하는거지 이거)
                
            *diff:
                맨 뒤의 동질성과 맨 앞의 동질성을 제외한 나머지 값을 뺌
            *d: 
                동질성 차이가 제일 큰 값을 diff_check에 저장 
            *i: 
                가장 차이가 큰 d의 index 수를 idx에 저장
            *idx: 
                LAMBDA_RANGE 중 best LAMBDA를 뽑기 위한 index
                
            *best_lambda:
                best_lambda를 선정
            *
            
            '''
            
            best_sc_list = np.array([splt[-1] \
                for splt in best_lambda_dict.values()])
                  
            diff = best_sc_list[:-1] - best_sc_list[1:]
            
            
            diff_check = 0
            #diff_check = 1.0
            idx = 0
            for i, d in enumerate(diff):     
                
                if d >= diff_check:
                #if d <= diff_check:
                    diff_check = d
                    idx = i
                else: break # this line
            
            if idx+1 == len(diff):
                

                idx += 1
                best_lambda = self.LAMBDA_RANGE[idx]
                
            else:
                best_lambda = self.LAMBDA_RANGE[idx]
            
            self.best_SC_lambda_info.append(\
                    [self.NODE_ID, self.DEPTH-1, best_lambda, best_sc_list[idx]])
            
            
            
            return best_lambda_dict[best_lambda][:-1]
        
        
        
        
        
        
        
        
        
        
        