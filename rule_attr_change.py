# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:05:34 2020

@author: geunseong
"""

import numpy as np
import pandas as pd


class rule_attribute_change(object):
    
    def rule_def(rule_dic_ex): 
    
        rule_dic = {}
        
        #rule_dic = dict.fromkeys(list(rule_dic_ex.keys())[:-1], [])
        
        for key, val in list(rule_dic_ex.items())[:-1]:
            
            for sc_base, booln in val[:-1]:
                
                print(sc_base)
                
                att, cond, value = sc_base.split()
                
                if cond == '>=':
                
                    if booln:
                        
                        cond_adv = '>='
                        
                        sc_adv = ' '.join([att,cond_adv, value])
                        
                        rule_dic.setdefault(key, []).append(sc_adv)
                        
                    else:
                        
                        cond_adv = '<'
                        
                        sc_adv = ' '.join([att,cond_adv, value])
                        
                        rule_dic.setdefault(key, []).append(sc_adv)
                        
                elif cond == '==':
                    
                    if booln:
                        
                        cond_adv = '=='
                        
                        sc_adv = ' '.join([att,cond_adv, value])
                        
                        rule_dic.setdefault(key, []).append(sc_adv)
                        
                    else:
                        
                        cond_adv = '!='
                        
                        sc_adv = ' '.join([att,cond_adv, value])
                        
                        rule_dic.setdefault(key, []).append(sc_adv)
                        
        df_attr = pd.DataFrame(pd.Series(rule_dic).reset_index()).set_axis(['Key','split_criterion'],1,inplace=False)
        others = list(rule_dic_ex.items())[-1]
        
        return df_attr, others

