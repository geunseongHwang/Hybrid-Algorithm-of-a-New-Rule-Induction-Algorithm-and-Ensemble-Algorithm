# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 04:39:33 2018

@author: hyeongyuy
"""
import numpy as np


class visGraph(object):
    def __init__(self):
        #node num
        self.NODE_NUM = 0
        #leaf node info in tree graph
        self.LEAF_BASE = \
            '[label=\"predict = {}\\nhomogeneity = {}\\ncoverage = {}\\nsamples/class = {}\\nmax_path = {}\"] ;'
        self.ROOT_LEAF_BASE = \
            '[label="Root_node\\npredict = {}\\nhomogeneity = {}\\nsamples = {}\\nsamples/class = {}\"] ;'
        self.INTER_NODE_BASE = \
            '[label=\"{} {} {}\\nsamples = {}\\nsamples/class = {}\\nmax_path = {}\"] ;'    
            
    def node_info(self, cnt_list, n_data, max_path, root = False):
        
        base_string = self.ROOT_LEAF_BASE if root else self.LEAF_BASE
        return base_string.format(\
            np.argmax(cnt_list), np.round(max(cnt_list)/sum(cnt_list),3), \
            np.round(sum(cnt_list)/n_data,3), cnt_list, str(max_path)) 
        
    def get_graph_tree(self, feature, cut_val, cnt_list, condition_list, sub_tree_list, max_path):
        pprint_tree={}
        pprint_tree[self.INTER_NODE_BASE.\
            format(feature, condition_list[0], cut_val, sum(cnt_list), cnt_list, max_path)]\
            = sub_tree_list[0]
        pprint_tree[self.INTER_NODE_BASE.\
            format(feature, condition_list[1], cut_val, sum(cnt_list), cnt_list, max_path)]\
            = sub_tree_list[1]

        return pprint_tree
    
    def get_node_edge(self, pprint_tree, node = 'digraph Tree {\nnode [shape=box] ;', \
                        edge ='', node_no=0):
        p_node_no = self.NODE_NUM
        if not isinstance(pprint_tree,dict):
            node += '\n{} {}'.format(p_node_no, pprint_tree)
            return node, edge
        if isinstance(pprint_tree,dict):
            if list(pprint_tree.keys())[0] == 'Root_node':
                return node, edge
            
            key = [k for k in pprint_tree.keys() if k.split()[1] == '>=']
            if len(key) >= 1:
                key = key[0]
                node += '\n{} {}'.format(self.NODE_NUM, key)
                self.NODE_NUM += 1
                edge += '\n{} -> {} [labeldistance=2.5, labelangle=45] ;'\
                    .format(p_node_no , self.NODE_NUM )
                
                rpl_key = key.replace('>=', '<')
      
                left_sub_pprint_tree= pprint_tree[rpl_key]
                node, edge = self.get_node_edge(left_sub_pprint_tree, node, edge)
                
                self.NODE_NUM += 1
                edge += '\n{} -> {};'.format(p_node_no , self.NODE_NUM)
                right_sub_pprint_tree = pprint_tree[key]
                node, edge = self.get_node_edge(right_sub_pprint_tree, node, edge)
    
            else:
                key = [k for k in pprint_tree.keys() if k.split()[1] == '=='][0]
                node += '\n{} {}'.format(p_node_no , key)
                
                self.NODE_NUM += 1

                edge += '\n{} -> {} [labeldistance=2.5, labelangle=45] ;'\
                    .format(p_node_no , self.NODE_NUM)
                
                rpl_key = key.replace('==', '!=')
                    
                left_sub_pprint_tree= pprint_tree[rpl_key]
                node, edge= self.get_node_edge(left_sub_pprint_tree, node, edge)
                
                self.NODE_NUM += 1
                right_sub_pprint_tree = pprint_tree[key]
                edge += '\n{} -> {};'.format(p_node_no , self.NODE_NUM)
                node, edge = self.get_node_edge(right_sub_pprint_tree, node, edge)
    
        return node, edge
    
    def tree_to_graph(self, pprint_tree):
        node, edge = self.get_node_edge(pprint_tree)
        return node + edge+'\n}'
   
