#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:14:24 2018

@author: LEIHAO
"""

import pyLDAvis
import pickle as pk


def mat_load(file):
    with open(file,'rb') as f:
        mat = pk.load(f)
    return mat

def voc_load(file):
    with open(file, 'r') as f:
        voc=f.read()
    return voc

directory = '/Users/leihao/Documents/Git/LDA/Remove_Comm_Words/Output_combine/'
prefix, date_range = 't_drop_comb_har_' , '_120501_120531.txt'

#doc_wds_file = prefix + 'doc_wds' + date_range
#tpc_wds_file = prefix + 'tpc_wds' + date_range
#doc_tpc_file = prefix + 'doc_tpc' + date_range
#doc_wds_mat=mat_load(directory + doc_wds_file)

data_input = []
mat_type_list = ['doc_wds', 'tpc_wds', 'doc_tpc']
for item in mat_type_list: 
    file = prefix + item + date_range
    mat  = mat_load(directory + file)
    data_input.append(mat)

voc_file     = prefix + 'vocab'   + date_range    
voc = sorted(voc_load(directory + voc_file).split('\n')[:-1])
data_input.append(voc)
#data_input.index = mat_type_list.append('vocab')

data = {'topic_term_dists': data_input[1], 
        'doc_topic_dists': data_input[2],
        'doc_lengths': data_input[0].sum(axis=1).A.squeeze(),
        'vocab': data_input[3],
        'term_frequency': data_input[0].sum(axis=0).A.squeeze()}

vis_data = pyLDAvis.prepare(**data)
pyLDAvis.save_html(vis_data, directory+'t_drop_comb_har_ldavis.html')
