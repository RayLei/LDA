#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 08:52:09 2018

@author: LEIHAO
"""

import numpy as np
import lda_util as lu
import drop_com_wds as dcw
import pandas as pd

directory = '/Users/leihao/Downloads/'
prefix, date_range = 't_trading_' , '_120501_120531.txt'

doc_wds_file = directory + prefix + 'doc_wds' + date_range
tpc_wds_file = directory + prefix + 'tpc_wds' + date_range
org_doc_wds  = lu.matrix_load(doc_wds_file)
org_tpc_wds  = lu.matrix_load(tpc_wds_file)
voc_file     = prefix + 'vocab'   + date_range    
voc          = sorted(lu.vocab_load(directory + voc_file).split('\n')[:-1])

com_wds_ht, new_voc_ht = dcw.comm_wds(org_tpc_wds, voc, pct = 0.4)
com_wds_km, new_voc_km = dcw.kmean_wds(org_tpc_wds, voc)
com_wds_ht2, new_voc_ht2 = dcw.comm_wds(org_tpc_wds, voc, pct = 0.5)


new_ind = pd.Index(set(com_wds_km.index).union(set(com_wds_ht.index)))
new_doc_wds  = dcw.dropcols_coo(org_doc_wds, new_ind)

type(new_voc_km)

new_voc = sorted(set(new_voc_km).intersection(set(new_voc_ht)))

len(new_voc)
new_doc_wds.shape


out_dir = '/Users/leihao/Documents/Git/LDA/Remove_Comm_Words/Output_combine/'
out_prefix = 't_drop_comb_har_'
lu.vocab_write(new_voc, out_dir, out_prefix, 'vocab' )
lu.matrix_dump(new_doc_wds, out_dir, out_prefix, 'doc_wds')
lu.lda_out(new_doc_wds, new_voc, out_dir, out_prefix, 20, 20 )
