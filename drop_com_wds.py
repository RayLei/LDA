#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 18:19:48 2018
t
@author: LEIHAO
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans


def comm_wds(tpc_wds_mat, voc, pct =0.5, K = 20):
    
    if isinstance(tpc_wds_mat, np.ndarray) is False:
        try:
            tpc_wds_mat = np.asmatrix(tpc_wds_mat)
        except:
            print("Unknown type of 'tpc_wds_mat'. \
                  The 'tpc_wds_mat' should be of type 'numpy.matrix' ")
    
    if isinstance(tpc_wds_mat, np.matrix) is False:
        try:
            tpc_wds_mat = np.asmatrix(tpc_wds_mat)
        except:
            print("Unknown type of 'tpc_wds_mat'. \
                  The 'tpc_wds_mat' should be of type 'numpy.matrix' ")
    
    if isinstance(voc, pd.Series) is False:
        try:
            voc = pd.Series(voc)
        except:
            print("Unknown type of 'voc'. \
                  The 'voc' should be of type 'list', 'tuple', 'numpy.1darray' or 'pandas.Series'. '")
      
    ind_mat = tpc_wds_mat - tpc_wds_mat.min(axis=1)
    wds_ind = (ind_mat > 0).sum(axis=0).A.squeeze()
    com_wds = voc[wds_ind >= pct * K]
    new_voc = voc[wds_ind < pct*K].values
    return (com_wds, new_voc)

def kmean_wds(tpc_wds_mat, voc):
    if isinstance(tpc_wds_mat, np.ndarray) is False:
        try:
            tpc_wds_mat = np.asmatrix(tpc_wds_mat)
        except:
            print("Unknown type of 'tpc_wds_mat'. \
                  The 'tpc_wds_mat' should be of type 'numpy.matrix' ")
    
    if isinstance(tpc_wds_mat, np.matrix) is False:
        try:
            tpc_wds_mat = np.asmatrix(tpc_wds_mat)
        except:
            print("Unknown type of 'tpc_wds_mat'. \
                  The 'tpc_wds_mat' should be of type 'numpy.matrix' ")
    
    if isinstance(voc, pd.Series) is False:
        try:
            voc = pd.Series(voc)
        except:
            print("Unknown type of 'voc'. \
                  The 'voc' should be of type 'list', 'tuple', 'numpy.1darray' or 'pandas.Series'. '")
      
    X = tpc_wds_mat.T.A
    kmeans = KMeans(n_clusters = 2, random_state = 0).fit(X)
    com_wds = pd.Series(voc)[kmeans.labels_.astype(bool)]
    new_voc = [item for item in voc if item not in com_wds.values]
    return (com_wds, new_voc)

def dropcols_fancy(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    keep = ~np.in1d(np.arange(M.shape[1]), idx_to_drop, assume_unique=True)
    return M[:, np.where(keep)[0]]

def dropcols_coo(M, idx_to_drop):
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()

