#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:36:14 2018

@author: LEIHAO
"""
import numpy as np
from scipy.sparse import csr_matrix
import ast

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

# ----------------------------
# read the word counting saved
# ----------------------------
enc='utf-8'
with open('/Users/leihao/Documents/t_memory_160101_160102.txt','r',encoding=enc) as f:
    dat=f.read().splitlines()

# -------------------------------------------
# The last line is the total number of pairs;
# remove it.
# -------------------------------------------
dat=dat[:-1]


#dat=[[(3,'apple'),(1,'orange'),(2,'3.14')],
#      [(2,'banana'),(2,'orange')]
#        ]
# -------------------------
# find the vocabulary set
# -------------------------
all_words=set()
for item in dat:
    item=ast.literal_eval(item) # the list is saved as a string
    for num, word in item:
        if len(word)>=3 and not word.isdigit() and not is_float(word):
            all_words.add(word)

vocab=sorted(all_words)
#del all_words
# ----------------------------------------
# transform counting into a sparse matrix
# ----------------------------------------
m, n=len(dat), len(vocab)
den_mat=np.zeros((m,n),dtype='H')
for item in dat:
    item=ast.literal_eval(item)
    row=dat.index(item)
    for num, word in item:
        if word in vocab:
            col=vocab.index(word)
            den_mat[row,col]=num
                   
spa_mat=csr_matrix(den_mat)
