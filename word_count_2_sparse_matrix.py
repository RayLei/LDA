#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 11:36:14 2018

@author: LEIHAO
"""
import numpy as np
from scipy.sparse import csr_matrix, save_npz, lil_matrix
import ast
import lda


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
spa_mat=lil_matrix((m,n),dtype=int)
for item in dat:
    row=dat.index(item)
    item=ast.literal_eval(item)    
    for num, word in item:
        if word in vocab:
            col=vocab.index(word)
            spa_mat[row,col]=num
                   
#spa_mat=csr_matrix(den_mat)

with open('/Users/leihao/Documents/t_sparse_matrix_160101_160102.txt','wb+') as f:
        #f.write("The matrix shape: " + str(sparse_matrix.shape)+"\n")
        save_npz(f,spa_mat,True)



for num_of_topics in (20,30,40,50):
        num_of_topics=20
        model=lda.LDA(n_topics=num_of_topics,n_iter=1500,random_state=1)
        model.fit(spa_mat)
        topic_word=model.topic_word_
        n_top_words=20
#        with open(directory+job_type+'_topic'+str(num_of_topics)+'_keywords20_'+date_range+'.txt','w+') as f:     
#                for i, topic_dist in enumerate(topic_word):
#                        topic_words=np.array(sorted(all_words))[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#                        f.write('Topic {0} : {1}\n'.format(i, ' '.join(topic_words).encode("utf-8")))