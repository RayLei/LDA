#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:04:12 2018

@author: LEIHAO
"""

import numpy as np
import lda


vocab=['law','finance','math','physics','art']
topic_num=5

dat_matrix=np.ones((10,5),dtype=int)*200
rand_mat=np.random.randint(0,11,(10,5),dtype=int)
dat_mat=dat_matrix+rand_mat                  

model=lda.LDA(n_topics=topic_num, n_iter=1500, random_state=1)
model.fit(dat_mat)

topic_word2=model.topic_word_

n_top_words=5
for i, topic_dist in enumerate(topic_word):
    topic_words=np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
    
dat_mat2=np.ones((2,5),dtype=int)*200
dat_mat2[:,4]=np.random.randint(1,11,2)

model2=lda.LDA(n_topics=topic_num, n_iter=1500, random_state=1)
model2.fit(dat_mat2)

topic_word2=model2.topic_word_

#n_top_words=5
for i, topic_dist in enumerate(topic_word2):
    topic_words=np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_word2)))