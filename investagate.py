#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:29:44 2018

@author: LEIHAO
"""

from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag, regexp_tokenize
from nltk.stem import WordNetLemmatizer

txt= ['This is a test with numbers. For example, 2.', \
      'A 2nd test.check']

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN    
    
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         self.word_pos=pos_tag(word_tokenize(doc))
         return [self.wnl.lemmatize(w,get_wordnet_pos(p)) for w,p in self.word_pos 
                 if len(w)>=3 and not w.isdigit() and not is_float(w) ]
def article2matrix(article):
    cnt_vectorizer = CountVectorizer(token_pattern=r'\b[a-zA-Z]+\b')
    doc_term_mat = cnt_vectorizer.fit_transform(article)
    vocab = cnt_vectorizer.get_feature_names()
    return (doc_term_mat, vocab)

t1, v1=article2matrix(txt)
t2, v2=article2matrix(txt)
t3, v3=article2matrix(txt)
t4, v4=article2matrix(txt)
t5, v5=article2matrix(txt)
