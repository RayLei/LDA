#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:49:26 2018

@author: LEIHAO
"""

import os
import io
import types
from textblob          import TextBlob, Word
from nltk.corpus       import stopwords
from nltk.stem.wordnet import WordNetLemmatizer as lemma

### Functions

# Function to convert Penn-Treebank POS tags to simplified (WordNet) POS tags
def posWN(posTB):
    if posTB.startswith('J'):
        return 'a'
    elif posTB.startswith('V'):
        return 'v' 
    elif posTB.startswith('N'):
        return 'n'
    elif posTB.startswith('R'):
        return 'r'
    elif posTB.startswith('A'):
        return 's'
    else:
        return ''

# POS function
def pos(blob):
    #tok   = [token[0] for token in blob.pos_tags if len(token[0])>3]
    #tokW  = [Word(token) for token in tok]
    #tokn  = len(tok)
    #posTB = [pos[1] for pos in blob.pos_tags if len(pos[0])>3]
    #posW  = [posWN(TB) for TB in posTB]
    posW = [posWN(pos[1]) for pos in blob.pos_tags if len(pos[0])>3]
    return posW

# Tokenizer function
def token(blob):
    tok   = [token[0] for token in blob.pos_tags if len(token[0])>3]
    return tok

# Lemmatizer function
def lem(article):
    from nltk.stem.wordnet import WordNetLemmatizer as lemma
    #tokn = len(article.blob.words)
    tokn=len(article.tok)
    posn = len(article.pos)
    if tokn == posn:
        #words = article.blob.words
        words=article.tok
    else:
        words = [token[0] for token in article.blob.pos_tags if len(token[0])>3]
        tokn  = len(words)
    
    lems = []
    for j in range(0, tokn):

        if article.pos[j] == '':
            verb = words[j].lemmatize('v')
            noun = words[j].lemmatize('n')

            if len(verb) == len(noun):
                lems.append(words[j])
            elif len(verb) < len(noun):
                lems.append(verb)
            else:
                lems.append(noun)
            
        else:
            lems.append(words[j].lemmatize(article.pos[j]))

    lems        = [token.lower() for token in lems]
    article.lem = lems
    return article

# Counter and stopwords remover
def cntr(lems, stpw):
    from collections import Counter

    cnt = Counter(lems)
    cnt = [(cnt[entry], entry) for entry in cnt]
    cnt.sort()
    cnt.reverse()

    wordfreq = []
    for i in range(0, len(cnt)):
        if cnt[i][1] not in stpw:
            wordfreq.append(cnt[i])

    return wordfreq


