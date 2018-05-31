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

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    

### Count words and remove stopwords
stpw = stopwords.words('english')
stpw += ["near","nearly","neither","would","year","month","still"]
# Counter and stopwords remover
def non_stp_num(x):
    if x not in stpw and not x.isdigit() and not is_float(x):
        return True 
    else:
        return False
    
def cntr(lems):
    from collections import Counter
    # combine all lems together before Counter
    cnt = Counter(lems)
    # -------------------------
    # find the vocabulary set 
    # -------------------------
    vocab = dict(cnt)
    vocab = {item for item in vocab if non_stp_num(item)}
    # -----------------------------------------------------
    # Convert the Counter-object to a list of (word, count)
    # And remove the stopwords and digits
    # -----------------------------------------------------
    cnt = cnt.items()      
    cnt = [item for item in cnt if non_stp_num(item[0],stpw)] 
    return (cnt,vocab)


