#from article_2_vector_word_count import *
#from collections import defaultdict
import lda
import sqlite3
import numpy as np
#from scipy.sparse import csr_matrix, save_npz
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
import time
import TXTnlp
from mpi4py import MPI

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
    


class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         self.word_pos=pos_tag(word_tokenize(doc))
         return [self.wnl.lemmatize(w,get_wordnet_pos(p)) for w,p in self.word_pos 
                 if len(w)>=3 and not w.isdigit() and not is_float(w) ]


# -----------------------------------
# Extracting features from database
# -----------------------------------
def article_extractor(sqlite_file,start_date, end_date): 
    conn=sqlite3.connect(sqlite_file)
    c=conn.cursor()
    articles_2016=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
    articles_tuple=articles_2016.fetchall()
    conn.close()
    articles=[item[0] for item in articles_tuple]
    return articles


directory='/Users/leihao/Downloads/'
sqlite_file=directory+'nasdaq.db'
# -----------------
# Get the MPI rank
# -----------------
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

start_date, end_date='2016-01-01', '2016-01-02' if rank==0 else ('2016-01-03', '2016-01-04')


#t0=time.time()
#articles=article_extractor(sqlite_file,start_date, end_date)
#t1=time.time()
#print("Vectorizer takes {0:.2f} seconds".format(t1-t0))
conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
articles_2016=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
articles_tuple=articles_2016.fetchall()
conn.close()


proc_cnt, proc_vocab = [], [] 
for item in articles_tuple:
    item=articles_tuple[1]
    txt = TXTnlp.TextBlob(item[0])
    ### Tokens and POS tags
    txttok = TXTnlp.token(txt)
    txtpos = TXTnlp.pos(txt)
    
    ### WordNet lemmatizer
    # Create namespace
    article = TXTnlp.types.SimpleNamespace()
    
    # Assign parts of namespace
    #article.blob = txt
    article.tok  = txttok
    article.pos  = txtpos
    
    # Lemmatize
    article = TXTnlp.lem(article)
    
    atcl_cnt,atcl_vocab = TXTnlp.cntr(article.lem)
    proc_cnt.append(atcl_cnt)
    proc_vocab.update(atcl_vocab)
    
vocab=set()
vocab=comm.Allgather(proc_vocab,vocab)

#convert vocab dictionary to vocab list
vocab=list(vocab)

ind_cnt=[]
for item in proc_cnt:
    atcl_ind_cnt=[(vocab.index(word),cnt) for word,cnt in item]
    ind_cnt.append(atcl_ind_cnt)
        

    