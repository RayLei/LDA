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
import pickle
from scipy import sparse

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
                 if len(w)>3 and not w.isdigit() and not is_float(w) ]


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
start_date, end_date='2016-05-01', '2016-05-31'

conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
art_all_info=c.execute("SELECT title, date, author, article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
art_all=art_all_info.fetchall()
conn.close()

#t0=time.time()
#articles=article_extractor(sqlite_file,start_date, end_date)
brexit_ind, tesla_ind=[],[]
for art in art_all:
    if 'brexit' in art[3] or 'Brexit' in art[3]:
        brexit_ind.append(art_all.index(art))
    
    if 'Tesla' in art or 'tesla' in art:
        tesla_ind.append(articles.index(art))



conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
art_all_info=c.execute("SELECT title, date, author, article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
art_all=art_all_info.fetchall()
conn.close()



#t1=time.time()
#print("Vectorizer takes {0:.2f} seconds".format(t1-t0))
#conn=sqlite3.connect(sqlite_file)
#c=conn.cursor()
#articles_2016=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
#articles_tuple=articles_2016.fetchall()
#conn.close()
#articles=[item[0] for item in articles_tuple]

#t3=time.time()
#import cProfile, pstats, io
#pr=cProfile.Profile()
#pr.enable()
c_vectorizer=CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english')

#h_vectorizer=HashingVectorizer(tokenizer=LemmaTokenizer(),stop_words='english',ngram_range=(1,2))
X_c=c_vectorizer.fit_transform(articles)
#pr.disable()
#s=io.StringIO()
#sortby='time'
#ps=pstats.Stats(pr,stream=s).sort_stats(sortby)
#ps.print_stats()
#print(s.getvalue())
#X_h=h_vectorizer.transform(articles)
#t4=time.time()
#print("Vectorizer takes {} seconds".format(t4-t3))



test_art=['apple apple banana this is','apple security security winter winter winter']
test_X=c_vectorizer.fit_transform(test_art)

vocab=c_vectorizer.get_feature_names()
with open(directory+'vocabulary.txt','wb+') as vocab_f:
    pickle.dump(vocab,vocab_f)
    
sparse.save_npz(directory+'gibbs_mat',test_X)

ldac_mat=lda.utils.dtm2ldac(test_X)
with open(directory+'ldac_mat.txt','w+') as ldac_f:
    for item in ldac_mat:
        ldac_f.write(item+'\n')


with open(directory+'vocabulary.txt','rb') as vocab_f:
    vocab_test=pickle.load(vocab_f)
    
sparse_mat=sparse.load_npz(directory+'gibbs_mat.txt.npz')

    




 #LDA Modelling
#for num_of_topics in (20,30):
num_of_topics=20
model=lda.LDA(n_topics=num_of_topics,n_iter=1500,random_state=1)
model.fit(X_c)
topic_word=model.topic_word_
n_top_words=20
with open(directory+'t_tpc'+str(num_of_topics)+'_kwd20_gfn_160101_160102.txt','w+') as f:	 
	for i, topic_dist in enumerate(topic_word):
		topic_words=np.array(c_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
		f.write('Topic {0} : {1}\n'.format(i, ', '.join(topic_words).encode("utf-8")))







#tot_num_per_row=a.getnnz(axis=1)
#all_pair=[]
#for row in range(3):
#    col_ind=a[row,].nonzero()[1]
#    val=np.asarray(a[row,col_ind].todense())[0]    
#    pair=list(zip(col_ind,val))
#    all_pair.append(pair)
#pre_result=list(zip(tot_num_per_row,all_pair))
#
#for num,item in pre_result:
#    print(num,end=' ')
#    for col, cnt in item:
#        print('{}:{}'.format(col,cnt),end=' ')
#    print()
