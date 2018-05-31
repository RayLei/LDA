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


# -----------------------------------
# Extracting features from database
# -----------------------------------
#def article_extractor(sqlite_file,start_date, end_date): 
#    conn=sqlite3.connect(sqlite_file)
#    c=conn.cursor()
#    articles_2016=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
#    articles_tuple=articles_2016.fetchall()
#    conn.close()
#    articles=[item[0] for item in articles_tuple]
#    return articles


directory='/Users/leihao/Downloads/'
sqlite_file=directory+'nasdaq.db'
start_date, end_date='2016-01-01', '2016-01-02'
conn=sqlite3.connect(sqlite_file)
c=conn.cursor()
articles_2016=c.execute("SELECT article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))

h_vectorizer=HashingVectorizer(tokenizer=LemmaTokenizer(),stop_words='english',ngram_range=(1,2))
for articles in articles_2016:
    X=h_vectorizer.transform(articles)
conn.close()

 #LDA Modelling
for num_of_topics in (20,30):
	#num_of_topics=20
	model=lda.LDA(n_topics=num_of_topics,n_iter=1500,random_state=1)
	model.fit(X)
	topic_word=model.topic_word_
	n_top_words=20
	with open(directory+'t_topic'+str(num_of_topics)+'_keywords20_160101_160102.txt','w+') as f:	 
		for i, topic_dist in enumerate(topic_word):
			topic_words=np.array(sorted(c_vectorizer.vocabulary_.keys()))[np.argsort(topic_dist)][:-(n_top_words+1):-1]
			f.write('Topic {0} : {1}\n'.format(i, ', '.join(topic_words).encode("utf-8")))





