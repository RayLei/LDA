import numpy as np
import lda
import lda.datasets
from numpy import genfromtxt
import csv

my_data = genfromtxt('Restaurant_data_matrix.csv', delimiter=',', skip_header=1)
word_matrix=np.delete(my_data,0,1).astype(int)   #delete the 1st column, which is NAN; convert from float to int type

with open('Restaurant_data_matrix.csv', newline='') as f:
    reader = csv.reader(f)
    vocab = next(reader)[1:]




#X=lda.datasets.load_reuters()
#vocab=lda.datasets.load_reuters_vocab()
#titles=lda.datasets.load_reuters_titles()
model=lda.LDA(n_topics=20,n_iter=1500, random_state=1)
model.fit(word_matrix)
topic_word=model.topic_word_   # it's the beta in the model
n_top_words=10
for i, topic_dist in enumerate(topic_word):
	topic_words=np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {0} : {1}'.format(i, ' '.join(topic_words)))

	
doc_topic = model.doc_topic_
for i in range(10):
     print("{} (top topic: {})".format(i, doc_topic[i].argmax()))