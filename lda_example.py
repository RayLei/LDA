import numpy as np
import lda
import lda.datasets

X=lda.datasets.load_reuters()
vocab=lda.datasets.load_reuters_vocab()
titles=lda.datasets.load_reuters_titles()
model=lda.LDA(n_topics=30,n_iter=1500, random_state=1)
model.fit(X)
topic_word=model.topic_word_   # it's the beta in the model
n_top_words=10
for i, topic_dist in enumerate(topic_word):
	topic_words=np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
	print('Topic {0} : {1}'.format(i, ' '.join(topic_words)))
