# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:16:41 2018

@author: DELL
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import ward, dendrogram, linkage
import matplotlib.pyplot as plt
corpus={'I like Shenyang','Shenyang is a good place','I still like shenyang',
'Beijing is a good place','Which is the best restaurant in Shenyang',
'Which is the best restaurant in Beijing','I like Beijing'}
print(corpus)
vectorizer=CountVectorizer()
x=vectorizer.fit_transform(corpus)
word=vectorizer.get_feature_names()
print(word)
print(x.toarray())
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(x)
print(tfidf.toarray())
dist=1-cosine_similarity(tfidf)
lmatrix=linkage(dist,method='complete',metric='cosine')
print(lmatrix)
dendrogram(lmatrix)
plt.show()
