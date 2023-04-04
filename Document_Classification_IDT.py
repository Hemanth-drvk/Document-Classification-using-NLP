#!/usr/bin/env python
# coding: utf-8

# ### Document Classification
# 
# We can  represent unstructured text as a vector of features each of which have an associated frequency count.  This allows us to to develop classification models using machine learning algorithms. Let’s use a subset newsgroups text to build a classification model and assess its accuracy.

# In[1]:


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np


# ### Load Data
# Download data from 20 news groups 

# In[2]:


newsgroups_train = fetch_20newsgroups(subset='train')
print(list(newsgroups_train.target_names))

newsgroups_test = fetch_20newsgroups(subset='train')


# ##Prepare the Data
# To keep it simple, let's filter only 5 of the 20 topics. 
# We will then convert the unstructured text to a structured vector of thousands of features made up of the words from the documents.  Stop words like “is”, “the”, “it” wil be removed.  Please look up   Each feature has a TFIDF value tha can be used calculate probabilities. Look up the SKLearn's TfidfVectorizer function to see ways that you may improve the data preparation - https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html 

# In[3]:


#Categories     0               1                   2               3             4
categories = ['alt.atheism', 'comp.graphics', 'rec.motorcycles', 'sci.space', 'talk.politics.guns']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                      shuffle=True, random_state=2017, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, 
                                     shuffle=True, random_state=2017, remove=('headers', 'footers', 'quotes'))

y_train = newsgroups_train.target
y_test = newsgroups_test.target

# Convert a collection of raw documents to a matrix of TF-IDF features
#vectorizer = TfidfVectorizer()    # This one is the basic text to feature vector function, try some of the options
#vectorizer = TfidfVectorizer(lowercase=False, stop_words='english')
#vectorizer = TfidfVectorizer(smooth_idf = True, max_df=0.5, stop_words='english')
#vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf = True, max_df=0.5,  ngram_range=(1, 2), stop_words='english')

#Using the below parameters
vectorizer = TfidfVectorizer(lowercase = True, sublinear_tf=True, smooth_idf = True,decode_error = 'strict', max_df= 1.0 ,min_df = 1 ,ngram_range=(1, 2), stop_words='english')

X_train = vectorizer.fit_transform(newsgroups_train.data)  # Learn vocabulary and idf, return term-document matrix.
X_test = vectorizer.transform(newsgroups_test.data)        # Transform documents to term-document matrix.

print("Train Dataset")
print("%d documents" % len(newsgroups_train.data))
print("%d categories" % len(newsgroups_train.target_names))
print("n_samples: %d, n_features: %d" % X_train.shape)

print("\nTest Dataset")
print("%d documents" % len(newsgroups_test.data))
print("%d categories" % len(newsgroups_test.target_names))
print("n_samples: %d, n_features: %d" % X_test.shape)


# ### Decision Tree Classifier Model
# Lets build a simple Decision Tree classifier and assess its performance on the train and independent test set.  This classifier can be replaced by any other SKLearn classification ML algorithm.

# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clf = DecisionTreeClassifier(min_samples_split = 2 , max_depth = None , criterion = 'gini')
clf = clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print("Categories: 0=alt.atheism, 1=comp.graphics, 2=rec.motorcycles, 3=sci.space, 4=talk.politics.guns\n")
print('Train accuracy_score: ', metrics.accuracy_score(y_train, y_train_pred)*100)
print('Test accuracy_score: ',metrics.accuracy_score(newsgroups_test.target, y_test_pred)*100)

print("Train Metrics: ")
print(metrics.classification_report(y_train, y_train_pred))
print("Test Metrics: ")
print(metrics.classification_report(newsgroups_test.target, y_test_pred))


# In[5]:


# Now let's look at one example.   Choose a test example by setting tx = value
# Try 0, 1801, 531, 1500, 99, 777
tx = 0

print("newsgroups_test example number", tx, ":")
print(newsgroups_test.data[tx])
#print(X_test.shape)

print("\nThe associated TFIDF vector:")
print(X_test[tx])

print("\nThe model classifies this example as:")
y_test_example = clf.predict(X_test[tx])
print("Category = ", y_test_example, "=", categories[int(y_test_example)])


# Rinda Digamarthi(157742d)

# In[ ]:




