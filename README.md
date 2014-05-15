Naive-Bayes-Document-Classifier
===============================

Document Classification in python with some help from the Natural Language Toolkit, using a Multinomial Naive Bayes Classifier and experimenting with various feature selectors, till now only Mutual Information.

 NaiveBayes.py
 
 The ' NaiveBayes.py' implements a Multinomial Naive Bayes Classifier without using feature selection and without using 
 any kind of processing help from the Natural Language Toolkit (nltk). 
 
 
 Multinomial Naive Bayes Clasification:
 
 http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes
 
 http://www.stanford.edu/class/cs124/lec/naivebayes.pdf
 
 
 Feature Selection:
 
 http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html
 
 
 Processing Text with Natural Language Toolkit:
 
 http://www.nltk.org/book/ch02.html
 
The processing tasks include:

a) Getting the category to document mapping.

b) Splitting the dataset into training set and test set.

c) Tokenization: Splitting the raw text from each document into tokens, for documents in both training dataset and test dataset.

The final result is reported as the following parameters:

a) Macro-Precision,
b) Micro-Precision,
c) Macro-Recall,
d) Micro-Recall,
e) Macro-F1-Measure
f) Micro-F1-Measure and 
g) Fraction of Mis-Classified Documents

The time taken to train the classifier and the time taken to run the classifier on the test set are also reported.

Sources:

http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html

http://en.wikipedia.org/wiki/F1_score

http://en.wikipedia.org/wiki/Precision_and_recall

NaiveBayes_nltk_movie_reviews.py

Here, nltk is used for the processing tasks (a - Getting the category to document mapping). The corpus is the movie_reviews corpus provided with nltk. 
