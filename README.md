Naive-Bayes-Document-Classifier
===============================

Document Classification in python with some help from the Natural Language Toolkit, using a Multinomial Naive Bayes Classifier and experimenting with various feature selectors, till now only Mutual Information.

Note to the reader: The source code files are described in the order of most simple to most advanced as you navigate from top to bottom.

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

Here, nltk is used for one of the processing tasks (a - Getting the category to document mapping). The corpus is the movie_reviews corpus provided with nltk. 


NaiveBayes_nltk_reuters.py

Here, nltk is used for processing tasks a) & b). The corpus is the reuters corpus provided with nltk. Only the documents with one tag are considered. This restricts the task to single-category classification for each document.

Out of these documents, only the categories with more than 20 documents in the training set and more than 20 documents in the test set are considered. This reduces the skewness of the dataset. 


NaiveBayes_nltk_reuters_MI.py

Same as previous, except mutual information is used for feature selection. The number of features per category is restricted to 500. This results in slight degradation of performance and slightly more time to train the classifier but less time to run the classifer over the test set.

http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf Page-20

Note: I have experimented and found that the best results are obtained with a slightly different formula. 
Refer the code for my change. The original formula seems to be more useful if there are only 2 categories but fails for more than 2 categories. 
