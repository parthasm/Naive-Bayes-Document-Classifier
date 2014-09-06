Naive-Bayes-Document-Classifier
===============================

This repository implements Supervised Document Classification in python. Here the task is to assign a document to one class or category. Some text-processing tasks are done by the Natural Language Toolkit, while the algorithm implementations are done from scratch. The classifiers are Multinomial Naive Bayes and Bernoulli Naive Bayes while the feature selectors, till now, Mutual Information and Gini Coefficient.

Read more on these topics from these links:

http://en.wikipedia.org/wiki/Document_classification

http://www.nltk.org/

http://www.nltk.org/book/ch02.html

http://en.wikipedia.org/wiki/Naive_Bayes_classifier

http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html

http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf Page-20

Gini Coefficient is an experimental feature selector. The concept used is from the paper "Feature Selection for Text Classification Based on
Gini Coefficient of Inequality" by Sanasam Ranbir Singh, Hema A. Murthy, Timothy A. Gonsalves. The basic concept is that the high conditional probability of a word given a category normalized by the sum of conditional probabilities of the word in all categories indicate strong association of the word with the category. Thus the word is a good feature for the category.

Link to the Paper:

http://jmlr.org/proceedings/papers/v10/sanasam10a/sanasam10a.pdf

##Evaluation of the Classifiers:


For multi-class classification, the respective function reports the following measures:
a) Macro-Precision,
b) Micro-Precision,
c) Macro-Recall,
d) Micro-Recall,
e) Macro-F1-Measure
f) Micro-F1-Measure and 
g) Fraction of Mis-Classified Documents

For binary classification, the respective function reports the following measures:
a) Precision,
b) Recall,
c) F1-Measure and 
d) Fraction of Mis-Classified Documents

Sources:

http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html

http://en.wikipedia.org/wiki/F1_score

http://en.wikipedia.org/wiki/Precision_and_recall

If precision or recall for a category in multi-class classification is not defined, the category is excluded from the calculation of the macro and micro variables except for Fraction of Mis-Classified Documents.
