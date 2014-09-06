Naive-Bayes-Document-Classifier
===============================

This repository implements Supervised Document Classification in python. Here the task is to assign a document to one class or category. Some text-processing tasks are done by the Natural Language Toolkit, while the algorithm implementations are done from scratch. The classifiers are Multinomial Naive Bayes and Bernoulli Naive Bayes while the feature selectors, till now, Mutual Information and Gini Coefficient.There are six versions of the classifier:

a) Bernoulli Navie Bayes, without Feature Selection

b) Bernoulli Navie Bayes, with Mutual Information as Feature Selector

c) Bernoulli Navie Bayes, with Gini Coefficient as Feature Selector

d) Multinomial Navie Bayes, without Feature Selection

e) Multinomial Navie Bayes, with Mutual Information as Feature Selector

f) Multinomial Navie Bayes, with Gini Coefficient as Feature Selector


Gini Coefficient is an experimental feature selector. The concept used is from the paper "Feature Selection for Text Classification Based on
Gini Coefficient of Inequality" by Sanasam Ranbir Singh, Hema A. Murthy, Timothy A. Gonsalves. The basic concept is that the high conditional probability of a word given a category normalized by the sum of conditional probabilities of the word in all categories indicate strong association of the word with the category. Thus the word is a good feature for the category.



##Evaluation of the Classifiers:


For multi-class classification, the respective function reports the following measures:
a) Macro-Precision

b) Micro-Precision

c) Macro-Recall

d) Micro-Recall

e) Macro-F1-Measure

f) Micro-F1-Measure 

g) Fraction of Mis-Classified Documents

For binary classification, the respective function reports the following measures:

a) Precision 

b) Recall 

c) F1-Measure

d) Fraction of Mis-Classified Documents


If precision or recall for a category in multi-class classification is not defined, the category is excluded from the calculation of the macro and micro variables except for Fraction of Mis-Classified Documents.


##Observations

Multinomial Naive Bayes without feature selection is already a strong and simple classifier. Therefore there is no improvement in accuracy and time taken respectively on using feature selection. 


In Bernoulli Naive Bayes, all the words in the vocabulary contribute to the score for each category for a document, unlike Multinomial Naive Bayes where only the words in the document contribute to the score for each category. Therefore, Bernoulli Naive Bayes without Feature Selection takes a lot of time, to traverse the entire vocabulary of the corpus for every document in the testset. It is also a weak classifier, without feature selection. On using feature selection, time taken to run the classifier over the test set is reduced by as much as 97%.


The movie reviews corpus is not suitable for Naive Bayes Classification. This is presumably because Naive Bayes Classifier considers words independently of each other and focusses on their counts. This strategy would work great if the categories have their specific jargon. Hence the great performance on the reuters corpus. The movie reviews corpus, on the other hand, would have similar or even same words in both the categories. Hence Naive Bayes Classification is not suitable in this case.


##Sources:

Document Classification: http://en.wikipedia.org/wiki/Document_classification

Natural Language ToolKit: http://www.nltk.org/

Text Processing with NLTK: http://www.nltk.org/book/ch02.html

Naive Bayes Classifier: http://en.wikipedia.org/wiki/Naive_Bayes_classifier

Feature Selection: http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html

Mutual Information: http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf Page-20

Gini Coefficient Paper:  http://jmlr.org/proceedings/papers/v10/sanasam10a/sanasam10a.pdf

Micro and Macro Averages: http://rushdishams.blogspot.com/2011/08/micro-and-macro-average-of-precision.html

F-1 Score: http://en.wikipedia.org/wiki/F1_score

Precision and Recall: http://en.wikipedia.org/wiki/Precision_and_recall
