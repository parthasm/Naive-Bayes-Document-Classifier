Naive-Bayes-Document-Classifier
===============================

Document Classification in python with some help from the Natural Language Toolkit, using Multinomial and Bernoulli Naive Bayes Classifiers and experimenting with various feature selectors, namely Mutual Information and Gini Coefficient.

Note to the reader: The source code files are described roughly in the order of most simple to most advanced as you navigate from top to bottom.

Helper Function Files:

##FilenameToCat.py

It gets the category of its argument filename string. While this is easily done by a nltk function call, it is important to explicitly not invoke nltk, in the Major Files, to increase portability. And this makes the code in the Major Files reusable when non-nltk corpora are used.

##Tokenizer.py

It extracts tokens from the file specified by its argument filename.  It uses nltk to get the raw text and then uses regular expression from the 're' package in python to tokenize the text. It then removes stopwords, non-alpha-numeric words and very small words. Presently, there is one tokenizer function for every corpus. As of now, stemming is not used. 

##Evaluation.py

It evaluates the performance of the classifier. 

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

If precision or recall are not defined, the evaluation_fraction_misclass() function reports only the ratio of number of misclassifications and number of documents in the testset , that is, fraction of misclassifications.


##Feature_Selector.py

It selects the suitable word features for Naive Bayes classification. This is an important step for Bernoulli Naive Bayes, whose accuracy is often low without feature selection.

The first used is Mutual Information 

http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf Page-20

The 2nd one is Gini Coefficient. The concept is used is from this paper

http://jmlr.org/proceedings/papers/v10/sanasam10a/sanasam10a.pdf

The concept used is from the paper "Feature Selection for Text Classification Based on
Gini Coefficient of Inequality" by Sanasam Ranbir Singh, Hema A. Murthy, Timothy A. Gonsalves. The basic concept is that the high conditional probability of a word given a category normalized by the sum of conditional probabilities of the word in all categories indicate strong association of the word with the category. Thus the word is a good feature for the category.

Major Files:

##Multinomial_NB.py
 
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

The final result is reported using the evaluation_multi_class() function in Evaluation.py.

The time taken to train the classifier and the time taken to run the classifier on the test set are also reported.


##Multinomial_NB_nltk_mr.py

Here, nltk is used for one of the processing tasks (a - Getting the category to document mapping). The corpus is the movie_reviews corpus provided with nltk. 


##Bernoulli_NB_nltk_reuters.py

Here, the simpler and weaker classifier, the Bernoulli Naive Bayes is used on the reuters nltk corpus. Its other characteritics are same as the code file below.

Bernoulli Naive Bayes: 

http://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes

http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html

Also, in Bernoulli Naive Bayes, all the words in the vocabulary contribute to the score for each category for a document, unlike Multinomial Naive Bayes where only the words in the document contribute to the score for each category. Therefore, Bernoulli Naive Bayes without feature Selection takes a lot of time, to traverse the entire vocabulary of the corpus for every document in the testset. 

##Multinomial_NB_nltk_reuters.py

Here, nltk is used for processing tasks a) & b). The corpus is the reuters corpus provided with nltk. Only the documents with one tag are considered. This restricts the task to single-category classification for each document.

Out of these documents, only the categories with more than 20 documents in the training set and more than 20 documents in the test set are considered. This reduces the skewness of the dataset. 


##Multinomial_NB_nltk_reuters_MI.py

Same as previous, except mutual information is used for feature selection. The number of features per category is restricted to 500. This results in slight degradation of performance and slightly more time to train the classifier but less time to run the classifer over the test set.

http://nlp.stanford.edu/IR-book/pdf/13bayes.pdf Page-20

Note: 

a) There is no improvement in performance - either in accuracy or time taken since Multinomial Naive Bayes without feature selection is already a strong and simple classifier respectively.

b) I have experimented and found that the best results are obtained with a slightly different formula. 
Refer the code for my change. The original formula seems to be more useful if there are only 2 categories but fails for more than 2 categories. 


##Multinomial_NB_nltk_reuters_gini.py

Same as previous, except instead of mutual information , gini coefficient is used as feature selector. The concept is described before under Feature_Selector.py. The performance is slightly worse than mutual information(9.98% versus 7.89%) for 500 features.

##Bernoulli_NB_nltk_reuters_MI.py

Same as Bernoulli_NB_nltk_reuters.py, except mutual information is used for feature selection. The number of features per category is restricted to 500.This results in significant improvment in performance(from 22% misclassification to 11% misclassification) and approximately 50% more time to train the classifier but 97% less time to run the classifer over the test set. This is because Bernoulli Naive Bayes is a weak classifier - hence the scope for improvement with feature selection. Since the time-consuming step is traversing the entire vocabulary for every test document, on reducing the vocabulary by feature selection, the time taken to run the classifier is drastically reduced.


##Bernoulli_NB_nltk_reuters_gini.py

Same as previous, except instead of mutual information , gini coefficient is used as feature selector. The concept is described before under Feature_Selector.py. The performance is slightly worse than mutual information(11.09% versus 9.76%) for 500 features. Again, time required is drastically used, compared to bare Bernoulli, without feature selection.
