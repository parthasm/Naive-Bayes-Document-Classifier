from __future__ import division
from Filename_To_Cat import f2c
from Tokenizer import get_list_tokens_nltk
import Evaluation
from math import log
import Preprocessor
import time

i = input('Enter within quotes, m for movie reviews corpus,'
          'r for reuters corpus( default is reuters) : ')
corpus=''
binary_classification=False
if i=='m' or i=='M':
    corpus='mr'
    binary_classification=True
else:
    corpus='reuters'
#Forming Prepare the cat_num_docs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set

start_time = time.time()

#Here, apart from the naive bayes classifier, everything is done by nltk

#2) refer the comments for the function get_testset_trainset()

li = Preprocessor.get_testset_trainset(corpus)
testset = li[1]
trainset = li[0]
li = Preprocessor.startup()
cat_num_docs = li[1]

#3)Create a dictionary with a word as the key and a dictionary as the value
     ## in the dictionary the category as key and number of documents in that category where it occurs as value

word_cat_num_doc_dict={}

#4)Loop through the reuters dataset, to get the entire text from  each file in the training set
    ## Parse the string to get individual words - done by get_list_tokens_nltk()
for file_name in trainset:
    list_words = get_list_tokens_nltk(corpus,file_name)
    cat = f2c(corpus,file_name)

    for w in set(list_words):
       word_cat_num_doc_dict[w]=word_cat_num_doc_dict.get(w,{})
       word_cat_num_doc_dict[w][cat]=word_cat_num_doc_dict[w].get(cat,0)
       word_cat_num_doc_dict[w][cat]+=1

for w in word_cat_num_doc_dict:
    for cat in cat_num_docs:
        nct = word_cat_num_doc_dict[w].get(cat,0)
        ratio = (nct+1)/(cat_num_docs[cat]+2)
        word_cat_num_doc_dict[w][cat]=ratio

####Congratulations! the Classifier is trained, now it is time to run the Bernoulli Naive Bayes Classifier on the test dataset
print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()


li_results=[]
#5) Like in the training set,Loop through the test set, to get the individual words
for file_name in testset:
    minimum_neg_log_prob=1000000000
    min_category=''
    set_list_words = set(get_list_tokens_nltk(corpus,file_name))

##6) Get the probability for each category,
    #using the cat_num_docs dictionary to wade through the categories
    for cat in  cat_num_docs:
        neg_log_prob=-log(cat_num_docs[cat]/len(trainset))
        for w in word_cat_num_doc_dict:
            if w in set_list_words:
                neg_log_prob-=log(word_cat_num_doc_dict[w][cat])
            else:
                neg_log_prob-=log(1-word_cat_num_doc_dict[w][cat])
        if minimum_neg_log_prob>neg_log_prob:
            min_category=cat
            minimum_neg_log_prob=neg_log_prob

    li_results.append((file_name,min_category,f2c(corpus,file_name)))

if binary_classification:
    Evaluation.evaluation_binary(li_results)
else:
    Evaluation.evaluation_multi_class(li_results,cat_num_docs.keys())



print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"


