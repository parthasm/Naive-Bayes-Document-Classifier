from __future__ import division
from Filename_To_Cat import f2c
import Evaluation
from Tokenizer import get_list_tokens_nltk
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
    


##4)Create a) a dictionary with a category as the key and dictionary of words-occurences as values
          #b) a dictionary with a category as the key and the number of words in it as the value
cat_word_dict={}
cat_word_count_dict={}
#val = my_dict.get(key, mydefaultval)

##5)Loop through the training set, to get the entire text from  each file
##6) Parse the string to get individual words
for file_name in trainset:
    list_words = get_list_tokens_nltk(corpus,file_name)
    

##7) Check if category exists in dictionary, if not, create an empty dictionary,
    #and put word count as zero
    #and then insert words into the category's dictionary in both cases and update the word count
    cat = f2c(corpus,file_name)
    cat_word_dict[cat] = cat_word_dict.get(cat,{})
    cat_word_count_dict[cat] = cat_word_count_dict.get(cat,0)
    

    cat_word_count_dict[cat]+=len(list_words)
    
    for w in list_words:
        cat_word_dict[cat][w] = cat_word_dict[cat].get(w, 0)
        cat_word_dict[cat][w]+=1
        
        

##8) Get the vocabulary length
vocab_length=0            
for dic in cat_word_dict.values():
     vocab_length+=len(dic)


print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()



####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset
length_train = len(trainset)
li_results=[]
#9) Like in the training set,Loop through the test set, to get the entire text from  each file
##10) Similar step, parse the string to get individual words
for file_name in testset:
    minimum_neg_log_prob=1000000000
    min_category=''
    list_words = get_list_tokens_nltk(corpus,file_name)


    
##11) Get the probability for each category,
    #can use any of the created dictionaries to wade through the categories
    for cat in  cat_word_count_dict:
        #print cat , cat_num_docs[cat]/len(trainset)
        neg_log_prob=-log(cat_num_docs[cat]/length_train)
        word_dict = cat_word_dict[cat]
        count_cat = cat_word_count_dict[cat]
        for w in list_words:
            count_word_train=word_dict.get(w,0)
            ratio = (count_word_train+1)/(count_cat+vocab_length)
            neg_log_prob-=log(ratio)           
                         
        if minimum_neg_log_prob>neg_log_prob:
            min_category=cat
            minimum_neg_log_prob=neg_log_prob

    li_results.append((file_name,min_category,f2c(corpus,file_name)))

###--------------------DEBUG STATEMENTS----------------------
#for t in li_results:
 #   if t[1]!=t[2]:
  #      print t
###--------------------DEBUG STATEMENTS----------------------
    
###--------------------DEBUG STATEMENTS----------------------

#12) Evaluating the classifier

if binary_classification:
    Evaluation.evaluation_binary(li_results)
else:
    Evaluation.evaluation_multi_class(li_results,cat_num_docs.keys())


print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"
