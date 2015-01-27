from __future__ import division
from Filename_To_Cat import f2c
from Tokenizer import get_list_tokens_nltk
import Evaluation
import Feature_Selector
import Preprocessor
from math import log
import time


#Forming Prepare the cat_num_docs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set


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

###--------------------DEBUG STATEMENTS----------------------
#for f in trainset:
 #   print f , FilenameCategoriesDict[f] 

#print "Freedom\n"

#for f in testset:
 #   print f    
###--------------------DEBUG STATEMENTS----------------------

i = input('Enter within quotes, g for gini coefficient,'
          'm for mutual information( default is mutual information)  as feature selector: ')

if i=='g' or i=='G':
    li = Feature_Selector.gini(trainset,corpus)    
else:
    li = Feature_Selector.mutual_information(trainset,corpus)    
word_features = li[0]
word_list = li[1]
    
##4)Create a) a dictionary with a category as the key and dictionary of words-occurences as values
          #b) a dictionary with a category as the key and the number of words in it as the value
cat_word_dict={}
cat_word_count_dict={}
#val = my_dict.get(key, mydefaultval)

##5)Loop through the training set, to get the entire text from  each file
##6) Parse the string to get individual words
for file_name in trainset:
    list_words = get_list_tokens_nltk(corpus,file_name)
    cat = f2c(corpus,file_name)
    list_words = [w for w in list_words if word_features[cat].get(w,-100000)!=-100000]
    #!!!!!!!!------Possible Improvement: Stemming--------------#


##7) Check if category exists in dictionary, if not, create an empty dictionary,
    #and put word count as zero
    #and then insert words into the category's dictionary in both cases and update the word count
    cat_word_dict[cat] = cat_word_dict.get(cat,{})
    cat_word_count_dict[cat] = cat_word_count_dict.get(cat,0)
    
 ##Update the dictionary - 2 possible ways
    ##A) loop over the set of words and update dictionary with log value
        ##Complexity- n(set)*n(count operation) = O(n^2)
    ##B) loop over list and update count for each occurence
        #at the end, loop over set and replace count with log value
        ##Complexity- n(list)+n(set) = O(n)
        ##B is better and takes one second lesser time to prepare the index

    cat_word_count_dict[cat]+=len(list_words)

    ##A)
    #for w in set(list_words):
     #   cat_word_dict[cat][w] = cat_word_dict[cat].get(w,0)
      #  cat_word_dict[cat][w]+= list_words.count(w)           
    ##B)
    for w in list_words:
        cat_word_dict[cat][w] = cat_word_dict[cat].get(w,0)
        cat_word_dict[cat][w]+= 1
        

##8) Get the vocabulary length
vocab_length=0            
for cat in cat_word_dict.keys():
    length = len(cat_word_dict[cat])
    ###--------------------DEBUG STATEMENTS----------------------
    #print cat, length
    ###--------------------DEBUG STATEMENTS----------------------
    vocab_length+=length


####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset
print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()



li_results=[]
#9) Like in the training set,Loop through the test set, to get the entire text from  each file
##10) Similar step, parse the string to get individual words
for file_name in testset:
    minimum_neg_log_prob=1000000000
    min_category=''
    list_words = get_list_tokens_nltk(corpus,file_name)
    list_words = [w for w in list_words if w in word_list]
    
    ###--------------------DEBUG STATEMENTS----------------------
    #if file_name=='test/15024':
     #   print list_words
    ###--------------------DEBUG STATEMENTS----------------------
##11) Get the probability for each category,
    #can use any of the created dictionaries to wade through the categories
    for cat in  cat_word_count_dict:
        ###--------------------DEBUG STATEMENTS----------------------
        #print cat , cat_num_docs[cat]/len(trainset)
        ###--------------------DEBUG STATEMENTS----------------------
        neg_log_prob=-log(cat_num_docs[cat]/len(trainset))
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
 #   print t    
###--------------------DEBUG STATEMENTS----------------------
    
if binary_classification:
    Evaluation.evaluation_binary(li_results)
else:
    Evaluation.evaluation_multi_class(li_results,cat_num_docs.keys())


print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"
