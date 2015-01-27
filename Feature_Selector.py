from __future__ import division
from Filename_To_Cat import f2c
from Tokenizer import get_list_tokens_nltk
import operator
from math import log
def mutual_information(trainset,corpus,num_top_words=500):
    ##3) Mutual Information - Feature Selection - including only those words as features which have the highest
     ##mutual information for a category - selecting top x words for a category
    ##A)Create a dictionary with a word as the key and a dictionary as the value
     ## in the dictionary the category as key and number of documents in that category where it occurs as value

    word_cat_num_doc_dict={}
    n = len(trainset)

    ##B)Loop through the reuters dataset, to get the entire text from  each file in the training set
    ##C) Parse the string to get individual words

    for file_name in trainset:
        list_words = get_list_tokens_nltk(corpus,file_name)
        cat = f2c(corpus,file_name)
    
    ##D) Update the dictionary
        for w in set(list_words):
            word_cat_num_doc_dict[w]=word_cat_num_doc_dict.get(w,{})
            word_cat_num_doc_dict[w][cat]=word_cat_num_doc_dict[w].get(cat,0)
            word_cat_num_doc_dict[w][cat]+=1


    ##E) Prepare a dictionary with key category and value as list of tuples of words with word strings  and
            #mutual information as the 2 elements of the tuple which will be the only ones considered as features
        ## and a list with all these word features
    word_features={}
    word_list=[]
    for w in word_cat_num_doc_dict.keys():
        dic = word_cat_num_doc_dict[w]
        n1x=0
        for cat in dic.keys():
            n1x+=dic[cat] ## number of documents in the training set where the word occurs
        for cat in dic.keys():
            word_features[cat]=word_features.get(cat,[])
            n11=dic[cat]
            mi=(n11/n)*log((n*n11)/(n1x*n1x))/log(2)
            if len(word_features[cat])<num_top_words:
                word_features[cat].append((w,mi))
                if len(word_features[cat])==num_top_words:
                    word_features[cat].sort(key=operator.itemgetter(1),reverse=True)                
            else:
                if word_features[cat][num_top_words-1][1]<mi:
                    word_features[cat][num_top_words-1]= (w,mi)
                    word_features[cat].sort(key=operator.itemgetter(1),reverse=True)                


    for cat in word_features.keys():
        #print cat
        #print word_features[cat]
        #print "\n"
        word_features[cat]=dict(word_features[cat])
        for w in word_features[cat]:
            word_list.append(w)
    
    return [word_features,set(word_list) ]



def gini(trainset,corpus,num_top_words=500):
    #The conditional probability of a word given a category is found by a technique very similar to
    #Multinomial Naive Bayes
    cat_word_dict={}
    cat_word_count_dict={}
    #val = my_dict.get(key, mydefaultval)
    word_list=[]
    word_features={}
    for file_name in trainset:
        list_words = get_list_tokens_nltk(corpus,file_name)
        cat = f2c(corpus,file_name)
        
        cat_word_dict[cat]=cat_word_dict.get(cat,{})
        cat_word_count_dict[cat]=cat_word_count_dict.get(cat,0)
        cat_word_count_dict[cat]+=len(list_words)

        for w in list_words:
            cat_word_dict[cat][w] = cat_word_dict[cat].get(w,0)
            cat_word_dict[cat][w]+= 1
        
    vocab_length=0            
    for dic in cat_word_dict.values():
        vocab_length+=len(dic)


    for cat in cat_word_dict:
        count_cat = cat_word_count_dict[cat]
        word_features[cat]=word_features.get(cat,[])
        for w in cat_word_dict[cat]:
            cond_prob=(cat_word_dict[cat][w]+1)/(count_cat+vocab_length)
            sum_cp=cond_prob
            for cate in cat_word_dict:
                if cate!=cat:
                    sum_cp+=((cat_word_dict[cate].get(w,0)+1)/(cat_word_count_dict[cate]+vocab_length))

            gini_coef = cond_prob/sum_cp
            if len(word_features[cat])<num_top_words:
                word_features[cat].append((w,gini_coef))
                if len(word_features[cat])==num_top_words:
                    word_features[cat].sort(key=operator.itemgetter(1),reverse=True)                
            else:
                if word_features[cat][num_top_words-1][1]<gini_coef:
                    word_features[cat][num_top_words-1]= (w,gini_coef)
                    word_features[cat].sort(key=operator.itemgetter(1),reverse=True)  

    for cat in word_features.keys():
        #print cat
        #print word_features[cat]
        #print "\n"
        word_features[cat]=dict(word_features[cat])
        for w in word_features[cat]:
            word_list.append(w)
    
    return [word_features,set(word_list) ]
