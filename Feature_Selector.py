from __future__ import division
from FilenameToCat import f2c
from Tokenizer import get_list_tokens_nltk
import operator
from math import log
def mutual_information(trainset,corpus,numTopWords=500):
    ##3) Mutual Information - Feature Selection - including only those words as features which have the highest
     ##mutual information for a category - selecting top x words for a category
    ##A)Create a dictionary with a word as the key and a dictionary as the value
     ## in the dictionary the category as key and number of documents in that category where it occurs as value

    WordCatNumDocDict={}
    N = len(trainset)

    ##B)Loop through the reuters dataset, to get the entire text from  each file in the training set
    ##C) Parse the string to get individual words

    for fileName in trainset:
        listWords = get_list_tokens_nltk(corpus,fileName)
        cat = f2c(corpus,fileName)
    
    ##D) Update the dictionary
        for w in set(listWords):
            WordCatNumDocDict[w]=WordCatNumDocDict.get(w,{})
            WordCatNumDocDict[w][cat]=WordCatNumDocDict[w].get(cat,0)
            WordCatNumDocDict[w][cat]+=1


    ##E) Prepare a dictionary with key category and value as list of tuples of words with word strings  and
            #mutual information as the 2 elements of the tuple which will be the only ones considered as features
        ## and a list with all these word features
    WordFeatures={}
    WordList=[]
    for w in WordCatNumDocDict.keys():
        dic = WordCatNumDocDict[w]
        N1x=0
        for cat in dic.keys():
            N1x+=dic[cat] ## number of documents in the training set where the word occurs
        for cat in dic.keys():
            WordFeatures[cat]=WordFeatures.get(cat,[])
            N11=dic[cat]
            MI=(N11/N)*log((N*N11)/(N1x*N1x))/log(2)
            if len(WordFeatures[cat])<numTopWords:
                WordFeatures[cat].append((w,MI))
                if len(WordFeatures[cat])==numTopWords:
                    WordFeatures[cat].sort(key=operator.itemgetter(1),reverse=True)                
            else:
                if WordFeatures[cat][numTopWords-1][1]<MI:
                    WordFeatures[cat][numTopWords-1]= (w,MI)
                    WordFeatures[cat].sort(key=operator.itemgetter(1),reverse=True)                


    for cat in WordFeatures.keys():
        #print cat
        #print WordFeatures[cat]
        #print "\n"
        WordFeatures[cat]=dict(WordFeatures[cat])
        for w in WordFeatures[cat]:
            WordList.append(w)
    
    return [WordFeatures,set(WordList) ]



def gini(trainset,corpus,numTopWords=500):
    #The conditional probability of a word given a category is found by a technique very similar to
    #Multinomial Naive Bayes
    CatWordDict={}
    CatWordCountDict={}
    #val = my_dict.get(key, mydefaultval)
    WordList=[]
    WordFeatures={}
    for fileName in trainset:
        listWords = get_list_tokens_nltk(corpus,fileName)
        cat = f2c(corpus,fileName)
        
        CatWordDict[cat]=CatWordDict.get(cat,{})
        CatWordCountDict[cat]=CatWordCountDict.get(cat,0)
        CatWordCountDict[cat]+=len(listWords)

        for w in listWords:
            CatWordDict[cat][w] = CatWordDict[cat].get(w,0)
            CatWordDict[cat][w]+= 1
        
    vocabLength=0            
    for dic in CatWordDict.values():
        vocabLength+=len(dic)


    for cat in CatWordDict:
        countCat = CatWordCountDict[cat]
        WordFeatures[cat]=WordFeatures.get(cat,[])
        for w in CatWordDict[cat]:
            cond_prob=(CatWordDict[cat][w]+1)/(countCat+vocabLength)
            sumCP=cond_prob
            for cate in CatWordDict:
                if cate!=cat:
                    sumCP+=((CatWordDict[cate].get(w,0)+1)/(CatWordCountDict[cate]+vocabLength))

            gini_coef = cond_prob/sumCP
            if len(WordFeatures[cat])<numTopWords:
                WordFeatures[cat].append((w,gini_coef))
                if len(WordFeatures[cat])==numTopWords:
                    WordFeatures[cat].sort(key=operator.itemgetter(1),reverse=True)                
            else:
                if WordFeatures[cat][numTopWords-1][1]<gini_coef:
                    WordFeatures[cat][numTopWords-1]= (w,gini_coef)
                    WordFeatures[cat].sort(key=operator.itemgetter(1),reverse=True)  

    for cat in WordFeatures.keys():
        #print cat
        #print WordFeatures[cat]
        #print "\n"
        WordFeatures[cat]=dict(WordFeatures[cat])
        for w in WordFeatures[cat]:
            WordList.append(w)
    
    return [WordFeatures,set(WordList) ]
