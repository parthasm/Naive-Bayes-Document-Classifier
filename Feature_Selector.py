from __future__ import division
from Tokenizer import get_list_tokens_nltk_reuters
import operator
from math import log
def mutual_information(FilenameCatsDict_Train,CatNumDocs,numTopWords=500):
    ##3) Information Theoritic Mutual Information - Feature Selection - including only those words as features which have the highest
     ##mutual information for a category - selecting top x words for a category
    ##A)Create a dictionary with a word as the key and a dictionary as the value
     ## in the dictionary the category as key and number of documents in that category where it occurs as value

    WordCatNumDocDict={}
    N = len(FilenameCatsDict_Train)

    ##B)Loop through the reuters dataset, to get the entire text from  each file in the training set
    ##C) Parse the string to get individual words

    for fileName in FilenameCatsDict_Train:
        listWords = get_list_tokens_nltk_reuters(fileName)
        cat = FilenameCatsDict_Train[fileName]
    
    ##D) Update the dictionary
        for w in set(listWords):
            WordCatNumDocDict[w]=WordCatNumDocDict.get(w,{})
            WordCatNumDocDict[w][cat]=WordCatNumDocDict[w].get(cat,0)
            WordCatNumDocDict[w][cat]+=1


    ##E) Prepare a dictionary with key category and value as list of tuples of words with word strings  and
            #mutual information as the 2 elements of the tuple which will be the only ones considered as features
        ## and a list with all these word features
    WordFeatures={}
    WordList={}
    for w in WordCatNumDocDict.keys():
        dic = WordCatNumDocDict[w]
        N1x=0
        for cat in dic.keys():
            N1x+=dic[cat] ## number of documents in the training set where the word occurs
        for cat in dic.keys():
            WordFeatures[cat]=WordFeatures.get(cat,[])
            Nx1=CatNumDocs[cat] ## number of documents in the training set of the particular category
            N11=dic[cat]
            N01=Nx1-N11 ## num documents of the category where the word does not occur in trainset
            N10=N1x-N11 ## num documents of other categories where the word occurs
            N00 = N - (N01+N10+N11)
            MI=(N11/N)*log((N*N11)/(N1x*N1x))/log(2)
            if N01!=0:
                MI-=(N01/N)*log((N*N01)/((N-N1x)*N1x))/log(2)
            if N10!=0:
                MI-=(N10/N)*log((N*N10)/((N-N1x)*N1x))/log(2)
            #if N00!=0:
         #   MI+=(N00/N)*log((N*N00)/((N-N1x)*(N-N1x)))/log(2)

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
            WordList[w]=True
    li =[]
    li.append(WordFeatures)
    li.append(WordList)
    return li
