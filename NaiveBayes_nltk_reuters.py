from __future__ import division
from nltk.corpus import reuters
from Tokenizer_nltk_reuters import get_list_tokens
from Evaluation import evaluation_multi_class
#from os import listdir
from os.path import isfile, join
from math import log
import time

def get_testset_trainset():
    cleanFiles = [f for f in reuters.fileids() if len(reuters.categories(fileids=f))==1]    
    testset = [f for f in cleanFiles if f[:5]=='test/']
    trainset = [f for f in cleanFiles if f[:9]=='training/']
    for cat in reuters.categories():
        li=[f for f in reuters.fileids(categories=cat) if f in trainset]
        liTe = [f for f in reuters.fileids(categories=cat) if f in testset]
        if len(li)>20 and len(liTe)>20:
            CatNumDocs[cat]=len(li)
            li.extend(liTe)
            categoriesFilenameDict[cat]=li
    return [[ f for f in trainset if reuters.categories(fileids=f)[0] in categoriesFilenameDict],
            [ f for f in testset if reuters.categories(fileids=f)[0] in categoriesFilenameDict]]



    





    

start_time = time.time()

#Here, apart from the naive bayes classifier, everything is done by nltk

##2)Forming Prepare the CatNumDocs dictionary, where the number of documents in the training set for each
             ##category are stored
    ##also forming the training set and test set
    ##No need of reverse dictionary as getting the category from the fileid is straighforward
categoriesFilenameDict={}
CatNumDocs={}
li = get_testset_trainset()
testset = li[1]
trainset = li[0]




    


###--------------------DEBUG STATEMENTS----------------------
#for f in trainset:
 #   print f , FilenameCategoriesDict[f] 

#print "Freedom\n"

#for f in testset:
 #   print f    
###--------------------DEBUG STATEMENTS----------------------
    


##4)Create a) a dictionary with a category as the key and dictionary of words-occurences as values
          #b) a dictionary with a category as the key and the number of words in it as the value
CatWordDict={}
CatWordCountDict={}
#val = my_dict.get(key, mydefaultval)

##5&6)Loop through the training set, to get the individual words

for fileName in trainset:
    listWords = get_list_tokens(fileName)


##7) Check if category exists in dictionary, if not, create an empty dictionary,
    #and put word count as zero
    #and then insert words into the category's dictionary in both cases and update the word count
    cat = reuters.categories(fileids=fileName)[0]
    CatWordDict[cat] = CatWordDict.get(cat,{})
    CatWordCountDict[cat]= CatWordCountDict.get(cat,0)
    
 ##Update the dictionary - 2 possible ways
    ##A) loop over the set of words and update dictionary with log value
        ##Complexity- n(set)*n(count operation) = O(n^2)
    ##B) loop over list and update count for each occurence
        #at the end, loop over set and replace count with log value
        ##Complexity- n(list)+n(set) = O(n)
        ##B is better and takes one second lesser time to prepare the index

    CatWordCountDict[cat]+=len(listWords)

    ##A)
    #for w in set(listWords):
     #   CatWordDict[cat][w] = CatWordDict[cat].get(w,0)
      #  CatWordDict[cat][w]+= listWords.count(w)           
    ##B)
    for w in listWords:
        CatWordDict[cat][w] = CatWordDict[cat].get(w,0)
        CatWordDict[cat][w]+= 1
        

##8) Get the vocabulary length
vocabLength=0            
for dic in CatWordDict.values():
     vocabLength+=len(dic)


####Congratulations! the Classifier is trained, now it is time to run the Multinomial Naive Bayes Classifier on the test dataset
print "The Classifier is trained and it took"
print time.time() - start_time, "seconds"
start_time = time.time()



liResults=[]
#9 & 10) Like in the training set,Loop through the test set, to get the individual words
for fileName in testset:
    minimumNegLogProb=1000000000
    minCategory=''
    listWords = get_list_tokens(fileName)



    
    

    
##11) Get the probability for each category,
    #can use any of the created dictionaries to wade through the categories
    for cat in  CatWordCountDict:
        ###--------------------DEBUG STATEMENTS----------------------
        #print cat , CatNumDocs[cat]/len(trainset)
        ###--------------------DEBUG STATEMENTS----------------------
        negLogProb=-log(CatNumDocs[cat]/len(trainset))
        wordDict = CatWordDict[cat]
        countCat = CatWordCountDict[cat]
        for w in listWords:
            countWordTrain=wordDict.get(w,0)
            ratio = (countWordTrain+1)/(countCat+vocabLength)
            negLogProb-=log(ratio)           
                         
        if minimumNegLogProb>negLogProb:
            minCategory=cat
            minimumNegLogProb=negLogProb

    liResults.append((fileName,minCategory,reuters.categories(fileids=fileName)[0]))

###--------------------DEBUG STATEMENTS----------------------
#for t in liResults:
 #   print t    
###--------------------DEBUG STATEMENTS----------------------


evaluation_multi_class(liResults,CatWordCountDict.keys())

print "The time taken by the trained classifier to assign labels"
print time.time() - start_time, "seconds"
