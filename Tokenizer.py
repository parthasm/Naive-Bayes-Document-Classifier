from nltk.corpus import stopwords
import re

stopwords_english = set(stopwords.words('english'))

def get_list_tokens_nltk(corpus, file_name):
    string=''
    if corpus=='mr':
        from nltk.corpus import movie_reviews
        string = movie_reviews.raw(fileids=file_name)
    else:
        from nltk.corpus import reuters
        string = reuters.raw(fileids=file_name)
    list_words = re.split(r'\W+',string)
    return [w.lower() for w in list_words if w.isalpha() and len(w)>1 and w.lower() not in stopwords_english]

#!!!!!!!!------Possible Improvement: Stemming--------------#

#*************NON-GENERALIZED CODE***********************************
    #list_words = re.split(r'\s+',string)
    #list_words = [w[:w.find('/')] for w in list_words if w.find('/')!=-1  and w[w.find('/')+1:]=='jj']
    #list_words = [w.lower() for w in list_words if w.isalnum() and w not in stopwords_english]

    
    #!!!!!!!!------Possible Improvement: Stemming--------------#
#********************************************************************    
