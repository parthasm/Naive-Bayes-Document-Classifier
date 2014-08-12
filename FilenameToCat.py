
def f2c(corpus,fileName):
    if corpus=='mr':
        from nltk.corpus import movie_reviews as mr
        return mr.categories(fileids = fileName)[0]    
    else:
        from nltk.corpus import reuters
        return reuters.categories(fileids = fileName)[0]    
    
    

