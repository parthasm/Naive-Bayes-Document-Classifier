from FilenameToCat import f2c
categoriesFilenameDict={}
CatNumDocs={}
def get_testset_trainset(corpus):
    if corpus=='mr':
        return get_testset_trainset_nltk_mr()
    else:
        return get_testset_trainset_nltk_reuters()
    #reuters is the default corpus
    

def startup():
    return [categoriesFilenameDict,CatNumDocs]

def get_testset_trainset_nltk_reuters():
    from nltk.corpus import reuters
    global categoriesFilenameDict
    global CatNumDocs
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
    return [[ f for f in trainset if f2c('reuters',f) in categoriesFilenameDict],
            [ f for f in testset if f2c('reuters',f) in categoriesFilenameDict]]            


def get_testset_trainset_nltk_mr(trainToTestRatio=0.3):
    from nltk.corpus import movie_reviews as mr
    train_test = [[],[]]
    for category in mr.categories():
        categoriesFilenameDict[category]=mr.fileids(categories=category)
    for cat in categoriesFilenameDict.keys():
        li = categoriesFilenameDict[cat]
        size=int(len(li)*trainToTestRatio)
        CatNumDocs[cat]=size
        train_test[0].extend(li[:size])
        train_test[1].extend(li[size:])
    return train_test
