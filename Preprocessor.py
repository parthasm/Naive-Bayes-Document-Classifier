from Filename_To_Cat import f2c
categories_file_name_dict={}
cat_num_docs={}
def get_testset_trainset(corpus):
    if corpus=='mr':
        return get_testset_trainset_nltk_mr()
    else:
        return get_testset_trainset_nltk_reuters()
    #reuters is the default corpus
    

def startup():
    return [categories_file_name_dict,cat_num_docs]

def get_testset_trainset_nltk_reuters():
    from nltk.corpus import reuters
    global categories_file_name_dict
    global cat_num_docs
    clean_files = [f for f in reuters.fileids() if len(reuters.categories(fileids=f))==1]    
    testset = [f for f in clean_files if f[:5]=='test/']
    trainset = [f for f in clean_files if f[:9]=='training/']
    for cat in reuters.categories():
        li=[f for f in reuters.fileids(categories=cat) if f in trainset]
        li_te = [f for f in reuters.fileids(categories=cat) if f in testset]
        if len(li)>20 and len(li_te)>20:
            cat_num_docs[cat]=len(li)
            li.extend(li_te)
            categories_file_name_dict[cat]=li
    return [[ f for f in trainset if f2c('reuters',f) in categories_file_name_dict],
            [ f for f in testset if f2c('reuters',f) in categories_file_name_dict]]            


def get_testset_trainset_nltk_mr(train_to_test_ratio=0.3):
    from nltk.corpus import movie_reviews as mr
    train_test = [[],[]]
    for category in mr.categories():
        categories_file_name_dict[category]=mr.fileids(categories=category)
    for cat in categories_file_name_dict.keys():
        li = categories_file_name_dict[cat]
        size=int(len(li)*train_to_test_ratio)
        cat_num_docs[cat]=size
        train_test[0].extend(li[:size])
        train_test[1].extend(li[size:])
    return train_test
