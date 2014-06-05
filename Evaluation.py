from __future__ import division
#Create a dictionary with category as the key and a list of 4 numbers as the value
 #These values are a) Number of docs in the category identified correctly a
                  #b) Number of docs identified incorrectly as in the category b
                  #c) Number of docs identified incorrectly as not in the category c
                  #d) Number of docs identified correctly as not in the category d
def evaluation_multi_class(liResults,listCats):
    CatResultsDict = {}
    for cat in listCats:
        CatResultsDict[cat]=[0,0,0,0]
        for t in liResults:
            if cat==t[1]:
                if cat==t[2]:
                    CatResultsDict[cat][0]+=1
                else:
                    CatResultsDict[cat][1]+=1
            else:
                if cat==t[2]:
                    CatResultsDict[cat][2]+=1
                else:
                    CatResultsDict[cat][3]+=1

    totPrec=0
    totRec=0
    A=0
    B=0
    C=0
    D=0
    for cat in CatResultsDict:
        a = CatResultsDict[cat][0]
        b = CatResultsDict[cat][1]
        c = CatResultsDict[cat][2]
        d = CatResultsDict[cat][3]
        totPrec+=a/(a+b)##Precision for this category
        totRec+=a/(a+c)##Recall for this category
        A+=a
        B+=b
        C+=c
        D+=d
###--------------------DEBUG STATEMENTS----------------------
    #print cat, a
    #print cat, b
    #print cat, c
    #print cat, d
    #print (a+b+c+d)==len(testset)
###--------------------DEBUG STATEMENTS----------------------
    MacroPrec = totPrec/len(CatResultsDict)
    MacroRec = totRec/len(CatResultsDict)
    MacroF = (2*MacroPrec*MacroRec)/(MacroPrec+MacroRec)

    MicroPrec = A/(A+B)
    MicroRec = A/(A+C)
    MicroF = (2*MicroPrec*MicroRec)/(MicroPrec+MicroRec)

    print "Macro Precision=",MacroPrec
    print "Macro Recall=",MacroRec
    print "Macro F-measure=",MacroF

    print "Micro Precision=",MicroPrec
    print "Micro Recall=",MicroRec
    print "Micro F-measure=",MicroF

    


    numErrors = sum(t[1]!=t[2] for t in liResults)
    print "Fraction of mis-classifications" , numErrors/len(liResults)


def evaluation_binary(liResults,listCats):
      
 #Calculate the precision, reacall and f-measure  
    a=0
    b=0
    c=0
    d=0
    if len(listCats)<3:
        cat = listCats[0]
        for t in liResults:
            if cat==t[1]:
                if cat==t[2]:
                    a+=1
                else:
                    b+=1
            else:
                if cat==t[2]:
                    c+=1
                else:
                    d+=1
        Precision = a/(a+b)
        Recall = a/(a+c)

        print "Precision =", Precision
        print "Recall =", Recall
        print "F-measure =", (2*Precision*Recall)/(Precision+Recall)

    else:
        print "Please use the evaluation_multi_class function"

###--------------------DEBUG STATEMENTS----------------------
#print (a+b+c+d)==len(testset)
###--------------------DEBUG STATEMENTS----------------------

    numErrors = sum(t[1]!=t[2] for t in liResults)
    print "Fraction of Errors = ", numErrors/len(liResults)
