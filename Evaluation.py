from __future__ import division
#Create a dictionary with category as the key and a list of 4 numbers as the value
 #These values are a) Number of docs in the category identified correctly a
                  #b) Number of docs identified incorrectly as in the category b
                  #c) Number of docs identified incorrectly as not in the category c
                  #d) Number of docs identified correctly as not in the category d
def evaluation_multi_class(li_results,list_cats):
    cat_results_dict = {}
    for cat in list_cats:
        cat_results_dict[cat]=[0,0,0,0]
        for t in li_results:
            if cat==t[1]:
                if cat==t[2]:
                    cat_results_dict[cat][0]+=1
                else:
                    cat_results_dict[cat][1]+=1
            else:
                if cat==t[2]:
                    cat_results_dict[cat][2]+=1
                else:
                    cat_results_dict[cat][3]+=1
    num_cats_undefined_prec = 0
    tot_prec=0
    tot_rec=0
    a_sum=0
    b_sum=0
    c_sum=0
    d_sum=0
    for cat in cat_results_dict:
        a = cat_results_dict[cat][0]
        b = cat_results_dict[cat][1]
        c = cat_results_dict[cat][2]
        d = cat_results_dict[cat][3]
        #print cat, a, b, c, d
        if a+b==0:
            print "precision is undefined for category ", cat
            print "This category is excluded from precision and recall calculations"
            num_cats_undefined_prec+=1
        else:            
            tot_prec+=a/(a+b)##precision for this category
            tot_rec+=a/(a+c)##recall for this category
            a_sum+=a
            b_sum+=b
            c_sum+=c
            d_sum+=d
###--------------------DEBUG STATEMENTS----------------------
    #print cat, a
    #print cat, b
    #print cat, c
    #print cat, d
    #print (a+b+c+d)==len(testset)
###--------------------DEBUG STATEMENTS----------------------
    macro_precision = tot_prec/(len(cat_results_dict)-num_cats_undefined_prec)
    macro_recall = tot_rec/(len(cat_results_dict)-num_cats_undefined_prec)
    macro_f_measure = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)

    micro_precision = a_sum/(a_sum+b_sum)
    micro_recall = a_sum/(a_sum+c_sum)
    micro_f_measure = (2*micro_precision*micro_recall)/(micro_precision+micro_recall)

    print "Macro precision=",macro_precision
    print "Macro recall=",macro_recall
    print "Macro F-measure=",macro_f_measure

    print "Micro precision=",micro_precision
    print "Micro recall=",micro_recall
    print "Micro F-measure=",micro_f_measure

    


    evaluation_fraction_misclass(li_results)

def evaluation_binary(li_results):
      
 #Calculate the precision, recall and f-measure  
    a=0
    b=0
    c=0
    d=0
    cat = li_results[0][1]
    for t in li_results:
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
    precision = a/(a+b)
    recall = a/(a+c)
    print "The following parameters are recorded for the category " , cat
    print "precision =", precision
    print "recall =", recall
    print "F-measure =", (2*precision*recall)/(precision+recall)


###--------------------DEBUG STATEMENTS----------------------
#print (a+b+c+d)==len(testset)
###--------------------DEBUG STATEMENTS----------------------

    evaluation_fraction_misclass(li_results)
    
def evaluation_fraction_misclass(li_results):
    num_errors = sum(t[1]!=t[2] for t in li_results)
    print "Fraction of Errors = ", num_errors/len(li_results)
