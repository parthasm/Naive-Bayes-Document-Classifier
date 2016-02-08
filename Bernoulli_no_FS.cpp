#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <map>
#include <cstdlib>    // library with the exit function
#include <sstream>
#include <list>
#include <set>
#include <ctime>
#include <cmath>
#include <vector>

void prune_rare_categories(std::map<std::string,int >& cate_num_docs_map,const std::map<std::string,int >& cate_num_docs_test_map)
{
	std::map<std::string,int >::iterator itr;
	std::map<std::string,int >::const_iterator citr;
	std::list<std::string> low_cates;
	std::list<std::string>::const_iterator cl_itr;
	
	for(citr=cate_num_docs_map.begin();citr!=cate_num_docs_map.end();++citr)
	{
		if(citr->second <=20)
			low_cates.push_back(citr->first);
	}
	for(citr=cate_num_docs_test_map.begin();citr!=cate_num_docs_test_map.end();++citr)
	{
		if(citr->second <=20)
			low_cates.push_back(citr->first);
	}	
	
	
	for(cl_itr = low_cates.begin(); cl_itr!=low_cates.end(); ++cl_itr)
	{
		itr = cate_num_docs_map.find(*cl_itr);
		if(itr!=cate_num_docs_map.end())
			cate_num_docs_map.erase(itr);
	}

}

void f_measure(const std::map<std::string, std::vector<unsigned int> >& result_cates_map)
{
	unsigned int num_cats_undefined_prec=0;
	unsigned int a_sum=0;
	unsigned int b_sum=0;
	unsigned int c_sum=0;
	unsigned int d_sum=0;
	
	double prec;
	double rec;
	double macro_prec;
	double macro_rec;
	
	for(std::map<std::string, std::vector<unsigned int> >::const_iterator cit = result_cates_map.begin();cit!=result_cates_map.end();++cit)
	{
		std::string categ;
		categ = cit->first;
		if(cit->second[0]+cit->second[1]==0)
		{
			std::cout << "Precision is undefined for the category " << categ << "\n";
			std::cout << "Category " << categ << " is excluded from precision and recall calculations.\n";
			++num_cats_undefined_prec;
		}
		else
		{
			prec += (cit->second[0]/((1.0*cit->second[0])+cit->second[1]));
			rec += (cit->second[0]/((1.0*cit->second[0])+cit->second[2]));
			a_sum += cit->second[0];
			b_sum += cit->second[1];
			c_sum += cit->second[2];
			d_sum += cit->second[3];
		}
	}
	
	macro_prec = prec/((result_cates_map.size()*1.0)-num_cats_undefined_prec);
	macro_rec = rec/((result_cates_map.size()*1.0)-num_cats_undefined_prec);
	
	std::cout << "Macro Precision = " << macro_prec << "\n";
	std::cout << "Macro Recall = " << macro_rec << "\n";
	std::cout << "Macro F-measure = " << (2*macro_prec*macro_rec)/(macro_prec+macro_rec) << "\n";
	
	std::cout << "Micro Precision = " << a_sum/((a_sum*1.0)+b_sum) << "\n";
	std::cout << "Micro Recall = " << a_sum/((a_sum*1.0)+c_sum) << "\n";
	std::cout << "Micro F-measure = " << (2*a_sum)/((2.0*a_sum)+b_sum+c_sum) << "\n";
}

int main()
{
	time_t start_time = time(NULL);
	std::string data_path("D:/Zmisc/Github/NLP/Naive-Bayes-Document-Classifier-master/trial/reuters_modified/"); //change this path according to your local data path
	std::ifstream file_cate((data_path+std::string("cats.txt")).c_str());
	//if file can't be opened, we exit
	if(!file_cate.good())
	{
		std::cerr << "Can't open cats.txt to read.\n";
		//exit(1); which is the same as 
		return EXIT_FAILURE;
	}
	
	
	std::string line;
	std::string token;
	std::string word1;
	std::string word2;
	std::string file_name;
	std::string min_cat;
	std::string train("training");
	std::string tes("test");
	
	//using the tree map implementation with O(log n) time for insertion, search, deletion etc 
	//because there are no hash table implementations in c++ stl
	
	std::map<std::string,std::string > test_map; //test file to category map
	std::map<std::string,std::string > train_map; //train file to category map
	std::map<std::string,std::string >::iterator fi_itr;
	std::map<std::string,std::string >::const_iterator cfi_itr;
	
	std::map<std::string,int > cate_num_docs_map;
	std::map<std::string,int > cate_num_docs_test_map;
	std::map<std::string,int >::iterator itr;
	std::map<std::string,int >::const_iterator citr;
	
	std::set<std::string> words_set; 
	std::set<std::string>::iterator set_itr;
	std::set<std::string>::const_iterator cset_itr;
	
	
	std::map<std::string, std::map<std::string,unsigned int> > word_cates_num_map;
	std::map<std::string, std::map<std::string,unsigned int> >::iterator big_itr;
	std::map<std::string, std::map<std::string,unsigned int> >::const_iterator cbig_itr;
	std::map<std::string,unsigned int>::iterator small_itr;
	std::map<std::string,unsigned int>::const_iterator csmall_itr;
//Create a map with a word as the key and a map as the value
// in the inner map the category as key and number of documents in that category where it occurs as value
	
	std::map<std::string, std::vector<unsigned int> > cates_results;
	std::map<std::string, std::vector<unsigned int> >::iterator res_itr;
	std::map<std::string, std::vector<unsigned int> >::const_iterator cres_itr;

	
	unsigned int len_train;
	unsigned int counter;
	unsigned int error_counter;
	unsigned int len_test;
	int nct;
	double time_diff;
	double minimum_neg_log_prob;
	double neg_log_prob;
	double ratio;
	
	
	
	
	//Format of each line of categs file - filename cate1 cate 2(if any) cate3(if any)
	while(getline(file_cate,line))
	{
		counter = 0;
		std::istringstream iss(line);
		while(iss >> token)
		{
			++counter;
			if(counter==1)
				word1 = token;
			else if(counter==2)
				word2 = token;
		}
			
		if(counter>2) //we only consider files which are in 1 category
			continue;
			
		if(word1.compare(0,4,tes)==0)
		{
			file_name = word1.substr(5);
			test_map[file_name] = word2;
			++cate_num_docs_test_map[word2];
		}
			
		
		
		else if(word1.compare(0,8,train)==0)
		{
			file_name = word1.substr(9);
			train_map[file_name] = word2;
			++cate_num_docs_map[word2];
		}

	}
	
	file_cate.close();
	
	
	
	//Removing categories with less than 21 files in either the training set or test set
	prune_rare_categories(cate_num_docs_map,cate_num_docs_test_map);
	
	
	for(cfi_itr = train_map.begin();cfi_itr!=train_map.end();++cfi_itr)
	{
		itr = cate_num_docs_map.find(cfi_itr->second);
		
		//File belongs to a low category
		if(itr==cate_num_docs_map.end())
			continue;
	
		file_name = data_path+std::string("training/")+cfi_itr->first;
		std::ifstream file_obj(file_name.c_str());
		//if file can't be opened, we exit
		if(!file_obj.good())
		{
			std::cerr << "Can't open training file to read.\n";
			//exit(1); which is the same as 
			return EXIT_FAILURE;
		}
		
		words_set.clear();
		while(file_obj >> token)
			words_set.insert(token);
		
		for(cset_itr = words_set.begin();cset_itr!=words_set.end();++cset_itr)
		{
			big_itr = word_cates_num_map.find(*cset_itr);	
			++word_cates_num_map[*cset_itr][cfi_itr->second];
		}
		file_obj.close();
		++len_train;
	}
	
	time_diff = difftime(time(NULL),start_time);
	std::cout << "The Classifier is trained and it took "<< time_diff << " seconds.\n" ;

	start_time = time(NULL);
	
	//Initialize cates_results
	for(citr=cate_num_docs_map.begin();citr!=cate_num_docs_map.end();++citr)		
	{
		cates_results[citr->first].push_back(0);
		cates_results[citr->first].push_back(0);
		cates_results[citr->first].push_back(0);
		cates_results[citr->first].push_back(0);
	}
	
	for(cfi_itr = test_map.begin();cfi_itr!=test_map.end();++cfi_itr)
	{
		itr = cate_num_docs_map.find(cfi_itr->second);
		
		//File belongs to a low category
		if(itr==cate_num_docs_map.end())
			continue;
		
		
		file_name = data_path+std::string("test/")+cfi_itr->first;
		std::ifstream file_obj(file_name.c_str());
		//if file can't be opened, we exit
		if(!file_obj.good())
		{
			std::cerr << "Can't open test file to read.\n";
			//exit(1); which is the same as 
			return EXIT_FAILURE;
		}
		
		words_set.clear();
		while(file_obj >> token)
			words_set.insert(token);
	
		minimum_neg_log_prob=1000000000;
		min_cat = "";
	
		for(citr=cate_num_docs_map.begin();citr!=cate_num_docs_map.end();++citr)		
		{
			neg_log_prob = -log(citr->second/(len_train*1.0));
			for(cbig_itr = word_cates_num_map.begin(); cbig_itr!=word_cates_num_map.end();++cbig_itr)
			{
				set_itr = words_set.find(cbig_itr->first);
				nct = word_cates_num_map[cbig_itr->first][citr->first];
				ratio = (nct+1)/(citr->second+2.0);
				if(set_itr!=words_set.end())
					neg_log_prob-= log(ratio);
				else
					neg_log_prob-=log(1-ratio);
			}
		
			if(minimum_neg_log_prob>neg_log_prob)
			{
				min_cat = citr->first;
				minimum_neg_log_prob = neg_log_prob;
			}
		}
		file_obj.close();
		++len_test;
		
		if(min_cat!=cfi_itr->second)
			++error_counter;
		
		for(cres_itr = cates_results.begin();cres_itr!=cates_results.end();++cres_itr)
		{
			word2 = cres_itr->first;
			if(word2==min_cat)
			{
				if(word2==cfi_itr->second)
					++cates_results[word2][0];
				else
					++cates_results[word2][1];
			}
			else
			{
				if(word2==cfi_itr->second)
					++cates_results[word2][2];
				else
					++cates_results[word2][3];
			}
		}
	}

	std::cout << "The fraction of errors is " << error_counter/(len_test*1.0) << "\n";
	
	//Evaluation by finer measures
	f_measure(cates_results);
	
	time_diff = difftime(time(NULL),start_time);
	std::cout << "The Classifier has run and it took "<< time_diff << " seconds.\n" ;	
	return 0;
}
