/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <vector>
#include "utils.h"
#include "FoxUtil.h"
#include <iostream>
#include "Token.h"
#include "SentSplitter.h"
#include "Tokenizer.h"
#include "N3L.h"
#include "Argument_helper.h"
#include "Options.h"
#include "Tool.h"
#include "BestPerformance.h"


#include "NNade3.h"


using namespace std;


int main(int argc, char **argv)
{
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif


	string optionFile;
	string annotatedPath;
	string processedPath;
	string fold;



	dsr::Argument_helper ah;
	ah.new_named_string("annotated", "", "", "", annotatedPath);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("processed", "", "", "", processedPath);
	ah.new_named_string("fold", "", "", "", fold);



	ah.process(argc, argv);
	cout<<"annotated path: " <<annotatedPath <<endl;
	cout<<"processed path: "<<processedPath<<endl;



	Options options;
	options.load(optionFile);

	options.showOptions();

	Tool tool(options);

	vector< vector<fox::Sent> > processedGroups;
	vector< vector<ADEsentence> > annotatedGroups;
	loadAnnotatedFile(annotatedPath, annotatedGroups);
	loadProcessedFile(processedPath, processedGroups);

	if(!options.embFile.empty()) {
		cout<< "load pre-trained emb"<<endl;
		tool.w2v->loadFromBinFile(options.embFile, false, true);
	}

	vector<BestPerformance> bestAll;
	int currentFold = atoi(fold.c_str());
	for(int crossValid=0;crossValid<annotatedGroups.size();crossValid++) {
		if(currentFold>=0 && crossValid!=currentFold) {
			continue;
		}
		cout<<"###### group ###### "<<crossValid<<endl;

		NNade3 nnade(options);

		// crossValid as test, crossValid+1 as dev, other as train
		vector<fox::Sent> processedTest;
		vector<ADEsentence> annotatedTest;
		vector<fox::Sent> processedDev;
		vector<ADEsentence> annotatedDev;
		vector<fox::Sent> processedTrain;
		vector<ADEsentence> annotatedTrain;

		for(int groupIdx=0;groupIdx<annotatedGroups.size();groupIdx++) {
			if(groupIdx == crossValid) {
				for(int sentIdx=0; sentIdx<annotatedGroups[groupIdx].size(); sentIdx++) {
					processedTest.push_back(processedGroups[groupIdx][sentIdx]);
					annotatedTest.push_back(annotatedGroups[groupIdx][sentIdx]);
				}
			} else if(groupIdx == (crossValid+1)%annotatedGroups.size()) {
				for(int sentIdx=0; sentIdx<annotatedGroups[groupIdx].size(); sentIdx++) {
					processedDev.push_back(processedGroups[groupIdx][sentIdx]);
					annotatedDev.push_back(annotatedGroups[groupIdx][sentIdx]);
				}
			} else {
				for(int sentIdx=0; sentIdx<annotatedGroups[groupIdx].size(); sentIdx++) {
					processedTrain.push_back(processedGroups[groupIdx][sentIdx]);
					annotatedTrain.push_back(annotatedGroups[groupIdx][sentIdx]);
				}
			}

		}

		BestPerformance best = nnade.trainAndTest(processedTest, annotatedTest, processedDev, annotatedDev, processedTrain, annotatedTrain, tool);
		bestAll.push_back(best);
	}

	if(currentFold<0) {
		// marcro-average
		double pDev_Entity = 0;
		double rDev_Entity = 0;
		double f1Dev_Entity = 0;
		double pDev_Relation = 0;
		double rDev_Relation = 0;
		double f1Dev_Relation = 0;
		for(int i=0;i<bestAll.size();i++) {
			pDev_Entity += bestAll[i].dev_pEntity/bestAll.size();
			rDev_Entity += bestAll[i].dev_rEntity/bestAll.size();
			pDev_Relation += bestAll[i].dev_pRelation/bestAll.size();
			rDev_Relation += bestAll[i].dev_rRelation/bestAll.size();
		}
		f1Dev_Entity = f1(pDev_Entity, rDev_Entity);
		f1Dev_Relation = f1(pDev_Relation, rDev_Relation);

		cout<<"### marcro-average ###"<<endl;
		cout<<"dev entity p "<<pDev_Entity<<endl;
		cout<<"dev entity r "<<rDev_Entity<<endl;
		cout<<"dev entity f1 "<<f1Dev_Entity<<endl;
		cout<<"dev relation p "<<pDev_Relation<<endl;
		cout<<"dev relation r "<<rDev_Relation<<endl;
		cout<<"dev relation f1 "<<f1Dev_Relation<<endl;


		double pTest_Entity = 0;
		double rTest_Entity = 0;
		double f1Test_Entity = 0;
		double pTest_Relation = 0;
		double rTest_Relation = 0;
		double f1Test_Relation = 0;
		for(int i=0;i<bestAll.size();i++) {
			pTest_Entity += bestAll[i].test_pEntity/bestAll.size();
			rTest_Entity += bestAll[i].test_rEntity/bestAll.size();
			pTest_Relation += bestAll[i].test_pRelation/bestAll.size();
			rTest_Relation += bestAll[i].test_rRelation/bestAll.size();
		}
		f1Test_Entity = f1(pTest_Entity, rTest_Entity);
		f1Test_Relation = f1(pTest_Relation, rTest_Relation);

		cout<<"test entity p "<<pTest_Entity<<endl;
		cout<<"test entity r "<<rTest_Entity<<endl;
		cout<<"test entity f1 "<<f1Test_Entity<<endl;
		cout<<"test relation p "<<pTest_Relation<<endl;
		cout<<"test relation r "<<rTest_Relation<<endl;
		cout<<"test relation f1 "<<f1Test_Relation<<endl;

	}



#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif

    return 0;

}

