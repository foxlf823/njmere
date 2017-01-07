
#ifndef NNADE_cooccur_H_
#define NNADE_cooccur_H_

#include <iosfwd>
#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "N3Lhelper.h"
#include "Utf.h"
#include "Token.h"
#include "Sent.h"
#include <sstream>
#include "Dependency.h"
#include "N3L.h"
#include "EnglishPos.h"
#include "Punctuation.h"
#include "Word2Vec.h"
#include "utils.h"
#include "Example.h"
#include "BestPerformance.h"
#include "Classifier3_base.h"
#include "Classifier3_char.h"
#include "Classifier3_pos.h"
#include "Classifier3_ner.h"

#include "Classifier3_label.h"
#include "Classifier3_dep.h"
#include "Classifier3_entity.h"

#include "Classifier3_nosdp.h"
#include "Classifier3_nojoint.h"

#include "Classifier3.h"

using namespace nr;
using namespace std;

// a implement of ACL 2016 end-to-end relation extraction
// use relation f1 on the development set

class NNade_cooccur {
public:
	Options m_options;

	Alphabet m_wordAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_nerAlphabet;
	Alphabet m_depAlphabet;
	Alphabet m_charAlphabet;

	string unknownkey;
	string nullkey;




	NNade_cooccur(const Options &options):m_options(options) {
		unknownkey = "-#unknown#-";
		nullkey = "-#null#-";
	}


  BestPerformance trainAndTest(vector<fox::Sent> & processedTest, vector<ADEsentence> & annotatedTest,
			vector<fox::Sent> & processedDev, vector<ADEsentence> & annotatedDev,
			vector<fox::Sent> & processedTrain, vector<ADEsentence> & annotatedTrain, Tool & tool) {

	  	BestPerformance ret = evaluateOnDev(tool, annotatedDev, processedDev, 0);


		return ret;
	}


	BestPerformance evaluateOnDev(Tool& tool, vector<ADEsentence> & annotated,
			vector<fox::Sent> & processed, int iter) {

		BestPerformance ret;

    	int ctGoldRelation = 0, ctPredictRelation = 0, ctCorrectRelation = 0;

    	int fp1Relations=0; // entity not match
    	int fp2Relations=0; // entity correct but relation wrong
    	int fn1Relations=0; // entity not find
    	int fn2Relations=0; // entity find but relation not find

		for(int sentIdx=0;sentIdx<processed.size();sentIdx++) {
			const fox::Sent & sent = processed[sentIdx];
			const ADEsentence & annotatedSent =annotated[sentIdx];

			vector<Relation> anwserRelations;

			for(int bIdx=0;bIdx<annotatedSent.entities.size();bIdx++) {
				const Entity& bEntity = annotatedSent.entities[bIdx];

				// a is before b
				for(int aIdx=0;aIdx<bIdx;aIdx++) {
					const Entity& aEntity = annotatedSent.entities[aIdx];

					// type constraint
					if(aEntity.type==bEntity.type)	{
						continue;
					}

					const Entity& Disease = aEntity.type==TYPE_Disease? aEntity:bEntity;
					const Entity& Chemical = aEntity.type==TYPE_Disease? bEntity:aEntity;

					Relation relation;
					newRelation(relation, Disease, Chemical);
					anwserRelations.push_back(relation);


				} // aIdx

			} // bIdx

			// evaluate by ourselves
			ctGoldRelation += annotatedSent.relations.size();
			ctPredictRelation += anwserRelations.size();
			for(int i=0;i<anwserRelations.size();i++) {
				for(int j=0;j<annotatedSent.relations.size();j++) {
					if(anwserRelations[i].equals(annotatedSent.relations[j])) {
						ctCorrectRelation ++;
						break;
					}
				}
			}

			for(int i=0;i<anwserRelations.size();i++) {
				// try to match gold entities
				bool entity1Matched = false;
				bool entity2Matched = false;

				for(int k=0;k<annotatedSent.entities.size();k++) {
					if(entity1Matched==false && anwserRelations[i].entity1.equals(annotatedSent.entities[k])) {
						entity1Matched = true;
					}
					if(entity2Matched==false && anwserRelations[i].entity2.equals(annotatedSent.entities[k])) {
						entity2Matched = true;
					}

					if(entity1Matched && entity2Matched)
						break;
				}

				if(!entity1Matched || !entity2Matched) {
					//fp1Relations.push_back(anwserRelations[i]);
					fp1Relations++;
				} else {
					// try to match gold relations

					int j=0;
					for(;j<annotatedSent.relations.size();j++) {
						if(anwserRelations[i].equals(annotatedSent.relations[j])) {
							break;
						}
					}

					if(j>=annotatedSent.relations.size()) {
						//fp2Relations.push_back(anwserRelations[i]);
						fp2Relations++;
					} else {
						// TP
					}

				}

			}

			for(int i=0;i<annotatedSent.relations.size();i++) {
				// try to match predicted entities
				bool entity1Matched = false;
				bool entity2Matched = false;

				for(int k=0;k<annotatedSent.entities.size();k++) {
					if(entity1Matched==false && annotatedSent.relations[i].entity1.equals(annotatedSent.entities[k])) {
						entity1Matched = true;
					}
					if(entity2Matched==false && annotatedSent.relations[i].entity2.equals(annotatedSent.entities[k])) {
						entity2Matched = true;
					}

					if(entity1Matched && entity2Matched)
						break;
				}

				if(!entity1Matched || !entity2Matched) {
					//fn1Relations.push_back(annotatedSent.relations[i]);
					fn1Relations++;
				} else {
					// try to match predicted relations
					int j=0;
					for(;j<anwserRelations.size();j++) {
						if(annotatedSent.relations[i].equals(anwserRelations[j])) {
							break;
						}
					}

					if(j>=anwserRelations.size()) {
						//fn2Relations.push_back(annotatedSent.relations[i]);
						fn2Relations++;
					} else {
						// TP
					}

				}
			}


		} // sent


		ret.dev_pRelation = precision(ctCorrectRelation, ctPredictRelation);
		ret.dev_rRelation = recall(ctCorrectRelation, ctGoldRelation);
		ret.dev_f1Relation = f1(ctCorrectRelation, ctGoldRelation, ctPredictRelation);

		cout<<"dev relation p: "<<ret.dev_pRelation<<", r:"<<ret.dev_rRelation<<", f1:"<<ret.dev_f1Relation<<endl;

		cout<<"fp entity wrong "<<fp1Relations/*.size()/ctTotalError*/<<endl;

		cout<<"fp entity correct but relation wrong "<<fp2Relations/*.size()/ctTotalError*/<<endl;

		cout<<"fn entity not found "<<fn1Relations/*.size()/ctTotalError*/<<endl;

		cout<<"fn entity found but relation lost "<<fn2Relations/*.size()/ctTotalError*/<<endl;


		return ret;
	}





};



#endif /* NNBB3_H_ */

