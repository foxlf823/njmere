
#ifndef NNADE3_H_
#define NNADE3_H_

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


#include "Classifier3.h"
using namespace nr;
using namespace std;


// use relation f1 on the development set

class NNade3 {
public:
	Options m_options;

	Alphabet m_wordAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_nerAlphabet;
	Alphabet m_depAlphabet;
	Alphabet m_charAlphabet;

	string unknownkey;
	string nullkey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else

  Classifier3<cpu> m_classifier;
#endif


  NNade3(const Options &options):m_options(options) {
		unknownkey = "-#unknown#-";
		nullkey = "-#null#-";
	}


  BestPerformance trainAndTest(vector<fox::Sent> & processedTest, vector<ADEsentence> & annotatedTest,
			vector<fox::Sent> & processedDev, vector<ADEsentence> & annotatedDev,
			vector<fox::Sent> & processedTrain, vector<ADEsentence> & annotatedTrain, Tool & tool) {

	  	BestPerformance ret;

		m_wordAlphabet.clear();
		m_wordAlphabet.from_string(unknownkey);
		m_wordAlphabet.from_string(nullkey);

		m_posAlphabet.clear();
		m_posAlphabet.from_string(unknownkey);
		m_posAlphabet.from_string(nullkey);

		m_depAlphabet.clear();
		m_depAlphabet.from_string(unknownkey);
		m_depAlphabet.from_string(nullkey);

		m_charAlphabet.clear();
		m_charAlphabet.from_string(unknownkey);
		m_charAlphabet.from_string(nullkey);

		// ner alphabet should be initialized directly, not from the dataset
		m_nerAlphabet.clear();
		m_nerAlphabet.from_string(unknownkey);
		m_nerAlphabet.from_string(nullkey);
		hash_map<string, int> ner_stat;
		ner_stat[B_Disease]++;
		ner_stat[I_Disease]++;
		ner_stat[L_Disease]++;
		ner_stat[U_Disease]++;
		ner_stat[B_Chemical]++;
		ner_stat[I_Chemical]++;
		ner_stat[L_Chemical]++;
		ner_stat[U_Chemical]++;
		ner_stat[OTHER]++;
		stat2Alphabet(ner_stat, m_nerAlphabet, "ner");

		createAlphabet(processedTrain, tool);

		if (!m_options.wordEmbFineTune) {
			createAlphabet(processedDev, tool);
			createAlphabet(processedTest, tool);
		}


		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				randomInitNrmat(wordEmb, m_wordAlphabet, m_options.wordEmbSize, 1000);
			} else {

				double* emb = new double[m_wordAlphabet.size()*m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				delete[] emb;
			}
		} else {
			if(m_options.embFile.empty()) {
				assert(0);
			} else {

				double* emb = new double[m_wordAlphabet.size()*m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				delete[] emb;
			}
		}

		NRMat<dtype> posEmb;
		randomInitNrmat(posEmb, m_posAlphabet, m_options.otherEmbSize, 1010);
		NRMat<dtype> nerEmb;
		randomInitNrmat(nerEmb, m_nerAlphabet, m_options.otherEmbSize, 1020);
		NRMat<dtype> depEmb;
		randomInitNrmat(depEmb, m_depAlphabet, m_options.otherEmbSize, 1030);
		NRMat<dtype> charEmb;
		randomInitNrmat(charEmb, m_charAlphabet, m_options.otherEmbSize, 1040);

		vector<Example> trainExamples;
		initialTrainingExamples(tool, annotatedTrain, processedTrain, trainExamples);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;

		  m_classifier.init(m_options);

		  m_classifier._words.initial(wordEmb);
		  m_classifier._words.setEmbFineTune(m_options.wordEmbFineTune);

		m_classifier._pos.initial(posEmb);
		m_classifier._pos.setEmbFineTune(true);

		m_classifier._dep.initial(depEmb);
		m_classifier._dep.setEmbFineTune(true);

		m_classifier._ner.initial(nerEmb);
		m_classifier._ner.setEmbFineTune(true);

		m_classifier._char.initial(charEmb);
		m_classifier._char.setEmbFineTune(true);


		static vector<Example> subExamples;

		dtype best = 0;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    // this coding block should be identical to initialTrainingExamples
		    // we don't regenerate training examples, instead fetch them directly
		    int exampleIdx = 0;

			for(int sentIdx=0;sentIdx<processedTrain.size();sentIdx++) {
				const fox::Sent & sent = processedTrain[sentIdx];
				const ADEsentence & annotatedSent = annotatedTrain[sentIdx];

				for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
					const fox::Token& token = sent.tokens[tokenIdx];

					subExamples.clear();
					subExamples.push_back(trainExamples[exampleIdx]);
					int curUpdateIter = iter * trainExamples.size() + exampleIdx;

					dtype cost = m_classifier.processNer(subExamples, curUpdateIter);

					//m_classifier.checkgradsNer(subExamples, curUpdateIter);
					m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);


					exampleIdx++;
				} // token

				for(int bIdx=0;bIdx<annotatedSent.entities.size();bIdx++) {
					const Entity& bEntity = annotatedSent.entities[bIdx];

					// a is before b
					for(int aIdx=0;aIdx<bIdx;aIdx++) {
						const Entity& aEntity = annotatedSent.entities[aIdx];

						// type constraint
						if(aEntity.type==bEntity.type)	{
							continue;
						}

						subExamples.clear();
						subExamples.push_back(trainExamples[exampleIdx]);
						int curUpdateIter = iter * trainExamples.size() + exampleIdx;

						dtype cost = m_classifier.processRel(subExamples, curUpdateIter);
						//m_classifier.checkgradsRel(subExamples, curUpdateIter);
						m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);


						exampleIdx++;

					} // aIdx

				} // bIdx


			} // sent



		    // an iteration end, begin to evaluate
		    if((iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordEmbFineTune && m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    		averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);

		    		averageUnkownEmb(m_depAlphabet, m_classifier._dep, m_options.otherEmbSize);

		    		averageUnkownEmb(m_nerAlphabet, m_classifier._ner, m_options.otherEmbSize);


		    		averageUnkownEmb(m_charAlphabet, m_classifier._char, m_options.otherEmbSize);
		    	}

		    	BestPerformance currentDev = evaluateOnDev(tool, annotatedDev, processedDev, iter);

				if (currentDev.dev_f1Relation > best) {
					cout << "Exceeds best performance of " << best << endl;
					best = currentDev.dev_f1Relation;

					BestPerformance currentTest = test(tool, annotatedTest, processedTest);


					ret.dev_pEntity = currentDev.dev_pEntity;
					ret.dev_rEntity = currentDev.dev_rEntity;
					ret.dev_f1Entity = currentDev.dev_f1Entity;
					ret.dev_pRelation = currentDev.dev_pRelation;
					ret.dev_rRelation = currentDev.dev_rRelation;
					ret.dev_f1Relation = currentDev.dev_f1Relation;
					ret.test_pEntity = currentTest.test_pEntity;
					ret.test_rEntity = currentTest.test_rEntity;
					ret.test_f1Entity = currentTest.test_f1Entity;
					ret.test_pRelation = currentTest.test_pRelation;
					ret.test_rRelation = currentTest.test_rRelation;
					ret.test_f1Relation = currentTest.test_f1Relation;

				}

		    }

		} // for iter

		m_classifier.release();

		return ret;
	}


	void initialTrainingExamples(Tool& tool, vector<ADEsentence> & annotated, vector<fox::Sent> & processed,
			vector<Example> & examples) {


		for(int sentIdx=0;sentIdx<processed.size();sentIdx++) {
			const fox::Sent & sent = processed[sentIdx];
			const ADEsentence & annotatedSent =annotated[sentIdx];
			string lastNerLabel = nullkey;
			vector<string> labelSequence;

			for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
				const fox::Token& token = sent.tokens[tokenIdx];
				int entityIdx = -1;
				string schemaLabel = OTHER;

				for(int i=0;i<annotatedSent.entities.size();i++) {
					const Entity& entity = annotatedSent.entities[i];

					string temp = isTokenInEntity(token, entity);
					if(temp != OTHER) {
						entityIdx = i;
						schemaLabel = temp;
						break;
					}
				}

				Example eg(false);
				string labelName = entityIdx!=-1 ? schemaLabel+"_"+annotatedSent.entities[entityIdx].type : OTHER;
				generateOneNerExample(eg, labelName, sent, lastNerLabel, tokenIdx);
				labelSequence.push_back(labelName);

				examples.push_back(eg);

				lastNerLabel = labelName;
			} // token

			for(int bIdx=0;bIdx<annotatedSent.entities.size();bIdx++) {
				const Entity& bEntity = annotatedSent.entities[bIdx];

				// a is before b
				for(int aIdx=0;aIdx<bIdx;aIdx++) {
					const Entity& aEntity = annotatedSent.entities[aIdx];

					// type constraint
					if((aEntity.type==bEntity.type))	{
						continue;
					}

					const Entity& Disease = aEntity.type==TYPE_Disease? aEntity:bEntity;
					const Entity& Chemical = aEntity.type==TYPE_Disease? bEntity:aEntity;

					Example eg(true);
					string labelName = Not_ADE;
					if(isADE(Disease, Chemical, annotatedSent)) {
						labelName = ADE;
					}

					generateOneRelExample(eg, labelName, sent, Disease, Chemical, labelSequence);

					examples.push_back(eg);

				} // aIdx

			} // bIdx


		} // sent



	}

	BestPerformance evaluateOnDev(Tool& tool, vector<ADEsentence> & annotated,
			vector<fox::Sent> & processed, int iter) {

		BestPerformance ret;

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;
    	int ctGoldRelation = 0, ctPredictRelation = 0, ctCorrectRelation = 0;


		for(int sentIdx=0;sentIdx<processed.size();sentIdx++) {
			const fox::Sent & sent = processed[sentIdx];
			const ADEsentence & annotatedSent =annotated[sentIdx];
			string lastNerLabel = nullkey;
			vector<string> labelSequence;

			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;


			for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
				const fox::Token& token = sent.tokens[tokenIdx];

				Example eg(false);
				string fakeLabelName = "";
				generateOneNerExample(eg, fakeLabelName, sent, lastNerLabel, tokenIdx);

				vector<dtype> probs;
				int predicted = m_classifier.predictNer(eg, probs);

				string labelName = NERlabelID2labelName(predicted);
				labelSequence.push_back(labelName);

				// decode entity label
				if(labelName == B_Disease || labelName == U_Disease ||
						labelName == B_Chemical || labelName == U_Chemical) {
					Entity entity;
					newEntity(token, labelName, entity);
					anwserEntities.push_back(entity);
				} else if(labelName == I_Disease || labelName == L_Disease ||
						labelName == I_Chemical || labelName == L_Chemical) {
					if(checkWrongState(labelSequence)) {
						Entity& entity = anwserEntities[anwserEntities.size()-1];
						appendEntity(token, entity);
					}
				}

				lastNerLabel = labelName;
			} // token

			for(int bIdx=0;bIdx<anwserEntities.size();bIdx++) {
				const Entity& bEntity = anwserEntities[bIdx];

				// a is before b
				for(int aIdx=0;aIdx<bIdx;aIdx++) {
					const Entity& aEntity = anwserEntities[aIdx];

					// type constraint
					if(aEntity.type==bEntity.type)	{
						continue;
					}

					const Entity& Disease = aEntity.type==TYPE_Disease? aEntity:bEntity;
					const Entity& Chemical = aEntity.type==TYPE_Disease? bEntity:aEntity;

					Example eg(true);
					string fakeLabelName = "";
					generateOneRelExample(eg, fakeLabelName, sent, Disease, Chemical, labelSequence);

					vector<dtype> probs;
					int predicted = m_classifier.predictRel(eg, probs);

					string labelName = RellabelID2labelName(predicted);

					// decode relation label
					if(labelName == ADE) {
						Relation relation;
						newRelation(relation, Disease, Chemical);
						anwserRelations.push_back(relation);
					}

				} // aIdx

			} // bIdx

			// evaluate by ourselves
			ctGoldEntity += annotatedSent.entities.size();
			ctPredictEntity += anwserEntities.size();
			for(int i=0;i<anwserEntities.size();i++) {
				int k=-1;
				int j=0;
				for(;j<annotatedSent.entities.size();j++) {
					if(anwserEntities[i].equalsBoundary(annotatedSent.entities[j])) {
						if(anwserEntities[i].equalsType(annotatedSent.entities[j])) {
							ctCorrectEntity ++;
							break;
						} else {
							k = j;
							break;
						}
					}
				}



			}


			ctGoldRelation += annotatedSent.relations.size();
			ctPredictRelation += anwserRelations.size();
			for(int i=0;i<anwserRelations.size();i++) {
				int j=0;
				for(;j<annotatedSent.relations.size();j++) {
					if(anwserRelations[i].equals(annotatedSent.relations[j])) {
						ctCorrectRelation ++;
						break;
					}
				}


			}



		} // sent

		ret.dev_pEntity = precision(ctCorrectEntity, ctPredictEntity);
		ret.dev_rEntity = recall(ctCorrectEntity, ctGoldEntity);
		ret.dev_f1Entity = f1(ctCorrectEntity, ctGoldEntity, ctPredictEntity);
		ret.dev_pRelation = precision(ctCorrectRelation, ctPredictRelation);
		ret.dev_rRelation = recall(ctCorrectRelation, ctGoldRelation);
		ret.dev_f1Relation = f1(ctCorrectRelation, ctGoldRelation, ctPredictRelation);

		cout<<"dev entity p: "<<ret.dev_pEntity<<", r:"<<ret.dev_rEntity<<", f1:"<<ret.dev_f1Entity<<endl;
		cout<<"dev relation p: "<<ret.dev_pRelation<<", r:"<<ret.dev_rRelation<<", f1:"<<ret.dev_f1Relation<<endl;


		return ret;
	}

	BestPerformance test(Tool& tool, vector<ADEsentence> & annotated, vector<fox::Sent> & processed) {

		BestPerformance ret;

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;
    	int ctGoldRelation = 0, ctPredictRelation = 0, ctCorrectRelation = 0;

		for(int sentIdx=0;sentIdx<processed.size();sentIdx++) {
			const fox::Sent & sent = processed[sentIdx];
			const ADEsentence & annotatedSent =annotated[sentIdx];
			string lastNerLabel = nullkey;
			vector<string> labelSequence;

			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;

			for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
				const fox::Token& token = sent.tokens[tokenIdx];

				Example eg(false);
				string fakeLabelName = "";
				generateOneNerExample(eg, fakeLabelName, sent, lastNerLabel, tokenIdx);

				vector<dtype> probs;
				int predicted = m_classifier.predictNer(eg, probs);

				string labelName = NERlabelID2labelName(predicted);
				labelSequence.push_back(labelName);

				// decode entity label
				if(labelName == B_Disease || labelName == U_Disease ||
						labelName == B_Chemical || labelName == U_Chemical ) {
					Entity entity;
					newEntity(token, labelName, entity);
					anwserEntities.push_back(entity);
				} else if(labelName == I_Disease || labelName == L_Disease ||
						labelName == I_Chemical || labelName == L_Chemical ) {
					if(checkWrongState(labelSequence)) {
						Entity& entity = anwserEntities[anwserEntities.size()-1];
						appendEntity(token, entity);
					}
				}

				lastNerLabel = labelName;
			} // token


			for(int bIdx=0;bIdx<anwserEntities.size();bIdx++) {
				const Entity& bEntity = anwserEntities[bIdx];

				// a is before b
				for(int aIdx=0;aIdx<bIdx;aIdx++) {
					const Entity& aEntity = anwserEntities[aIdx];

					// type constraint
					if(aEntity.type==bEntity.type)	{
						continue;
					}

					const Entity& Disease = aEntity.type==TYPE_Disease? aEntity:bEntity;
					const Entity& Chemical = aEntity.type==TYPE_Disease? bEntity:aEntity;

					Example eg(true);
					string fakeLabelName = "";
					generateOneRelExample(eg, fakeLabelName, sent, Disease, Chemical, labelSequence);

					vector<dtype> probs;
					int predicted = m_classifier.predictRel(eg, probs);

					string labelName = RellabelID2labelName(predicted);

					// decode relation label
					if(labelName == ADE) {
						Relation relation;
						newRelation(relation, Disease, Chemical);
						anwserRelations.push_back(relation);
					}

				} // aIdx

			} // bIdx

			// evaluate by ourselves
			ctGoldEntity += annotatedSent.entities.size();
			ctPredictEntity += anwserEntities.size();
			for(int i=0;i<anwserEntities.size();i++) {
				for(int j=0;j<annotatedSent.entities.size();j++) {
					if(anwserEntities[i].equals(annotatedSent.entities[j])) {
						ctCorrectEntity ++;
						break;
					}
				}
			}

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


		} // sent


		ret.test_pEntity = precision(ctCorrectEntity, ctPredictEntity);
		ret.test_rEntity = recall(ctCorrectEntity, ctGoldEntity);
		ret.test_f1Entity = f1(ctCorrectEntity, ctGoldEntity, ctPredictEntity);
		ret.test_pRelation = precision(ctCorrectRelation, ctPredictRelation);
		ret.test_rRelation = recall(ctCorrectRelation, ctGoldRelation);
		ret.test_f1Relation = f1(ctCorrectRelation, ctGoldRelation, ctPredictRelation);

		cout<<"test entity p: "<<ret.test_pEntity<<", r:"<<ret.test_rEntity<<", f1:"<<ret.test_f1Entity<<endl;
		cout<<"test relation p: "<<ret.test_pRelation<<", r:"<<ret.test_rRelation<<", f1:"<<ret.test_f1Relation<<endl;


		return ret;


	}

	// Only used when current label is I or L, check state from back to front
	// in case that "O I I", etc.
	bool checkWrongState(const vector<string>& labelSequence) {
		int positionNew = -1; // the latest type-consistent B
		int positionOther = -1; // other label except type-consistent I

		const string& currentLabel = labelSequence[labelSequence.size()-1];

		for(int j=labelSequence.size()-2;j>=0;j--) {
			if(currentLabel==I_Disease || currentLabel==L_Disease) {
				if(positionNew==-1 && labelSequence[j]==B_Disease) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Disease) {
					positionOther = j;
				}
			} else if(currentLabel==I_Chemical || currentLabel==L_Chemical) {
				if(positionNew==-1 && labelSequence[j]==B_Chemical) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Chemical) {
					positionOther = j;
				}
			}

			if(positionOther!=-1 && positionNew!=-1)
				break;
		}

		if(positionNew == -1)
			return false;
		else if(positionOther<positionNew)
			return true;
		else
			return false;
	}

	void generateOneNerExample(Example& eg, const string& labelName, const fox::Sent& sent,
			const string& lastNerLabel, const int tokenIdx) {
		if(!labelName.empty()) {
			int labelID = NERlabelName2labelID(labelName);
			for(int i=0;i<MAX_ENTITY;i++)
				eg._nerLabels.push_back(0);
			eg._nerLabels[labelID] = 1;
			eg.nerGoldLabel = labelID;

		}

		for(int i=0;i<sent.tokens.size();i++) {
			eg._words.push_back(featureName2ID(m_wordAlphabet, feature_word(sent.tokens[i])));
			eg._postags.push_back(featureName2ID(m_posAlphabet, feature_pos(sent.tokens[i])));
			vector<int> chars;
			featureName2ID(m_charAlphabet, feature_character(sent.tokens[i]), chars);
			eg._seq_chars.push_back(chars);
		}

		eg._prior_ner = featureName2ID(m_nerAlphabet, lastNerLabel);
		eg._current_idx = tokenIdx;
	}

	void generateOneRelExample(Example& eg, const string& labelName, const fox::Sent& sent,
			const Entity& Disease, const Entity& Chemical, const vector<string>& labelSequence) {
		if(!labelName.empty()) {
			int labelID = RellabelName2labelID(labelName);
			for(int i=0;i<MAX_RELATION;i++)
				eg._relLabels.push_back(0);
			eg._relLabels[labelID] = 1;
			eg.relGoldLabel = labelID;

		}

		int bacTkEnd = -1;
		int locTkEnd = -1;
		const Entity& former = Disease.begin<Chemical.begin? Disease:Chemical;
		const Entity& latter = Disease.begin<Chemical.begin ? Chemical:Disease;

		for(int i=0;i<sent.tokens.size();i++) {
			const fox::Token& token = sent.tokens[i];

			eg._words.push_back(featureName2ID(m_wordAlphabet, feature_word(token)));
			eg._postags.push_back(featureName2ID(m_posAlphabet, feature_pos(token)));
			vector<int> chars;
			featureName2ID(m_charAlphabet, feature_character(sent.tokens[i]), chars);
			eg._seq_chars.push_back(chars);
			eg._deps.push_back(featureName2ID(m_depAlphabet, feature_dep(token)));
			eg._ners.push_back(featureName2ID(m_nerAlphabet, labelSequence[i]));

			if(boolTokenInEntity(token, Disease)) {
				eg._idx_e1.insert(i);

				// if like "Listeria sp.", "." is not in dependency tree
				if(bacTkEnd == -1 /*&& sent.tokens[i].depGov!=-1*/)
					bacTkEnd = i;
				else if(bacTkEnd < i /*&& sent.tokens[i].depGov!=-1*/)
					bacTkEnd = i;
			}

			if(boolTokenInEntity(token, Chemical)) {
				eg._idx_e2.insert(i);

				if(locTkEnd == -1 /*&& sent.tokens[i].depGov!=-1*/)
					locTkEnd = i;
				else if(locTkEnd < i /*&& sent.tokens[i].depGov!=-1*/)
					locTkEnd = i;
			}

			if(isTokenBetweenTwoEntities(token, former, latter)) {
				eg._between_words.push_back(i);
			}
		}

		if(eg._between_words.size()==0) {
			eg._between_words.push_back(featureName2ID(m_wordAlphabet, nullkey));
		}

		assert(bacTkEnd!=-1);
		assert(locTkEnd!=-1);


		// use SDP based on the last word of the entity
		vector<int> sdpA;
		vector<int> sdpB;
		int common = fox::Dependency::getCommonAncestor(sent.tokens, bacTkEnd, locTkEnd,
				sdpA, sdpB);


		//assert(common!=-2); // no common ancestor
		assert(common!=-1); // common ancestor is root

		if(common == -2) {
			eg._idxOnSDP_E12A.push_back(bacTkEnd);
			eg._idxOnSDP_E22A.push_back(locTkEnd);
		} else {
			for(int sdpANodeIdx=0;sdpANodeIdx<sdpA.size();sdpANodeIdx++) {
				eg._idxOnSDP_E12A.push_back(sdpA[sdpANodeIdx]-1);
			}

			for(int sdpBNodeIdx=0;sdpBNodeIdx<sdpB.size();sdpBNodeIdx++) {
				eg._idxOnSDP_E22A.push_back(sdpB[sdpBNodeIdx]-1);
			}
		}


	}

	void createAlphabet (vector<fox::Sent> & processed, Tool& tool) {

		hash_map<string, int> word_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> dep_stat;
		hash_map<string, int> char_stat;

		for(int i=0;i<processed.size();i++) {
			fox::Sent & sent = processed[i];

			for(int j=0;j<sent.tokens.size();j++) {
				string curword = feature_word(sent.tokens[j]);
				word_stat[curword]++;

				string pos = feature_pos(sent.tokens[j]);
				pos_stat[pos]++;

				vector<string> characters = feature_character(sent.tokens[j]);
				for(int i=0;i<characters.size();i++)
					char_stat[characters[i]]++;

				string dep = feature_dep(sent.tokens[j]);
				dep_stat[dep]++;
			}

		}


		stat2Alphabet(word_stat, m_wordAlphabet, "word");

		stat2Alphabet(pos_stat, m_posAlphabet, "pos");

		stat2Alphabet(dep_stat, m_depAlphabet, "dep");

		stat2Alphabet(char_stat, m_charAlphabet, "char");
	}

	string feature_word(const fox::Token& token) {
		string ret = normalize_to_lowerwithdigit(token.word);
		//string ret = normalize_to_lowerwithdigit(token.lemma);

		return ret;
	}


	string feature_pos(const fox::Token& token) {
		return token.pos;
	}

	string feature_dep(const fox::Token& token) {
		return token.depType;
	}

	vector<string> feature_character(const fox::Token& token) {
		vector<string> ret;
		string word = feature_word(token);
		for(int i=0;i<word.length();i++)
			ret.push_back(word.substr(i, 1));
		return ret;
	}

	void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet, int embSize, int seed) {
		double* emb = new double[alphabet.size()*embSize];
		fox::initArray2((double *)emb, (int)alphabet.size(), embSize, 0.0);

		vector<string> known;
		map<string, int> IDs;
		alphabet2vectormap(alphabet, known, IDs);

		fox::randomInitEmb((double*)emb, embSize, known, unknownkey,
				IDs, false, m_options.initRange, seed);

		nrmat.resize(alphabet.size(), embSize);
		array2NRMat((double*) emb, alphabet.size(), embSize, nrmat);

		delete[] emb;
	}

	template<typename xpu>
	void averageUnkownEmb(Alphabet& alphabet, LookupTable<xpu>& table, int embSize) {

		// unknown cannot be trained, use the average embedding
		int unknownID = alphabet.from_string(unknownkey);
		Tensor<cpu, 2, dtype> temp = NewTensor<cpu>(Shape2(1, embSize), d_zero);
		int number = table._nVSize-1;
		table._E[unknownID] = 0.0;
		for(int i=0;i<table._nVSize;i++) {
			if(i==unknownID)
				continue;
			table.GetEmb(i, temp);
			table._E[unknownID] += temp[0]/number;
		}

		FreeSpace(&temp);

	}

	void stat2Alphabet(hash_map<string, int>& stat, Alphabet& alphabet, const string& label) {

		cout << label<<" num: " << stat.size() << endl;
		alphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator feat_iter;
		for (feat_iter = stat.begin(); feat_iter != stat.end(); feat_iter++) {
			// if not fine tune, add all the words; if fine tune, add the words considering wordCutOff
			// in order to train unknown
			if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
			  alphabet.from_string(feat_iter->first);
			}
		}
		cout << "alphabet "<< label<<" num: " << alphabet.size() << endl;
		alphabet.set_fixed_flag(true);

	}


	void featureName2ID(Alphabet& alphabet, const string& featureName, vector<int>& vfeatureID) {
		int id = alphabet.from_string(featureName);
		if(id >=0)
			vfeatureID.push_back(id);
		else
			vfeatureID.push_back(0); // assume unknownID is zero
	}

	void featureName2ID(Alphabet& alphabet, const vector<string>& featureName, vector<int>& vfeatureID) {
		for(int i=0;i<featureName.size();i++) {
			int id = alphabet.from_string(featureName[i]);
			if(id >=0)
				vfeatureID.push_back(id);
			else
				vfeatureID.push_back(0); // assume unknownID is zero
		}
	}

	int featureName2ID(Alphabet& alphabet, const string& featureName) {
		int id = alphabet.from_string(featureName);
		if(id >=0)
			return id;
		else
			return 0; // assume unknownID is zero
	}

};



#endif /* NNBB3_H_ */

