
#ifndef NNADE_IMP_BEAM_H_
#define NNADE_IMP_BEAM_H_

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
#include "Classifier.h"
#include "Example.h"
#include "BestPerformance.h"
#include "Prediction.h"
#include "beamsearch.h"


using namespace nr;
using namespace std;

// a implement of ACL 2016 end-to-end relation extraction
// use relation f1 on the development set

class NNade_imp_beam {
public:
	Options m_options;

	Alphabet m_wordAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_nerAlphabet;
	Alphabet m_depAlphabet;

	string unknownkey;
	string nullkey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  Classifier<cpu> m_classifier;
#endif


  NNade_imp_beam(const Options &options):m_options(options) {
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

		  m_classifier.init(m_options);

		  m_classifier._words.initial(wordEmb);
		  m_classifier._words.setEmbFineTune(m_options.wordEmbFineTune);

		m_classifier._pos.initial(posEmb);
		m_classifier._pos.setEmbFineTune(true);

		m_classifier._dep.initial(depEmb);
		m_classifier._dep.setEmbFineTune(true);

		m_classifier._ner.initial(nerEmb);
		m_classifier._ner.setEmbFineTune(true);


		static vector<Example> subNerExamples;
		static vector<Example> subRelExamples;

		dtype best = 0;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

			for(int sentIdx=0;sentIdx<processedTrain.size();sentIdx++) {
				const fox::Sent & sent = processedTrain[sentIdx];
				const ADEsentence & annotatedSent = annotatedTrain[sentIdx];

				Prediction gold(true);
				initialGoldPrediction(annotatedSent, sent, gold);

				Prediction predict = beamSearch(sent, gold, m_options.beamSize1, 0);

				subNerExamples.clear();
				int min = predict.labels.size() < sent.tokens.size() ? predict.labels.size() : sent.tokens.size();
				for(int tokenIdx=0;tokenIdx<min;tokenIdx++) {
					if(predict.labels[tokenIdx] != gold.labels[tokenIdx]) {
						subNerExamples.push_back(gold.examples[tokenIdx]);
					}
				}

				subRelExamples.clear();
				for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
					const Relation& predictRelation = predict.relations[relationIdx];

					if(-1 == containsEntity(gold.entities, predictRelation.entity1)) {
						Example egRel(true);
						generateOneRelExample(egRel, IMP_Disease, sent, predictRelation.entity1, predictRelation.entity2, predict.labelNames);
						subRelExamples.push_back(egRel);

					} else if(-1 == containsEntity(gold.entities, predictRelation.entity2)) {
						Example egRel(true);
						generateOneRelExample(egRel, IMP_Chemical, sent, predictRelation.entity1, predictRelation.entity2, predict.labelNames);
						subRelExamples.push_back(egRel);

					} else {
						int goldRelationIdx = containsRelation(gold.relations, predictRelation);
						assert(goldRelationIdx != -1);
						int goldLabel = gold.labels[goldRelationIdx+sent.tokens.size()];
						int predictLabel = predict.labels[relationIdx+sent.tokens.size()];
						if(goldLabel != predictLabel)
							subRelExamples.push_back(gold.examples[goldRelationIdx+sent.tokens.size()]);
					}

				}

				if(subNerExamples.size()!=0 || subRelExamples.size()!=0) {
					int curUpdateIter = iter + sentIdx;

					if(subNerExamples.size()!=0) {
						dtype cost = m_classifier.processNer(subNerExamples, curUpdateIter);
					}

					if(subRelExamples.size()!=0) {
						dtype cost = m_classifier.processRel(subRelExamples, curUpdateIter);
					}

					m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
				}


			} // sent



		    // an iteration end, begin to evaluate
		    if((iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    		averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);

		    		averageUnkownEmb(m_depAlphabet, m_classifier._dep, m_options.otherEmbSize);

		    		averageUnkownEmb(m_nerAlphabet, m_classifier._ner, m_options.otherEmbSize);
		    	}

		    	BestPerformance currentDev = evaluateOnDev(tool, annotatedDev, processedDev);

				if (currentDev.dev_f1Relation > best) {
					cout << "Exceeds best performance of " << best << endl;
					best = currentDev.dev_f1Relation;

					BestPerformance currentTest = test(tool, annotatedTest, processedTest);

					// record the best performance for this group
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

	void initialGoldPrediction(const ADEsentence& annotatedSent, const fox::Sent& sent, Prediction& prediction) {
		string lastNerLabel = nullkey;

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

			prediction.labels.push_back(NERlabelName2labelID(labelName));
			prediction.labelNames.push_back(labelName);
			prediction.examples.push_back(eg);

			if(labelName == B_Disease || labelName == U_Disease ||
					labelName == B_Chemical || labelName == U_Chemical ) {
				Entity entity;
				newEntity(token, labelName, entity);
				prediction.entities.push_back(entity);
			} else if(labelName == I_Disease || labelName == L_Disease ||
					labelName == I_Chemical || labelName == L_Chemical ) {
				if(checkWrongState(prediction.labelNames)) {
					Entity& entity = prediction.entities[prediction.entities.size()-1];
					appendEntity(token, entity);
				}
			}

			lastNerLabel = labelName;
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

				const Entity& Disease = aEntity.type==TYPE_Disease? aEntity:bEntity;
				const Entity& Chemical = aEntity.type==TYPE_Disease? bEntity:aEntity;

				Example eg(true);
				string labelName = Not_ADE;
				if(isADE(Disease, Chemical, annotatedSent)) {
					labelName = ADE;
				}
				prediction.labels.push_back(RellabelName2labelID(labelName));
				prediction.labelNames.push_back(labelName);

				generateOneRelExample(eg, labelName, sent, Disease, Chemical, prediction.labelNames);

				prediction.examples.push_back(eg);

				// only ADE or not_ADE
				Relation relation;
				newRelation(relation, Disease, Chemical);
				prediction.relations.push_back(relation);


			} // aIdx

		} // bIdx



	}

	// model: 0-train 1-dev 2-test
	Prediction beamSearch(const fox::Sent & sent, const Prediction& gold,
			const int beamSize1, int model) {
		// entity
		vector<Prediction> beam;
		list<Prediction> buffer;

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			const fox::Token& token = sent.tokens[tokenIdx];

			if(tokenIdx==0) {
				Example eg(false);
				generateOneNerExample(eg, "", sent, nullkey, tokenIdx);

				vector<dtype> probs;
				m_classifier.predictNer(eg, probs);

				for(int labelIdx=0;labelIdx<probs.size();labelIdx++) {
					Prediction predict(false);
					predict.labels.push_back(labelIdx);
					predict.addLogProb(probs[labelIdx]);

					addToBuffer(buffer, predict);
				}
			} else {
				for(int beamIdx=0;beamIdx<beam.size();beamIdx++) {
					const Prediction& old = beam[beamIdx];
					string lastNerLabel = NERlabelID2labelName(old.labels[old.labels.size()-1]);

					Example eg(false);
					generateOneNerExample(eg, "", sent, lastNerLabel, tokenIdx);

					vector<dtype> probs;
					m_classifier.predictNer(eg, probs);

					for(int labelIdx=0;labelIdx<probs.size();labelIdx++) {
						Prediction predict(false);

						copyPrediction(old, predict);

						predict.labels.push_back(labelIdx);
						predict.addLogProb(probs[labelIdx]);

						addToBuffer(buffer, predict);
					}
				}
			}

			beam.clear();
			Kbest(beam, buffer, beamSize1);
			buffer.clear();

			if(model==0 && earlyUpdate(beam, gold)) {
				return beam[0];
			}


		} // token

		// relation
		buffer.clear();

		Prediction& p = beam[0];

		// generate entities based on the token label sequence of beam[0]
		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			const fox::Token& token = sent.tokens[tokenIdx];
			string labelName = NERlabelID2labelName(p.labels[tokenIdx]);

			p.labelNames.push_back(labelName);

			if(labelName == B_Disease || labelName == U_Disease ||
					labelName == B_Chemical || labelName == U_Chemical ) {
				Entity entity;
				newEntity(token, labelName, entity);
				p.entities.push_back(entity);
			} else if(labelName == I_Disease || labelName == L_Disease ||
					labelName == I_Chemical || labelName == L_Chemical ) {
				if(checkWrongState(p.labelNames)) {
					Entity& entity = p.entities[p.entities.size()-1];
					appendEntity(token, entity);
				}
			}
		}


		for(int bIdx=0;bIdx<p.entities.size();bIdx++) {
			const Entity& bEntity = p.entities[bIdx];

			// a is before b
			for(int aIdx=0;aIdx<bIdx;aIdx++) {
				const Entity& aEntity = p.entities[aIdx];

				// type constraint
				if(aEntity.type==bEntity.type)	{
					continue;
				}

				const Entity& Disease = aEntity.type==TYPE_Disease? aEntity:bEntity;
				const Entity& Chemical = aEntity.type==TYPE_Disease? bEntity:aEntity;

				Example eg(true);
				generateOneRelExample(eg, "", sent, Disease, Chemical, p.labelNames);

				vector<dtype> probs;
				int bestLabel = m_classifier.predictRel(eg, probs);
				string predictLabelName = RellabelID2labelName(bestLabel);
				p.labelNames.push_back(predictLabelName);

				p.labels.push_back(bestLabel);
				p.addLogProb(probs[bestLabel]);

				// all relation types
				Relation relation;
				newRelation(relation, Disease, Chemical);
				p.relations.push_back(relation);



			} // aIdx

		} // bIdx

		return p;
	}


	BestPerformance evaluateOnDev(Tool& tool, vector<ADEsentence> & annotated, vector<fox::Sent> & processed) {

		BestPerformance ret;

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;
    	int ctGoldRelation = 0, ctPredictRelation = 0, ctCorrectRelation = 0;


		for(int sentIdx=0;sentIdx<processed.size();sentIdx++) {
			const fox::Sent & sent = processed[sentIdx];
			const ADEsentence & annotatedSent =annotated[sentIdx];

			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;

			Prediction fakeGold(false);
			Prediction predict = beamSearch(sent, fakeGold, m_options.beamSize1, 1);

			// add live_in to answer
			for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
				const Relation& predictRelation = predict.relations[relationIdx];
				const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

				if(ADE == labelName) {
					anwserRelations.push_back(predictRelation);
				}
			}

			// check entity that need to be deleted
			for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
				const Relation& predictRelation = predict.relations[relationIdx];
				const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

				if(IMP_Disease == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.entity1)) {
					// delete entity
					deleteEntity(predict.entities, predictRelation.entity1);
				} else if(IMP_Chemical == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.entity2)) {
					// delete entity
					deleteEntity(predict.entities, predictRelation.entity2);
				}

			}

			// add left entity to answer
			for(int i=0;i<predict.entities.size();i++) {
				anwserEntities.push_back(predict.entities[i]);
			}


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

			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;

			Prediction fakeGold(false);
			Prediction predict = beamSearch(sent, fakeGold, m_options.beamSize1, 2);

			// add live_in to answer
			for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
				const Relation& predictRelation = predict.relations[relationIdx];
				const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

				if(ADE == labelName) {
					anwserRelations.push_back(predictRelation);
				}
			}

			// check entity that need to be deleted
			for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
				const Relation& predictRelation = predict.relations[relationIdx];
				const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

				if(IMP_Disease == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.entity1)) {
					// delete entity
					deleteEntity(predict.entities, predictRelation.entity1);
				} else if(IMP_Chemical == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.entity2)) {
					// delete entity
					deleteEntity(predict.entities, predictRelation.entity2);
				}

			}

			// add left entity to answer
			for(int i=0;i<predict.entities.size();i++) {
				anwserEntities.push_back(predict.entities[i]);
			}

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

		for(int i=0;i<sent.tokens.size();i++) {
			const fox::Token& token = sent.tokens[i];

			eg._words.push_back(featureName2ID(m_wordAlphabet, feature_word(token)));
			eg._postags.push_back(featureName2ID(m_posAlphabet, feature_pos(token)));
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

		for(int i=0;i<processed.size();i++) {
			fox::Sent & sent = processed[i];

			for(int j=0;j<sent.tokens.size();j++) {
				string curword = feature_word(sent.tokens[j]);
				word_stat[curword]++;

				string pos = feature_pos(sent.tokens[j]);
				pos_stat[pos]++;

				string dep = feature_dep(sent.tokens[j]);
				dep_stat[dep]++;
			}

		}


		stat2Alphabet(word_stat, m_wordAlphabet, "word");

		stat2Alphabet(pos_stat, m_posAlphabet, "pos");

		stat2Alphabet(dep_stat, m_depAlphabet, "dep");
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

