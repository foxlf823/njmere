
#ifndef NNBB_1beam_early_H_
#define NNBB_1beam_early_H_

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
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"
#include "Word2Vec.h"
#include "utils.h"
#include "Classifier.h"
#include "Example.h"
#include "Prediction.h"
#include "beamsearch.h"


using namespace nr;
using namespace std;

// extension to NNbb_imp.h with beamsearch and earlyupdate

class NNbb_1beam_early {
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


  NNbb_1beam_early(const Options &options):m_options(options) {
		unknownkey = "-#unknown#-";
		nullkey = "-#null#-";
	}


	void trainAndTest(const string& trainFile, const string& devFile, const string& testFile,
			Tool& tool,
			const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile) {


		// load train data
		vector<Document> trainDocuments;
		parseBB3(trainFile, trainDocuments);
		loadNlpFile(trainNlpFile, trainDocuments);

		vector<Document> devDocuments;
		if(!devFile.empty()) {
			parseBB3(devFile, devDocuments);
			loadNlpFile(devNlpFile, devDocuments);
		}
		vector<Document> testDocuments;
		if(!testFile.empty()) {
			parseBB3(testFile, testDocuments);
			loadNlpFile(testNlpFile, testDocuments);
		}


		cout << "Creating Alphabet..." << endl;

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
		ner_stat[B_Bacteria]++;
		ner_stat[I_Bacteria]++;
		ner_stat[L_Bacteria]++;
		ner_stat[U_Bacteria]++;
		ner_stat[B_Habitat]++;
		ner_stat[I_Habitat]++;
		ner_stat[L_Habitat]++;
		ner_stat[U_Habitat]++;
		ner_stat[B_Geographical]++;
		ner_stat[I_Geographical]++;
		ner_stat[L_Geographical]++;
		ner_stat[U_Geographical]++;
		ner_stat[OTHER]++;
		stat2Alphabet(ner_stat, m_nerAlphabet, "ner");

		createAlphabet(trainDocuments, tool);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool);
		}


		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				randomInitNrmat(wordEmb, m_wordAlphabet, m_options.wordEmbSize, 1000);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);

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
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);

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



		static Metric eval, metric_dev;
		static vector<Example> subNerExamples;
		static vector<Example> subRelExamples;

		dtype best = 0;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    eval.reset();


			for(int docIdx=0;docIdx<trainDocuments.size();docIdx++) {
				const Document& doc = trainDocuments[docIdx];


				for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
					const fox::Sent & sent = doc.sents[sentIdx];

					Prediction gold(true);
					initialGoldPrediction(tool, doc, sent, gold);


					Prediction predict = beamSearch(sent, gold, m_options.beamSize1, 0);


					subNerExamples.clear();
/*					int min = predict.labels.size() < sent.tokens.size() ? predict.labels.size():sent.tokens.size();
					for(int tokenIdx=0;tokenIdx<min;tokenIdx++) {
						if(predict.labels[tokenIdx] != gold.labels[tokenIdx]) {
							subNerExamples.push_back(gold.examples[tokenIdx]);
						}
					}*/
					if(predict.labels.size() < sent.tokens.size()) {
						//cout<<"1111"<<endl;
						for(int tokenIdx=0;tokenIdx<predict.labels.size();tokenIdx++) {
							if(predict.labels[tokenIdx] != gold.labels[tokenIdx]) {
								subNerExamples.push_back(gold.examples[tokenIdx]);
							}
						}
					} else {
						//cout<<"hahhaha"<<endl;
						for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
							subNerExamples.push_back(gold.examples[tokenIdx]);
						}
					}


					subRelExamples.clear();
					for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
						const Relation& predictRelation = predict.relations[relationIdx];


						if(-1 == containsEntity(gold.entities, predictRelation.bacteria)) {
							Example egRel(true);
							generateOneRelExample(egRel, IMP_BAC, sent, predictRelation.bacteria, predictRelation.location, predict.labelNames);
							subRelExamples.push_back(egRel);

						} else if(-1 == containsEntity(gold.entities, predictRelation.location)) {
							Example egRel(true);
							generateOneRelExample(egRel, IMP_LOC, sent, predictRelation.bacteria, predictRelation.location, predict.labelNames);
							subRelExamples.push_back(egRel);

						} else {
							int goldRelationIdx = containsRelation(gold.relations, predictRelation);
							assert(goldRelationIdx != -1); // the two former "if" make sure this assertion
							int goldLabel = gold.labels[goldRelationIdx+sent.tokens.size()];
							int predictLabel = predict.labels[relationIdx+sent.tokens.size()];
							if(goldLabel != predictLabel)
								subRelExamples.push_back(gold.examples[goldRelationIdx+sent.tokens.size()]);
						}

					}


					if(subNerExamples.size()!=0 || subRelExamples.size()!=0) {
						int curUpdateIter = iter * trainDocuments.size() + sentIdx;

						if(subNerExamples.size()!=0) {
							dtype cost = m_classifier.processNer(subNerExamples, curUpdateIter);
							//m_classifier.checkgradsNer(subNerExamples, curUpdateIter);
						}

						if(subRelExamples.size()!=0) {
							dtype cost = m_classifier.processRel(subRelExamples, curUpdateIter);
							//m_classifier.checkgradsRel(subRelExamples, curUpdateIter);
						}

						eval.overall_label_count += m_classifier._eval.overall_label_count;
						eval.correct_label_count += m_classifier._eval.correct_label_count;


						m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
					}


				} // sent


			} // doc

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    		averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);

		    		averageUnkownEmb(m_depAlphabet, m_classifier._dep, m_options.otherEmbSize);

		    		averageUnkownEmb(m_nerAlphabet, m_classifier._ner, m_options.otherEmbSize);
		    	}

		    	evaluateOnDev(tool, devDocuments, metric_dev);

				if (metric_dev.getAccuracy() > best) {
					cout << "Exceeds best performance of " << best << endl;
					best = metric_dev.getAccuracy();

					if(testDocuments.size()>0) {

						// clear output dir
						string s = "rm -f "+m_options.output+"/*";
						system(s.c_str());

						test(tool, testDocuments);

					}
				}



		    } // devExamples > 0

		} // for iter




		m_classifier.release();

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
				//m_classifier.predictNerScore(eg, probs);

				for(int labelIdx=0;labelIdx<probs.size();labelIdx++) {
					Prediction predict(false);
					predict.labels.push_back(labelIdx);
					//predict.addScore(probs[labelIdx]);
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
					//m_classifier.predictNerScore(eg, probs);

					for(int labelIdx=0;labelIdx<probs.size();labelIdx++) {
						Prediction predict(false);

						copyPrediction(old, predict);

						predict.labels.push_back(labelIdx);
						//predict.addScore(probs[labelIdx]);
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

			if(labelName == B_Bacteria || labelName == U_Bacteria ||
					labelName == B_Habitat || labelName == U_Habitat ||
					labelName == B_Geographical || labelName == U_Geographical) {
				Entity entity;
				newEntity(token, labelName, entity, 0);
				p.entities.push_back(entity);
			} else if(labelName == I_Bacteria || labelName == L_Bacteria ||
					labelName == I_Habitat || labelName == L_Habitat ||
					labelName == I_Geographical || labelName == L_Geographical) {
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
				if((aEntity.type==bEntity.type) || (aEntity.type==TYPE_Hab && bEntity.type==TYPE_Geo) ||
						(aEntity.type==TYPE_Geo && bEntity.type==TYPE_Hab))	{
					continue;
				}

				const Entity& bacteria = aEntity.type==TYPE_Bac? aEntity:bEntity;
				const Entity& location = aEntity.type==TYPE_Bac? bEntity:aEntity;

				Example eg(true);
				generateOneRelExample(eg, "", sent, bacteria, location, p.labelNames);

				vector<dtype> probs;
				int bestLabel = m_classifier.predictRel(eg, probs);
				string predictLabelName = RellabelID2labelName(bestLabel);
				p.labelNames.push_back(predictLabelName);

				p.labels.push_back(bestLabel);
				//p.addScore(probs[bestLabel]);
				p.addLogProb(probs[bestLabel]);

				// all relation types
				Relation relation;
				newRelation(relation, bacteria, location, 0);
				p.relations.push_back(relation);



			} // aIdx

		} // bIdx

		return p;
	}

	void initialGoldPrediction(Tool& tool, const Document& doc, const fox::Sent& sent, Prediction& prediction) {
		string lastNerLabel = nullkey;

		for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
			const fox::Token& token = sent.tokens[tokenIdx];
			int entityIdx = -1;
			string schemaLabel = OTHER;

			for(int i=0;i<doc.entities.size();i++) {
				const Entity& entity = doc.entities[i];

				string temp = isTokenInEntity(token, entity);
				if(temp != OTHER) {
					entityIdx = i;
					schemaLabel = temp;
					break;
				}
			}

			Example eg(false);
			string labelName = entityIdx!=-1 ? schemaLabel+"_"+doc.entities[entityIdx].type : OTHER;
			generateOneNerExample(eg, labelName, sent, lastNerLabel, tokenIdx);

			prediction.labels.push_back(NERlabelName2labelID(labelName));
			prediction.labelNames.push_back(labelName);
			prediction.examples.push_back(eg);

			if(labelName == B_Bacteria || labelName == U_Bacteria ||
					labelName == B_Habitat || labelName == U_Habitat ||
					labelName == B_Geographical || labelName == U_Geographical) {
				Entity entity;
				newEntity(token, labelName, entity, 0);
				prediction.entities.push_back(entity);
			} else if(labelName == I_Bacteria || labelName == L_Bacteria ||
					labelName == I_Habitat || labelName == L_Habitat ||
					labelName == I_Geographical || labelName == L_Geographical) {
				if(checkWrongState(prediction.labelNames)) {
					Entity& entity = prediction.entities[prediction.entities.size()-1];
					appendEntity(token, entity);
				}
			}

			lastNerLabel = labelName;
		} // token


		// after a sentence has been tagged with ner labels, we generate relation examples
		vector<Entity> entities;
		findEntityInSent(sent.begin, sent.end, doc, entities);

		for(int bIdx=0;bIdx<entities.size();bIdx++) {
			const Entity& bEntity = entities[bIdx];

			// a is before b
			for(int aIdx=0;aIdx<bIdx;aIdx++) {
				const Entity& aEntity = entities[aIdx];

				// type constraint
				if((aEntity.type==bEntity.type) || (aEntity.type==TYPE_Hab && bEntity.type==TYPE_Geo) ||
						(aEntity.type==TYPE_Geo && bEntity.type==TYPE_Hab))	{
					continue;
				}

				const Entity& bacteria = aEntity.type==TYPE_Bac? aEntity:bEntity;
				const Entity& location = aEntity.type==TYPE_Bac? bEntity:aEntity;

				Example eg(true);
				string labelName = Not_Lives_In;
				if(isLoc(bacteria, location, doc)) {
					labelName = Lives_In;
				}
				prediction.labels.push_back(RellabelName2labelID(labelName));
				prediction.labelNames.push_back(labelName);

				generateOneRelExample(eg, labelName, sent, bacteria, location, prediction.labelNames);

				prediction.examples.push_back(eg);

				// only live_in or not_live_in
				Relation relation;
				newRelation(relation, bacteria, location, 0);
				prediction.relations.push_back(relation);


			} // aIdx

		} // bIdx



	}

	void evaluateOnDev(Tool& tool, const vector<Document>& documents, Metric& metric_dev) {
    	metric_dev.reset();

    	int ctGoldEntity = 0;
    	int ctPredictEntity = 0;
    	int ctCorrectEntity = 0;
    	int ctGoldRelation = 0, ctPredictRelation = 0, ctCorrectRelation = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];
			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;

			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];

				Prediction fakeGold(false);
				Prediction predict = beamSearch(sent, fakeGold, m_options.beamSize1, 1);

				// add live_in to answer
				for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
					const Relation& predictRelation = predict.relations[relationIdx];
					const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

					if(Lives_In == labelName) {
						anwserRelations.push_back(predictRelation);
					}
				}

				// check entity that need to be deleted
				for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
					const Relation& predictRelation = predict.relations[relationIdx];
					const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

					if(IMP_BAC == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.bacteria)) {
						// delete entity
						deleteEntity(predict.entities, predictRelation.bacteria);
					} else if(IMP_LOC == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.location)) {
						// delete entity
						deleteEntity(predict.entities, predictRelation.location);
					}

				}

				// add left entity to answer
				for(int i=0;i<predict.entities.size();i++) {
					anwserEntities.push_back(predict.entities[i]);
				}


			} // sent

			// evaluate by ourselves
			ctGoldEntity += doc.entities.size();
			ctPredictEntity += anwserEntities.size();
			for(int i=0;i<anwserEntities.size();i++) {
				for(int j=0;j<doc.entities.size();j++) {
					if(anwserEntities[i].equals(doc.entities[j])) {
						ctCorrectEntity ++;
						break;
					}
				}
			}

			ctGoldRelation += doc.relations.size();
			ctPredictRelation += anwserRelations.size();
			for(int i=0;i<anwserRelations.size();i++) {
				for(int j=0;j<doc.relations.size();j++) {
					if(anwserRelations[i].equals(doc.relations[j])) {
						ctCorrectRelation ++;
						break;
					}
				}
			}

		} // doc



		cout<<"entity p: "<<precision(ctCorrectEntity, ctPredictEntity)<<", r:"<<recall(ctCorrectEntity, ctGoldEntity)<<", f1:"<<f1(ctCorrectEntity, ctGoldEntity, ctPredictEntity)<<endl;
		cout<<"relation p: "<<precision(ctCorrectRelation, ctPredictRelation)<<", r:"<<recall(ctCorrectRelation, ctGoldRelation)<<", f1:"<<f1(ctCorrectRelation, ctGoldRelation, ctPredictRelation)<<endl;

		metric_dev.overall_label_count = ctGoldRelation;
		metric_dev.predicated_label_count = ctPredictRelation;
		metric_dev.correct_label_count = ctCorrectRelation;
		metric_dev.print();
	}

	void test(Tool& tool, const vector<Document>& documents) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];
			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;
			int entityId = doc.maxParagraphId+1;
			int relationId = 1;


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];
				int answerRelationStartId = anwserRelations.size();

				Prediction fakeGold(false);
				Prediction predict = beamSearch(sent, fakeGold, m_options.beamSize1, 2);

				// add live_in to answer
				for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
					const Relation& predictRelation = predict.relations[relationIdx];
					const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

					if(Lives_In == labelName) {
						anwserRelations.push_back(predictRelation);
						anwserRelations[anwserRelations.size()-1].setId(relationId);
						relationId++;
					}
				}

				// check entity that need to be deleted
				for(int relationIdx=0;relationIdx<predict.relations.size();relationIdx++) {
					const Relation& predictRelation = predict.relations[relationIdx];
					const string& labelName = predict.labelNames[relationIdx+sent.tokens.size()];

					if(IMP_BAC == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.bacteria)) {
						// delete entity
						deleteEntity(predict.entities, predictRelation.bacteria);
					} else if(IMP_LOC == labelName && -1 == relationContainsEntity(anwserRelations, predictRelation.location)) {
						// delete entity
						deleteEntity(predict.entities, predictRelation.location);
					}

				}

				// add left entity to answer
				for(int i=0;i<predict.entities.size();i++) {
					anwserEntities.push_back(predict.entities[i]);
					anwserEntities[anwserEntities.size()-1].setId(entityId);
					entityId++;
				}

				// link entity in relation with entity
				for(int i=answerRelationStartId;i<anwserRelations.size();i++) {
					int bacIdx = containsEntity(anwserEntities, anwserRelations[i].bacteria);
					assert(bacIdx!=-1);
					anwserRelations[i].setBacId(anwserEntities[bacIdx].id);

					int locIdx = containsEntity(anwserEntities, anwserRelations[i].location);
					assert(locIdx!=-1);
					anwserRelations[i].setLocId(anwserEntities[locIdx].id);

				}

			} // sent

			outputResults(doc.id, anwserEntities, anwserRelations, m_options.output);
		} // doc

	}

	// Only used when current label is I or L, check state from back to front
	// in case that "O I I", etc.
	bool checkWrongState(const vector<string>& labelSequence) {
		int positionNew = -1; // the latest type-consistent B
		int positionOther = -1; // other label except type-consistent I

		const string& currentLabel = labelSequence[labelSequence.size()-1];

		for(int j=labelSequence.size()-2;j>=0;j--) {
			if(currentLabel==I_Bacteria || currentLabel==L_Bacteria) {
				if(positionNew==-1 && labelSequence[j]==B_Bacteria) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Bacteria) {
					positionOther = j;
				}
			} else if(currentLabel==I_Habitat || currentLabel==L_Habitat) {
				if(positionNew==-1 && labelSequence[j]==B_Habitat) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Habitat) {
					positionOther = j;
				}
			} else {
				if(positionNew==-1 && labelSequence[j]==B_Geographical) {
					positionNew = j;
				} else if(positionOther==-1 && labelSequence[j]!=I_Geographical) {
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
			const Entity& bacteria, const Entity& location, const vector<string>& labelSequence) {
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

			if(boolTokenInEntity(token, bacteria)) {
				eg._idx_e1.insert(i);

				// if like "Listeria sp.", "." is not in dependency tree
				if(bacTkEnd == -1 /*&& sent.tokens[i].depGov!=-1*/)
					bacTkEnd = i;
				else if(bacTkEnd < i /*&& sent.tokens[i].depGov!=-1*/)
					bacTkEnd = i;
			} else if(boolTokenInEntity(token, location)) {
				eg._idx_e2.insert(i);

				if(locTkEnd == -1 /*&& sent.tokens[i].depGov!=-1*/)
					locTkEnd = i;
				else if(locTkEnd < i /*&& sent.tokens[i].depGov!=-1*/)
					locTkEnd = i;
			}
		}
/*		if(bacTkEnd==-1||locTkEnd==-1)
			cout<<endl;*/
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

	void createAlphabet (const vector<Document>& documents, Tool& tool) {

		hash_map<string, int> word_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> dep_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<documents[docIdx].sents.size();i++) {

				for(int j=0;j<documents[docIdx].sents[i].tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sents[i].tokens[j]);
					word_stat[curword]++;

					string pos = feature_pos(documents[docIdx].sents[i].tokens[j]);
					pos_stat[pos]++;

					string dep = feature_dep(documents[docIdx].sents[i].tokens[j]);
					dep_stat[dep]++;
				}


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

