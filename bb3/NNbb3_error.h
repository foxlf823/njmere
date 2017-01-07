
#ifndef NNBB3_error_H_
#define NNBB3_error_H_

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
#include "Classifier3.h"
#include "Example.h"




using namespace nr;
using namespace std;

// error analysis on development set

class NNbb3_error {
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


  NNbb3_error(const Options &options):m_options(options) {
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

		m_charAlphabet.clear();
		m_charAlphabet.from_string(unknownkey);
		m_charAlphabet.from_string(nullkey);

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


		NRMat<dtype> posEmb;
		randomInitNrmat(posEmb, m_posAlphabet, m_options.otherEmbSize, 1010);
		NRMat<dtype> nerEmb;
		randomInitNrmat(nerEmb, m_nerAlphabet, m_options.otherEmbSize, 1020);
		NRMat<dtype> depEmb;
		randomInitNrmat(depEmb, m_depAlphabet, m_options.otherEmbSize, 1030);
		NRMat<dtype> charEmb;
		randomInitNrmat(charEmb, m_charAlphabet, m_options.otherEmbSize, 1040);

		vector<Example> trainExamples;
		initialTrainingExamples(tool, trainDocuments, trainExamples);
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



		static Metric eval, metric_dev;
		static vector<Example> subExamples;

		dtype best = 0;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    eval.reset();

		    // this coding block should be identical to initialTrainingExamples
		    // we don't regenerate training examples, instead fetch them directly
		    int exampleIdx = 0;
			for(int docIdx=0;docIdx<trainDocuments.size();docIdx++) {
				const Document& doc = trainDocuments[docIdx];


				for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
					const fox::Sent & sent = doc.sents[sentIdx];

					for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
						const fox::Token& token = sent.tokens[tokenIdx];

						subExamples.clear();
						subExamples.push_back(trainExamples[exampleIdx]);
						int curUpdateIter = iter * trainExamples.size() + exampleIdx;

						dtype cost = m_classifier.processNer(subExamples, curUpdateIter);

						eval.overall_label_count += m_classifier._eval.overall_label_count;
						eval.correct_label_count += m_classifier._eval.correct_label_count;


						//m_classifier.checkgradsNer(subExamples, curUpdateIter);
/*						  if (m_options.verboseIter>0 && (curUpdateIter + 1) % m_options.verboseIter == 0) {
							std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
							std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
						  }*/
						m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);


						exampleIdx++;
					} // token

					// begin to use relation
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

							subExamples.clear();
							subExamples.push_back(trainExamples[exampleIdx]);
							int curUpdateIter = iter * trainExamples.size() + exampleIdx;

							dtype cost = m_classifier.processRel(subExamples, curUpdateIter);

							eval.overall_label_count += m_classifier._eval.overall_label_count;
							eval.correct_label_count += m_classifier._eval.correct_label_count;


							//m_classifier.checkgradsRel(subExamples, curUpdateIter);
/*							  if (m_options.verboseIter>0 && (curUpdateIter + 1) % m_options.verboseIter == 0) {
								std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
								std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
							  }*/
							m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);


							exampleIdx++;

						} // aIdx

					} // bIdx


				} // sent


			} // doc

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordEmbFineTune && m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    		averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);

		    		averageUnkownEmb(m_depAlphabet, m_classifier._dep, m_options.otherEmbSize);

		    		averageUnkownEmb(m_nerAlphabet, m_classifier._ner, m_options.otherEmbSize);

		    		averageUnkownEmb(m_charAlphabet, m_classifier._char, m_options.otherEmbSize);
		    	}

		    	if(iter == m_options.maxIter-1) {
		    		evaluateOnDev(tool, devDocuments, metric_dev, iter);
		    	}

/*				if (metric_dev.getAccuracy() > best) {
					cout << "Exceeds best performance of " << best << endl;
					best = metric_dev.getAccuracy();

					if(testDocuments.size()>0) {

						// clear output dir
						string s = "rm -f "+m_options.output+"/*";
						system(s.c_str());

						test(tool, testDocuments);

					}
				}*/



		    } // devExamples > 0

		} // for iter




		m_classifier.release();

	}


	void initialTrainingExamples(Tool& tool, const vector<Document>& documents, vector<Example>& examples) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];
				string lastNerLabel = nullkey;
				vector<string> labelSequence;

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
					labelSequence.push_back(labelName);

					examples.push_back(eg);

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
						labelSequence.push_back(labelName);

						generateOneRelExample(eg, labelName, sent, bacteria, location, labelSequence);

						examples.push_back(eg);

					} // aIdx

				} // bIdx


			} // sent


		} // doc

	}

	void evaluateOnDev(Tool& tool, const vector<Document>& documents, Metric& metric_dev, int iter) {
    	metric_dev.reset();


    	int ct_fp1Entities = 0;
    	int ct_fp2Entities = 0;
    	int ct_fn1Entities = 0;
    	int ct_fn2Entities = 0;
    	int ct_fp1Relations = 0;
    	int ct_fp2Relations = 0;
    	int ct_fn1Relations = 0;
    	int ct_fn2Relations = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];
			vector<Entity> anwserEntities;
			vector<Relation> anwserRelations;

	    	vector<Entity> fp1Entities;
	    	vector<Entity> fp2Entities;
	    	vector<Entity> fn1Entities;
	    	vector<Entity> fn2Entities;
	    	vector<Relation> fp1Relations;
	    	vector<Relation> fp2Relations;
	    	vector<Relation> fn1Relations;
	    	vector<Relation> fn2Relations;


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];
				string lastNerLabel = nullkey;
				vector<string> labelSequence;

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
					if(labelName == B_Bacteria || labelName == U_Bacteria ||
							labelName == B_Habitat || labelName == U_Habitat ||
							labelName == B_Geographical || labelName == U_Geographical) {
						Entity entity;
						newEntity(token, labelName, entity, 0);
						anwserEntities.push_back(entity);
					} else if(labelName == I_Bacteria || labelName == L_Bacteria ||
							labelName == I_Habitat || labelName == L_Habitat ||
							labelName == I_Geographical || labelName == L_Geographical) {
						if(checkWrongState(labelSequence)) {
							Entity& entity = anwserEntities[anwserEntities.size()-1];
							appendEntity(token, entity);
						}
					}



					lastNerLabel = labelName;
				} // token

				// begin to relation
				vector<Entity> entities;
				findEntityInSent(sent.begin, sent.end, anwserEntities, entities);

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
						string fakeLabelName = "";
						generateOneRelExample(eg, fakeLabelName, sent, bacteria, location, labelSequence);

						vector<dtype> probs;
						int predicted = m_classifier.predictRel(eg, probs);

						string labelName = RellabelID2labelName(predicted);
						labelSequence.push_back(labelName);

						// decode relation label
						if(labelName == Lives_In) {
							Relation relation;
							newRelation(relation, bacteria, location, 0);
							anwserRelations.push_back(relation);
						}

					} // aIdx

				} // bIdx

			} // sent

			// evaluate by ourselves
			for(int i=0;i<anwserEntities.size();i++) {
				int j=0;
				for(;j<doc.entities.size();j++) {
					if(anwserEntities[i].equalsBoundary(doc.entities[j])) {
						break;
					}
				}

				if(j>=doc.entities.size()) { // boundary not match
					fp1Entities.push_back(anwserEntities[i]);
				} else  {
					int k=0;
					for(;k<doc.entities.size();k++)  {
						if(anwserEntities[i].equals(doc.entities[k])) {
							break;
						}
					}

					if(k>=doc.entities.size()) {
						fp2Entities.push_back(anwserEntities[i]); // type not match
					} else {
						// TP
					}

				}
			}

			for(int i=0;i<doc.entities.size();i++) {
				int j=0;
				for(;j<anwserEntities.size();j++) {
					if(doc.entities[i].equalsBoundary(anwserEntities[j])) {
						break;
					}

				}

				if(j>=anwserEntities.size()) { // boundary not match
					fn1Entities.push_back(doc.entities[i]);
				} else {
					int k=0;
					for(;k<anwserEntities.size();k++)  {
						if(doc.entities[i].equals(anwserEntities[k])) {
							break;
						}
					}

					if(k>=anwserEntities.size()) {
						fn2Entities.push_back(doc.entities[i]); // type not match
					} else {
						// TP
					}

				}

			}


			for(int i=0;i<anwserRelations.size();i++) {
				// try to match gold entities
				bool entity1Matched = false;
				bool entity2Matched = false;

				for(int k=0;k<doc.entities.size();k++) {
					if(entity1Matched==false && anwserRelations[i].bacteria.equals(doc.entities[k])) {
						entity1Matched = true;
					}
					if(entity2Matched==false && anwserRelations[i].location.equals(doc.entities[k])) {
						entity2Matched = true;
					}

					if(entity1Matched && entity2Matched)
						break;
				}

				if(!entity1Matched || !entity2Matched) {
					fp1Relations.push_back(anwserRelations[i]);
				} else {
					// try to match gold relations

					int j=0;
					for(;j<doc.relations.size();j++) {
						if(anwserRelations[i].equals(doc.relations[j])) {
							break;
						}
					}

					if(j>=doc.relations.size()) {
						fp2Relations.push_back(anwserRelations[i]);
					} else {
						// TP
					}

				}
			}

			for(int i=0;i<doc.relations.size();i++) {
				// try to match predicted entities
				bool entity1Matched = false;
				bool entity2Matched = false;

				for(int k=0;k<anwserEntities.size();k++) {
					if(entity1Matched==false && doc.relations[i].bacteria.equals(anwserEntities[k])) {
						entity1Matched = true;
					}
					if(entity2Matched==false && doc.relations[i].location.equals(anwserEntities[k])) {
						entity2Matched = true;
					}

					if(entity1Matched && entity2Matched)
						break;
				}

				if(!entity1Matched || !entity2Matched) {
					fn1Relations.push_back(doc.relations[i]);
				} else {
					// try to match predicted relations
					int j=0;
					for(;j<anwserRelations.size();j++) {
						if(doc.relations[i].equals(anwserRelations[j])) {
							break;
						}
					}

					if(j>=anwserRelations.size()) {
						fn2Relations.push_back(doc.relations[i]);
					} else {
						// TP
					}

				}
			}


			if(iter == m_options.maxIter-1) {
		    	ct_fp1Entities += fp1Entities.size();
		    	ct_fp2Entities += fp2Entities.size();
		    	ct_fn1Entities += fn1Entities.size();
		    	ct_fn2Entities += fn2Entities.size();
		    	ct_fp1Relations += fp1Relations.size();
		    	ct_fp2Relations += fp2Relations.size();
		    	ct_fn1Relations += fn1Relations.size();
		    	ct_fn2Relations += fn2Relations.size();


			}

		} // doc



		if(iter == m_options.maxIter-1) {
			cout<<"### error analysis begin ###"<<endl;
			double ctTotalError = ct_fp1Entities+ct_fp2Entities+ct_fn1Entities+ct_fn2Entities;
			cout<<"fp boundary not match "<<ct_fp1Entities/ctTotalError<<endl;
			cout<<"fp type not match "<<ct_fp2Entities/ctTotalError<<endl;

			cout<<"fn boundary not match "<<ct_fn1Entities/ctTotalError<<endl;

			cout<<"fn type not match "<<ct_fn2Entities/ctTotalError<<endl;


			ctTotalError = ct_fp1Relations+ct_fp2Relations+ct_fn1Relations+ct_fn2Relations;
			cout<<"fp entity wrong "<<ct_fp1Relations/ctTotalError<<endl;

			cout<<"fp entity correct but relation wrong "<<ct_fp2Relations/ctTotalError<<endl;

			cout<<"fn entity not found "<<ct_fn1Relations/ctTotalError<<endl;

			cout<<"fn entity found but relation lost "<<ct_fn2Relations/ctTotalError<<endl;


			cout<<"### error analysis end ###"<<endl;
		}
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
			vector<int> chars;
			featureName2ID(m_charAlphabet, feature_character(sent.tokens[i]), chars);
			eg._seq_chars.push_back(chars);
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
		const Entity& former = bacteria.begin<location.begin? bacteria:location;
		const Entity& latter = bacteria.begin<location.begin ? location:bacteria;

		for(int i=0;i<sent.tokens.size();i++) {
			const fox::Token& token = sent.tokens[i];

			eg._words.push_back(featureName2ID(m_wordAlphabet, feature_word(token)));
			eg._postags.push_back(featureName2ID(m_posAlphabet, feature_pos(token)));
			vector<int> chars;
			featureName2ID(m_charAlphabet, feature_character(sent.tokens[i]), chars);
			eg._seq_chars.push_back(chars);
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

	void createAlphabet (const vector<Document>& documents, Tool& tool) {

		hash_map<string, int> word_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> dep_stat;
		hash_map<string, int> char_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<documents[docIdx].sents.size();i++) {

				for(int j=0;j<documents[docIdx].sents[i].tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sents[i].tokens[j]);
					word_stat[curword]++;

					string pos = feature_pos(documents[docIdx].sents[i].tokens[j]);
					pos_stat[pos]++;

					vector<string> characters = feature_character(documents[docIdx].sents[i].tokens[j]);
					for(int i=0;i<characters.size();i++)
						char_stat[characters[i]]++;

					string dep = feature_dep(documents[docIdx].sents[i].tokens[j]);
					dep_stat[dep]++;
				}


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

