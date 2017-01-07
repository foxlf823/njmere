
#ifndef NNBB_NER_H_
#define NNBB_NER_H_

#include <iosfwd>
#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "N3Lhelper.h"
#include "Utf.h"
#include "Token.h"
#include "Sent.h"
#include <sstream>

#include "N3L.h"
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"
#include "Word2Vec.h"
#include "utils.h"
#include "ClassifierNer.h"
#include "NerExample.h"


using namespace nr;
using namespace std;

// only perform named entity recognition

class NNbb_ner {
public:
	Options m_options;

	Alphabet m_wordAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_nerAlphabet;

	string unknownkey;
	string nullkey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  ClassifierNer<cpu> m_classifier;
#endif


  NNbb_ner(const Options &options):m_options(options) {
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

		vector<NerExample> trainExamples;
		initialTrainingExamples(tool, trainDocuments, trainExamples);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;

		  m_classifier.init(m_options);

		  m_classifier._words.initial(wordEmb);
		  m_classifier._words.setEmbFineTune(m_options.wordEmbFineTune);

		m_classifier._pos.initial(posEmb);
		m_classifier._pos.setEmbFineTune(true);

		m_classifier._ner.initial(nerEmb);
		m_classifier._ner.setEmbFineTune(true);



		static Metric eval, metric_dev;
		static vector<NerExample> subExamples;

		dtype best = 0;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

			cout << "##### Iteration " << iter << std::endl;

		    eval.reset();

		    // this coding block shold be identical to initialTrainingExamples
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

						dtype cost = m_classifier.process(subExamples, curUpdateIter);

						eval.overall_label_count += m_classifier._eval.overall_label_count;
						eval.correct_label_count += m_classifier._eval.correct_label_count;


						//m_classifier.checkgrads(subExamples, curUpdateIter);
/*						  if (m_options.verboseIter>0 && (curUpdateIter + 1) % m_options.verboseIter == 0) {
							std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
							std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
						  }*/
						m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);


						exampleIdx++;
					} // token


				} // sent


			} // doc

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    		averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);

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


	void initialTrainingExamples(Tool& tool, const vector<Document>& documents, vector<NerExample>& examples) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];
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

					NerExample eg;
					string labelName = entityIdx!=-1 ? schemaLabel+"_"+doc.entities[entityIdx].type : OTHER;
					generateOneNerExample(eg, labelName, sent, lastNerLabel, tokenIdx);

					examples.push_back(eg);

					lastNerLabel = labelName;
				} // token


			} // sent


		} // doc

	}

	void evaluateOnDev(Tool& tool, const vector<Document>& documents, Metric& metric_dev) {
    	metric_dev.reset();

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];


			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];
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

					NerExample eg;
					string labelName = entityIdx!=-1 ? schemaLabel+"_"+doc.entities[entityIdx].type : OTHER;
					generateOneNerExample(eg, labelName, sent, lastNerLabel, tokenIdx);

					int predicted = m_classifier.predict(eg);

					metric_dev.overall_label_count += 1;
					if(predicted == eg.goldLabel) {
						metric_dev.correct_label_count++;
					} /*else {
						cout<<eg.goldLabel<<" "<<predicted<<endl;
					}*/

					lastNerLabel = labelName;
				} // token


			} // sent


		} // doc

		metric_dev.print();
	}

	void test(Tool& tool, const vector<Document>& documents) {

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			const Document& doc = documents[docIdx];
			vector<Entity> entities;
			int entityId = doc.maxParagraphId+1;

			for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
				const fox::Sent & sent = doc.sents[sentIdx];
				string lastNerLabel = nullkey;
				vector<string> labelSequence;

				for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
					const fox::Token& token = sent.tokens[tokenIdx];

					NerExample eg;
					string fakeLabelName = "";
					generateOneNerExample(eg, fakeLabelName, sent, lastNerLabel, tokenIdx);

					int predicted = m_classifier.predict(eg);

					string labelName = NERlabelID2labelName(predicted);
					labelSequence.push_back(labelName);

					// decode entity label
					if(labelName == B_Bacteria || labelName == U_Bacteria ||
							labelName == B_Habitat || labelName == U_Habitat ||
							labelName == B_Geographical || labelName == U_Geographical) {
						Entity entity;
						newEntity(token, labelName, entity, entityId);
						entities.push_back(entity);
						entityId++;
					} else if(labelName == I_Bacteria || labelName == L_Bacteria ||
							labelName == I_Habitat || labelName == L_Habitat ||
							labelName == I_Geographical || labelName == L_Geographical) {
						if(checkWrongState(labelSequence)) {
							Entity& entity = entities[entities.size()-1];
							appendEntity(token, entity);
						}
					}

/*					else if((lastNerLabel == B_Bacteria|| lastNerLabel == I_Bacteria) &&
							(labelName == I_Bacteria || labelName == L_Bacteria)) {
						Entity& entity = entities[entities.size()-1];
						appendEntity(token, entity);
					} else if((lastNerLabel == B_Habitat || lastNerLabel == I_Habitat) &&
							(labelName == I_Habitat || labelName == L_Habitat)) {
						Entity& entity = entities[entities.size()-1];
						appendEntity(token, entity);
					} else if((lastNerLabel == B_Geographical || lastNerLabel == I_Geographical) &&
							(labelName == I_Geographical || labelName == L_Geographical)) {
						Entity& entity = entities[entities.size()-1];
						appendEntity(token, entity);
					} else if(lastNerLabel == OTHER &&
							(labelName == I_Bacteria||labelName == I_Habitat||labelName == I_Geographical)) {
						cout<<"warning O I: "<<doc.id<<" "<<token.word<<" "<<token.begin<<endl;
					}*/



					lastNerLabel = labelName;
				} // token


			} // sent

			outputEnityResults(doc.id, entities, m_options.output);
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

	void generateOneNerExample(NerExample& eg, const string& labelName, const fox::Sent& sent,
			const string& lastNerLabel, const int tokenIdx) {
		if(!labelName.empty()) {
			int labelID = NERlabelName2labelID(labelName);
			for(int i=0;i<MAX_ENTITY;i++)
				eg.m_labels.push_back(0);
			eg.m_labels[labelID] = 1;
			eg.goldLabel = labelID;

		}

		for(int i=0;i<sent.tokens.size();i++) {
			eg._words.push_back(featureName2ID(m_wordAlphabet, feature_word(sent.tokens[i])));
			eg._postags.push_back(featureName2ID(m_posAlphabet, feature_pos(sent.tokens[i])));
		}

		eg._prior_ner = featureName2ID(m_nerAlphabet, lastNerLabel);
		eg._current_idx = tokenIdx;
	}

	void createAlphabet (const vector<Document>& documents, Tool& tool) {

		hash_map<string, int> word_stat;
		hash_map<string, int> pos_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<documents[docIdx].sents.size();i++) {

				for(int j=0;j<documents[docIdx].sents[i].tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sents[i].tokens[j]);
					word_stat[curword]++;

					string pos = feature_pos(documents[docIdx].sents[i].tokens[j]);
					pos_stat[pos]++;


				}


			}


		}

		stat2Alphabet(word_stat, m_wordAlphabet, "word");

		stat2Alphabet(pos_stat, m_posAlphabet, "pos");


	}

	string feature_word(const fox::Token& token) {
		//string ret = normalize_to_lowerwithdigit(token.word);
		string ret = normalize_to_lowerwithdigit(token.lemma);

		return ret;
	}


	string feature_pos(const fox::Token& token) {
		return token.pos;
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

