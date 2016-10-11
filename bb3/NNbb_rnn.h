/*
 * NNcdr.h
 *
 *  Created on: Dec 27, 2015
 *      Author: fox
 */

#ifndef NNBB3_RNN_H_
#define NNBB3_RNN_H_

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
#include "WordNet.h"
#include "Word2Vec.h"
#include "utils.h"
#include "Example.h"
#include "PoolLSTMClassifier.h"
#include "PoolLSTMAttClassifier2.h"

using namespace nr;
using namespace std;



class NNbb3_rnn {
public:
	Options m_options;

	Alphabet m_wordAlphabet;

	Alphabet m_randomWordAlphabet;
	Alphabet m_characterAlphabet;
	Alphabet m_nerAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_sstAlphabet;

	string nullkey;
	string unknownkey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  PoolLSTMClassifier<cpu> m_classifier;
  //PoolLSTMAttClassifier2<cpu> m_classifier;
#endif


  NNbb3_rnn(const Options &options):m_options(options)/*, m_classifier(options)*/ {
		nullkey = "-#null#-";
		unknownkey = "-#unknown#-";

	}


	void train(bool usedev, const string& trainFile, const string& devFile, const string& testFile, const string& otherDir,
			Tool& tool,
			const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile,
			const string& otherNlpDir) {


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
		vector<Document> otherDocuments;
		if(!otherDir.empty()) {
			parseBB2(otherDir, otherDocuments);
			loadNlpFile(otherNlpDir, otherDocuments);
			// add otherDocuments to trainDocuments
			for(int i=0;i<otherDocuments.size();i++) {
				trainDocuments.push_back(otherDocuments[i]);
			}
		}


/*		if(usedev) {
			for(int i=0;i<devDocuments.size();i++) {
				trainDocuments.push_back(devDocuments[i]);
			}
			devDocuments.clear();
			for(int i=0;i<testDocuments.size();i++) {
				devDocuments.push_back(testDocuments[i]);
			}
		}*/


		cout << "Creating Alphabet..." << endl;

		// For all alphabets, unknownkey and nullkey should be 0 and 1.

		m_wordAlphabet.clear();
		m_wordAlphabet.from_string(unknownkey);
		m_wordAlphabet.from_string(nullkey);

		m_randomWordAlphabet.clear();
		m_randomWordAlphabet.from_string(unknownkey);
		m_randomWordAlphabet.from_string(nullkey);

		m_characterAlphabet.clear();
		m_characterAlphabet.from_string(unknownkey);
		m_characterAlphabet.from_string(nullkey);

		m_nerAlphabet.clear();
		m_nerAlphabet.from_string(unknownkey);
		m_nerAlphabet.from_string(nullkey);

		m_posAlphabet.clear();
		m_posAlphabet.from_string(unknownkey);
		m_posAlphabet.from_string(nullkey);

		m_sstAlphabet.clear();
		m_sstAlphabet.from_string(unknownkey);
		m_sstAlphabet.from_string(nullkey);


		createAlphabet(trainDocuments, tool, true);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool, false);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool, false);
		}


		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				randomInitNrmat(wordEmb, m_wordAlphabet, m_options.wordEmbSize, 1000);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);
				// format the words of pre-trained embeddings
				//formatWords(tool.w2v);
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

				// format the words of pre-trained embeddings
				//formatWords(tool.w2v);

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

		NRMat<dtype> randomWordEmb;
		randomInitNrmat(randomWordEmb, m_randomWordAlphabet, m_options.entity_embsize, 1001);
		NRMat<dtype> characterEmb;
		randomInitNrmat(characterEmb, m_characterAlphabet, m_options.otherEmbSize, 1002);
		NRMat<dtype> nerEmb;
		randomInitNrmat(nerEmb, m_nerAlphabet, m_options.otherEmbSize, 1003);
		NRMat<dtype> posEmb;
		randomInitNrmat(posEmb, m_posAlphabet, m_options.otherEmbSize, 1004);
		NRMat<dtype> sstEmb;
		randomInitNrmat(sstEmb, m_sstAlphabet, m_options.otherEmbSize, 1005);



		vector<Example> trainExamples;
		initialExamples(tool, trainDocuments, trainExamples, true, false);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;



/*		m_classifier.init(2, wordEmb, wordnetEmb,brownEmb, nerEmb, posEmb, sstEmb,
				 m_sparseAlphabet.size());*/
		  m_classifier.init(wordEmb, m_options);
		  m_classifier.resetRemove(m_options.removePool);
		  m_classifier.setDropValue(m_options.dropProb);
		  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);

			m_classifier._randomWord.initial(randomWordEmb);
			m_classifier._randomWord.setEmbFineTune(true);

			m_classifier._character.initial(characterEmb);
			m_classifier._character.setEmbFineTune(true);

			m_classifier._ner.initial(nerEmb);
			m_classifier._ner.setEmbFineTune(true);

			m_classifier._pos.initial(posEmb);
			m_classifier._pos.setEmbFineTune(true);

			m_classifier._sst.initial(sstEmb);
			m_classifier._sst.setEmbFineTune(true);



		int inputSize = trainExamples.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval, metric_dev;
		static vector<Example> subExamples;

		dtype best = 0;
		vector<Example> toBeOutput;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {

				cout << "##### Iteration " << iter << std::endl;

		    random_shuffle(indexes.begin(), indexes.end());
		    eval.reset();

		    // use all batches to train during an iteration
		    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * m_options.batchSize;
				int end_pos = (updateIter + 1) * m_options.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.push_back(trainExamples[indexes[idy]]);
				}

				int curUpdateIter = iter * batchBlock + updateIter;
				dtype cost = m_classifier.process(subExamples, curUpdateIter);

				eval.overall_label_count += m_classifier._eval.overall_label_count;
				eval.correct_label_count += m_classifier._eval.correct_label_count;

		      if (m_options.verboseIter>0 && (curUpdateIter + 1) % m_options.verboseIter == 0) {
		        //m_classifier.checkgrads(subExamples, curUpdateIter+1);
		        //std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
		        //std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
		      }
		      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);

		    }

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordEmbFineTune && m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

					if((m_options.channelMode & 2) == 2) {
						averageUnkownEmb(m_randomWordAlphabet, m_classifier._randomWord, m_options.entity_embsize);
					}
					if((m_options.channelMode & 4) == 4) {
						averageUnkownEmb(m_characterAlphabet, m_classifier._character, m_options.otherEmbSize);
					}
					if((m_options.channelMode & 8) == 8) {
						averageUnkownEmb(m_nerAlphabet, m_classifier._ner, m_options.otherEmbSize);
					}
					if((m_options.channelMode & 16) == 16) {
						averageUnkownEmb(m_posAlphabet, m_classifier._pos, m_options.otherEmbSize);
					}
					if((m_options.channelMode & 32) == 32) {
						averageUnkownEmb(m_sstAlphabet, m_classifier._sst, m_options.otherEmbSize);
					}
		    	}

		    	metric_dev.reset();

/*		    	if(usedev) {
					// clear output dir
					string s = "rm -f "+m_options.output+"/*";
					system(s.c_str());

					for(int i=0;i<devDocuments.size();i++) {
						vector<Example> tempExamples;
						vector<Document> temp;
						temp.push_back(devDocuments[i]);
						initialExamples(tool, temp, tempExamples, false, true);

						toBeOutput.clear();
						for (int idx = 0; idx < tempExamples.size(); idx++) {
							vector<double> scores(2);
							m_classifier.predict(tempExamples[idx], scores);
							bool predicted = scores[0]>scores[1] ? true:false; // positive

							if(predicted) {
								toBeOutput.push_back(tempExamples[idx]);
							}
						}
						outputResults(devDocuments[i].id, toBeOutput, m_options.output);

					}
		    	} else {*/
			    	for (int i = 0; i < devDocuments.size(); i++) {
						vector<Example> tempExamples;
						vector<Document> temp;
						temp.push_back(devDocuments[i]);
						initialExamples(tool, temp, tempExamples, false, false);

						metric_dev.overall_label_count += devDocuments[i].relations.size();
						for (int idx = 0; idx < tempExamples.size(); idx++) {
							vector<double> scores(2);
							m_classifier.predict(tempExamples[idx], scores);
							bool predicted = scores[0]>scores[1] ? true:false; // positive

							if(predicted) {
								metric_dev.predicated_label_count ++;
								if(isLoc(tempExamples[idx].idBacteria, tempExamples[idx].idLocation, devDocuments[i]))
									metric_dev.correct_label_count++;
							}
						}

					}
					metric_dev.print();

					if (metric_dev.getAccuracy() > best) {
						cout << "Exceeds best performance of " << best << endl;
						best = metric_dev.getAccuracy();
						// if the current exceeds the best, we do the blind test on the test set
						// but don't evaluate and store the results for the official evaluation
						if(!testFile.empty()) {

							// clear output dir
							string s = "rm -f "+m_options.output+"/*";
							system(s.c_str());

							for(int i=0;i<testDocuments.size();i++) {
								vector<Example> testExamples;
								vector<Document> temp;
								temp.push_back(testDocuments[i]);
								initialExamples(tool, temp, testExamples, false, true);

								toBeOutput.clear();
								for (int idx = 0; idx < testExamples.size(); idx++) {
									vector<double> scores(2);
									m_classifier.predict(testExamples[idx], scores);
									bool predicted = scores[0]>scores[1] ? true:false; // positive

									if(predicted) {
										toBeOutput.push_back(testExamples[idx]);
									}
								}
								outputResults(testDocuments[i].id, toBeOutput, m_options.output);

							}


						}
					}
		    	//}



		    } // devExamples > 0

		} // for iter




		m_classifier.release();

	}

	void initialExamples(Tool& tool, const vector<Document>& documents, vector<Example>& examples
			, bool bStatistic, bool bTestSet) {
		int ctPositive = 0;
		int ctNegtive = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {


			for(int sentIdx=0;sentIdx<documents[docIdx].sents.size();sentIdx++) {

				// find all the entities in the current sentence
				vector<Entity> Bentity;
				findEntityInSent(documents[docIdx].sents[sentIdx].begin, documents[docIdx].sents[sentIdx].end, documents[docIdx], Bentity);

				/* for each entity B, scan all the entities A before it but in the sent_window
				 * so A is before B and their type should not be the same
				 */
				int windowBegin = sentIdx-m_options.sent_window+1 >=0 ? sentIdx-m_options.sent_window+1 : 0;

				for(int b=0;b<Bentity.size();b++) {

					for(int i=windowBegin;i<=sentIdx;i++) {
						vector<Entity> Aentity;
						findEntityInSent(documents[docIdx].sents[i].begin, documents[docIdx].sents[i].end, documents[docIdx], Aentity);
						for(int a=0;a<Aentity.size();a++) {

							// > seems more rational since 4329237 T4 and T5, but >= seems to be better empirically
							if(Aentity[a].begin >= Bentity[b].begin)
								continue;
							// get rid of overlapped entity
							if(isEntityOverlapped(Aentity[a], Bentity[b]))
								continue;

							Entity bacteria;
							Entity location;

							if(Aentity[a].type == "Bacteria" &&
									Bentity[b].type != "Bacteria") {
								bacteria = Aentity[a];
								location = Bentity[b];
							} else if(Aentity[a].type != "Bacteria" &&
									Bentity[b].type == "Bacteria") {
								bacteria = Bentity[b];
								location = Aentity[a];
							} else
								continue;

							Example eg;
							if(!bTestSet) {
								if(isLoc(bacteria, location, documents[docIdx])) {
									// positive
									eg.m_labels.push_back(1);
									eg.m_labels.push_back(0);

								} else {
									// negative
									eg.m_labels.push_back(0);
									eg.m_labels.push_back(1);
								}
							}

							int seqIdx = 0;

							for(int j=i;j<=sentIdx;j++) {
								for(int k=0;k<documents[docIdx].sents[j].tokens.size();k++) {
/*									if(fox::Punctuation::isEnglishPunc(documents[docIdx].sents[j].tokens[k].word[0])) {

										continue;
									}*/

									if(isTokenInEntity(documents[docIdx].sents[j].tokens[k], Aentity[a])) {

										if(eg.formerTkBegin == -1)
											eg.formerTkBegin = seqIdx;

										if(eg.formerTkEnd == -1)
											eg.formerTkEnd = seqIdx;
										else if(eg.formerTkEnd < seqIdx)
											eg.formerTkEnd = seqIdx;

									} else if(isTokenInEntity(documents[docIdx].sents[j].tokens[k], Bentity[b])) {

										if(eg.latterTkBegin == -1)
											eg.latterTkBegin = seqIdx;

										if(eg.latterTkEnd == -1)
											eg.latterTkEnd = seqIdx;
										else if(eg.latterTkEnd < seqIdx)
											eg.latterTkEnd = seqIdx;

									} /*else if(isTokenBeforeEntity(documents[docIdx].sents[j].tokens[k], Aentity[a]) ||
											isTokenAfterEntity(documents[docIdx].sents[j].tokens[k], Bentity[b]))
										continue;*/
									/*else if(isTokenBetweenTwoEntities(documents[docIdx].sents[j].tokens[k], Aentity[a], Bentity[b]) ||
											isTokenAfterEntity(documents[docIdx].sents[j].tokens[k], Bentity[b]))
										continue;*/
									else if(isTokenBetweenTwoEntities(documents[docIdx].sents[j].tokens[k], Aentity[a], Bentity[b]) ||
											isTokenBeforeEntity(documents[docIdx].sents[j].tokens[k], Aentity[a]))
									continue;


									Feature feature;

									featureName2ID(m_wordAlphabet, feature_word(documents[docIdx].sents[j].tokens[k]), feature.words);

									if((m_options.channelMode & 2) == 2) {
										feature.randomWord = featureName2ID(m_randomWordAlphabet, feature_randomWord(documents[docIdx].sents[j].tokens[k]));
									}
									if((m_options.channelMode & 4) == 4) {
										featureName2ID(m_characterAlphabet, feature_character(documents[docIdx].sents[j].tokens[k]), feature.characters);
									}
									if((m_options.channelMode & 8) == 8) {
										feature.ner = featureName2ID(m_nerAlphabet, feature_ner(documents[docIdx].sents[j].tokens[k]));
									}
									if((m_options.channelMode & 16) == 16) {
										feature.pos = featureName2ID(m_posAlphabet, feature_pos(documents[docIdx].sents[j].tokens[k]));
									}
									if((m_options.channelMode & 32) == 32) {
										feature.sst = featureName2ID(m_sstAlphabet, feature_sst(documents[docIdx].sents[j].tokens[k]));
									}

									eg.m_features.push_back(feature);

									seqIdx++;

								}
							}


							eg.idBacteria = bacteria.id;
							eg.idLocation = location.id;

/*							assert(eg.formerTkBegin!=-1 && eg.formerTkEnd!=-1);
							assert(eg.latterTkBegin!=-1 && eg.latterTkEnd!=-1);*/
							if(eg.formerTkBegin==-1 && eg.formerTkEnd==-1) {
								cout<<"warning: "<<documents[docIdx].id<<" "<<Aentity[a].text<<" "<<Aentity[a].begin<<" "<<Aentity[a].end<<endl;
								continue;
							}
							if(eg.latterTkBegin==-1 && eg.latterTkEnd==-1) {
								cout<<"warning: "<<documents[docIdx].id<<" "<<Bentity[b].text<<" "<<Bentity[b].begin<<" "<<Bentity[b].end<<endl;
								continue;
							}

							examples.push_back(eg);

							if(!bTestSet) {
								if(eg.m_labels[0]==1) {
									// positive
									ctPositive++;

								} else {
									// negative
									ctNegtive++;
								}
							}

						}


					}

				}


			}


		}

		if(bStatistic) {
			cout<<"Positive example: "<<ctPositive<<endl;
			cout<<"Negative example: "<<ctNegtive<<endl;
			cout<<"Proportion +:"<< (ctPositive*1.0)/(ctPositive+ctNegtive)
					<<" , -:"<<(ctNegtive*1.0)/(ctPositive+ctNegtive)<<endl;
		}

	}

	void createAlphabet (const vector<Document>& documents, Tool& tool,
			bool isTrainSet) {

		hash_map<string, int> word_stat;
		hash_map<string, int> randomWord_stat;
		hash_map<string, int> character_stat;
		hash_map<string, int> ner_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> sst_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<documents[docIdx].sents.size();i++) {

				for(int j=0;j<documents[docIdx].sents[i].tokens.size();j++) {

					string curword = feature_word(documents[docIdx].sents[i].tokens[j]);
					word_stat[curword]++;

					if(isTrainSet && (m_options.channelMode & 2) == 2) {
						string ranWord = feature_randomWord(documents[docIdx].sents[i].tokens[j]);
						randomWord_stat[ranWord]++;
					}
					if(isTrainSet && (m_options.channelMode & 4) == 4) {
						vector<string> characters = feature_character(documents[docIdx].sents[i].tokens[j]);
						for(int i=0;i<characters.size();i++)
							character_stat[characters[i]]++;
					}
					if(isTrainSet && (m_options.channelMode & 8) == 8) {
						string ner = feature_ner(documents[docIdx].sents[i].tokens[j]);
						ner_stat[ner]++;
					}
					if(isTrainSet && (m_options.channelMode & 16) == 16) {
						string pos = feature_pos(documents[docIdx].sents[i].tokens[j]);
						pos_stat[pos]++;
					}
					if(isTrainSet && (m_options.channelMode & 32) == 32) {
						string sst = feature_sst(documents[docIdx].sents[i].tokens[j]);
						sst_stat[sst]++;
					}




				}


			}


		}

		stat2Alphabet(word_stat, m_wordAlphabet, "word");

		if(isTrainSet && (m_options.channelMode & 2) == 2) {
			stat2Alphabet(randomWord_stat, m_randomWordAlphabet, "random word");
		}
		if(isTrainSet && (m_options.channelMode & 4) == 4) {
			stat2Alphabet(character_stat, m_characterAlphabet, "character");
		}
		if(isTrainSet && (m_options.channelMode & 8) == 8) {
			stat2Alphabet(ner_stat, m_nerAlphabet, "ner");
		}
		if(isTrainSet && (m_options.channelMode & 16) == 16) {
			stat2Alphabet(pos_stat, m_posAlphabet, "pos");
		}
		if(isTrainSet && (m_options.channelMode & 32) == 32) {
			stat2Alphabet(sst_stat, m_sstAlphabet, "sst");
		}
	}

	string feature_word(const fox::Token& token) {
		//string ret = normalize_to_lowerwithdigit(token.word);
		string ret = normalize_to_lowerwithdigit(token.lemma);

		return ret;
	}

	string feature_randomWord(const fox::Token& token) {
		return feature_word(token);
	}

	vector<string> feature_character(const fox::Token& token) {
		vector<string> ret;
		string word = feature_word(token);
		for(int i=0;i<word.length();i++)
			ret.push_back(word.substr(i, 1));
		return ret;
	}

/*	string feature_word(const string& str) {
		string ret = normalize_to_lowerwithdigit(str);
		return ret;
	}*/

	string feature_wordnet(const fox::Token& token) {

		string lemmalow = fox::toLowercase(token.lemma);
		char buffer[64] = {0};
		sprintf(buffer, "%s", lemmalow.c_str());

		int pos = -1;
		fox::EnglishPosType type = fox::EnglishPos::getType(token.pos);
		if(type == fox::FOX_NOUN)
			pos = WNNOUN;
		else if(type == fox::FOX_VERB)
			pos = WNVERB;
		else if(type == fox::FOX_ADJ)
			pos = WNADJ;
		else if(type == fox::FOX_ADV)
			pos = WNADV;

		if(pos != -1) {
			string id = fox::getWnID(buffer, pos, 1);
			if(!id.empty())
				return id;
			else
				return unknownkey;
		} else
			return unknownkey;


	}

	string feature_wordnet(const string& str, Tool& tool) {

		string lemmalow = fox::toLowercase(str);
		char buffer[256] = {0};
		sprintf(buffer, "%s", lemmalow.c_str());

		string id = fox::getWnID(buffer, WNNOUN, 1);
		if(!id.empty())
			return id;
		else
			return unknownkey;

	}

	string feature_brown(const fox::Token& token, Tool& tool) {
		string brownID = tool.brown.get(fox::toLowercase(token.word));
		if(!brownID.empty())
			return brownID;
		else
			return unknownkey;
	}

/*	string feature_brown(const string& str, Tool& tool) {
		string brownID = tool.brown.get(fox::toLowercase(str));
		if(!brownID.empty())
			return brownID;
		else
			return unknownkey;
	}*/

	string feature_ner(const fox::Token& token) {
		return token.sst;
	}

	string feature_pos(const fox::Token& token) {
		return token.pos;
	}

	string feature_sst(const fox::Token& token) {
		int pos = token.sst.find("B-");
		if(pos!=-1) {
			return token.sst.substr(pos+2);
		} else {
			pos = token.sst.find("I-");
			if(pos!=-1) {
				return token.sst.substr(pos+2);
			} else
				return token.sst;
		}


	}

	// Firstly, if lemma+prep matches, generate TGVB+triggerVerb+triggerPrep
	// or if only lemma matches, you can generate TGVB+triggerVerb
/*	string feature_trigger(const vector<fox::Token>& tokens, int idx, Tool& tool) {

		string verb = tool.trigger.findVerb(tokens[idx].lemma);
		if(!verb.empty()) {
			string prep = tool.trigger.findPrep(tokens[idx+1].lemma);
			if(!prep.empty())
				return verb+"_"+prep;
			else
				return verb;
		} else
			return "";
	}*/



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

