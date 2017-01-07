/*
 * Coreference.h
 *
 *  Created on: Sep 15, 2016
 *      Author: fox
 */

#ifndef COREFERENCE_H_
#define COREFERENCE_H_
#include <set>
#include <string>
#include "Entity.h"
#include "Document.h"
#include "utils.h"

using namespace std;

class Coreference {
public:
	Coreference() {

			dict.insert("cell");dict.insert("pathogen");dict.insert("strain");dict.insert("isolate");
			dict.insert("organism");dict.insert("bacteria");
			dict.insert("bacterium");dict.insert("genus");dict.insert("species");

			pronoun.insert("this");pronoun.insert("This");pronoun.insert("that");pronoun.insert("That");
			pronoun.insert("these");pronoun.insert("These");pronoun.insert("those");pronoun.insert("Those");
			pronoun.insert("the");pronoun.insert("The");


	}

	bool isDict(const string& str) {
		if(dict.find(str) != dict.end())
			return true;
		else
			return false;
	}

	bool isPronoun(const string& str) {
		if(pronoun.find(str) != pronoun.end())
			return true;
		else
			return false;
	}

	bool isLocation(vector<Entity> & recognizedEntities, const fox::Token & token) {
		for(int i=0;i<recognizedEntities.size();i++) {
			const Entity & entity = recognizedEntities[i];
			if(entity.type == TYPE_Hab || entity.type == TYPE_Geo) {
				if(boolTokenInEntity(token, entity))
					return true;
			}
		}
		return false;
	}

	int findClosestBacEntity(vector<Entity> & recognizedEntities, int position) {
		int lastBacIdx = -1;
		for(int i=0;i<recognizedEntities.size();i++) {
			if(recognizedEntities[i].type != TYPE_Bac)
				continue;

			if(recognizedEntities[i].end>=position) {
				break;
			}
			lastBacIdx = i;
		}
		return lastBacIdx;
	}

	void newEntity(const fox::Token& token, const string& labelName, Entity& entity, int entityId) {
		stringstream ss;
		ss<<"T"<<entityId;
		entity.id = ss.str();
		entity.type = labelName.substr(labelName.find("_")+1);
		entity.begin = token.begin;
		entity.end = token.end;
		entity.text = token.word;
	}

	void addToCorefChain(Entity& closest, Entity& current, vector< vector<Entity> > & corefChains) {
		if(corefChains.size()==0) {
			// no chains, create a new one
			vector<Entity> chain;
			chain.push_back(closest);
			chain.push_back(current);
			corefChains.push_back(chain);
			return;
		}

		for(int chainIdx=0;chainIdx<corefChains.size();chainIdx++) {
			vector<Entity> & chain = corefChains[chainIdx];

			for(int entityIdx = 0;entityIdx<chain.size();entityIdx++) {
				Entity & old = chain[entityIdx];

				if(old.equals(closest)) { // exist
					chain.push_back(current);
					return;
				}
			}
		}

		// no match, create a new one
		vector<Entity> chain;
		chain.push_back(closest);
		chain.push_back(current);
		corefChains.push_back(chain);
		return;
	}

	int findEntityChain(const Entity & target, vector< vector<Entity> > & corefChains) {
		int bacteriaChainIdx = -1;
		for(int chainIdx=0;chainIdx<corefChains.size();chainIdx++) {
			const vector<Entity>& chain = corefChains[chainIdx];

			for(int entityIdx=0;entityIdx<chain.size();entityIdx++) {
				const Entity & entity = chain[entityIdx];

				if(entity.equals(target)) {
					bacteriaChainIdx = chainIdx;
					return bacteriaChainIdx;
				}
			}
		}
		return bacteriaChainIdx;
	}

	void coref(const Document & doc, vector<Entity> & recognizedEntities, vector< vector<Entity> > & corefChains) {

		for(int sentIdx=0;sentIdx<doc.sents.size();sentIdx++) {
			const fox::Sent & sent = doc.sents[sentIdx];

			for(int tokenIdx=0;tokenIdx<sent.tokens.size();tokenIdx++) {
				const fox::Token& token = sent.tokens[tokenIdx];

				if(isDict(token.lemma) && !isLocation(recognizedEntities, token)) {
					if(tokenIdx==0 || isPronoun(sent.tokens[tokenIdx-1].word)) {
						// dict word is the first word of a sentence
						// this + dict word
						int idx = findClosestBacEntity(recognizedEntities, token.begin);
						if(idx >= 0) {
							Entity entity;
							newEntity(token, TYPE_Bac, entity, 0);
							entity.sentIdx = sentIdx;
							Entity & closestBac = recognizedEntities[idx];

							addToCorefChain(closestBac, entity, corefChains);
						}

					}
				}
			}

		}


	}





private:
	set<string> dict;

	set<string> pronoun;

};

#endif /* COREFERENCE_H_ */
