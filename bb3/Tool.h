/*
 * Tool.h
 *
 *  Created on: Dec 27, 2015
 *      Author: fox
 */

#ifndef TOOL_H_
#define TOOL_H_

#include "SentSplitter.h"
#include "Tokenizer.h"
#include "Options.h"
#include "Word2Vec.h"
#include "FoxUtil.h"
#include "Trigger.h"
//#include "wnb/core/wordnet.hh"

//using namespace wnb;

class Tool {
public:
	Options option;
	fox::SentSplitter sentSplitter;
	fox::Tokenizer tokenizer;
	fox::Word2Vec* w2v;
	fox::BrownClusterUtil brown;
/*	wordnet wn;
	vector<pos_t> wn_pos;*/
	Trigger trigger;

	Tool(Options option) : option(option), sentSplitter(NULL, &option.abbrPath),
			tokenizer(&option.puncPath), /*wn(option.wordnet),*/ brown(option.brown) {

/*		wn_pos.push_back(wnb::N);wn_pos.push_back(wnb::V);
		wn_pos.push_back(wnb::A);  wn_pos.push_back(wnb::S); wn_pos.push_back(wnb::R);*/

		w2v = new fox::Word2Vec();

	}
	virtual ~Tool() {
		delete w2v;
	}

};



#endif /* TOOL_H_ */
