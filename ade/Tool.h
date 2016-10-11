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


class Tool {
public:
	Options option;
	fox::SentSplitter sentSplitter;
	fox::Tokenizer tokenizer;
	fox::Word2Vec* w2v;


	Tool(Options option) : option(option), sentSplitter(NULL, &option.abbrPath),
			tokenizer(&option.puncPath) {

		w2v = new fox::Word2Vec();

	}
	virtual ~Tool() {
		delete w2v;
	}

};



#endif /* TOOL_H_ */
