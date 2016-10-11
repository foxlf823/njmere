/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_
#include <vector>
#include "N3L.h"

using namespace std;

class Example {

public:
	//used by ner
  vector<int> _nerLabels;
  int nerGoldLabel;

  vector<int> _words;
  vector<int> _postags;
  vector< vector<int> > _seq_chars;

  int _prior_ner;

  int _current_idx;

  // used by relation
  bool _isRelation;

  vector<int> _relLabels;
  int relGoldLabel;

  vector<int> _deps;
  vector<int> _ners;

  vector<int> _idxOnSDP_E12A;
  vector<int> _idxOnSDP_E22A;

  hash_set<int> _idx_e1;
  hash_set<int> _idx_e2;

  vector<int> _between_words;

public:
  Example(bool isrel)
  {
	  nerGoldLabel = -1;
	  _prior_ner = -1;
	  _current_idx = -1;

	  _isRelation = isrel;

	  relGoldLabel = -1;

  }
/*  virtual ~Example()
  {

  }*/



};

#endif /* SRC_EXAMPLE_H_ */
