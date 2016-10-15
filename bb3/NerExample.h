/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_NerEXAMPLE_H_
#define SRC_NerEXAMPLE_H_
#include <vector>

using namespace std;

class NerExample {

public:
  vector<int> m_labels;
  int goldLabel;

  vector<int> _words;
  vector<int> _postags;

  int _prior_ner;

  int _current_idx;


public:
  NerExample()
  {

  }
/*  virtual ~Example()
  {

  }*/



};

#endif /* SRC_EXAMPLE_H_ */
