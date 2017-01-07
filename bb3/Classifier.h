
#ifndef SRC_Classifier_H_
#define SRC_Classifier_H_

#include <iostream>

#include <assert.h>
#include "N3L.h"
#include "Example.h"
#include "Prediction.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Classifier {
public:
	Classifier() {

  }
  ~Classifier() {

  }

public:
  LookupTable<xpu> _words;
  LookupTable<xpu> _pos;
  LookupTable<xpu> _ner;
  LookupTable<xpu> _dep;

  int _wordDim;
  int _otherDim;

// ner
  int _nerlabelSize;
  int _nerrnn_inputsize;
  int _nerrnn_outputsize;
  int _ner_hidden_inputsize;
  int _ner_hidden_outputsize;
  LSTM<xpu> _nerrnn_left;
  LSTM<xpu> _nerrnn_right;
  UniLayer<xpu> _ner_hidden_layer;
  UniLayer<xpu> _ner_olayer_linear;

  // relation
  int _rellabelSize;
  int _relrnn_inputsize;
  int _relrnn_outputsize;
  int _rel_hidden_inputsize;
  int _rel_hidden_outputsize;
  LSTM<xpu> _relrnn_bottomup;
  LSTM<xpu> _relrnn_topdown;
  UniLayer<xpu> _rel_hidden_layer;
  UniLayer<xpu> _rel_output_layer;

  Metric _eval;

  Options options;


public:

  inline void init(Options options) {

	  this->options = options;

	  _wordDim = options.wordEmbSize;
	  _otherDim = options.otherEmbSize;

	  _nerrnn_inputsize = _wordDim+_otherDim; // word+pos
	  _nerrnn_outputsize = options.rnnHiddenSize;
	  _ner_hidden_inputsize = _nerrnn_outputsize*2 + _otherDim; // birnn+ner
	  _ner_hidden_outputsize = options.hiddenSize;
	  _nerlabelSize = MAX_ENTITY;

	  _nerrnn_left.initial(_nerrnn_outputsize, _nerrnn_inputsize, true, 10);
	  _nerrnn_right.initial(_nerrnn_outputsize, _nerrnn_inputsize, false, 20);
	  _ner_hidden_layer.initial(_ner_hidden_outputsize, _ner_hidden_inputsize, true, 30, 0);
	  _ner_olayer_linear.initial(_nerlabelSize, _ner_hidden_outputsize, true, 40, 2);


	  _rellabelSize = MAX_RELATION;
	  _relrnn_inputsize = _nerrnn_outputsize*2 + _otherDim + _otherDim; // birnn+dep+ner
	  _relrnn_outputsize = options.rnnHiddenSize;
	  _rel_hidden_inputsize = _relrnn_outputsize*4 + _nerrnn_outputsize*2 + _nerrnn_outputsize*2; // e12a+e22a+a2e1+a2e2+e1+e2
	  _rel_hidden_outputsize = options.hiddenSize;
	  _relrnn_bottomup.initial(_relrnn_outputsize, _relrnn_inputsize, true, 50);
	  _relrnn_topdown.initial(_relrnn_outputsize, _relrnn_inputsize, false, 60);
	  _rel_hidden_layer.initial(_rel_hidden_outputsize, _rel_hidden_inputsize, true, 70, 0);
	  _rel_output_layer.initial(_rellabelSize, _rel_hidden_outputsize, true, 80, 2);


    	cout<<"Classifier initial"<<endl;
  }

  inline void release() {
    _words.release();
    _pos.release();
    _ner.release();
    _dep.release();

    _nerrnn_left.release();
    _nerrnn_left.release();
    _ner_hidden_layer.release();
    _ner_olayer_linear.release();

    _relrnn_bottomup.release();
    _relrnn_topdown.release();
    _rel_hidden_layer.release();
    _rel_output_layer.release();
  }

  inline dtype processRel(const vector<Example>& examples, int iter) {
    _eval.reset();
    int example_num = examples.size();
    dtype cost = 0.0;
    for (int count = 0; count < example_num; count++) {
    	const Example& example = examples[count];

      int seq_size = example._words.size();
      int sdp_e1_size = example._idxOnSDP_E12A.size();
      int sdp_e2_size = example._idxOnSDP_E22A.size();

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> posprime, posprimeLoss, posprimeMask;

      Tensor<xpu, 3, dtype> nerrnn_input, nerrnn_inputLoss;

      Tensor<xpu, 3, dtype> nerrnn_hidden_left, nerrnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	  	  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right, nerrnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	  	  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

      Tensor<xpu, 3, dtype> nerrnn_hidden_merge, nerrnn_hidden_mergeLoss;

      Tensor<xpu, 2, dtype> pool_e1, pool_e1_Loss;
      Tensor<xpu, 3, dtype> poolIndex_e1;
      Tensor<xpu, 2, dtype> pool_e2, pool_e2_Loss;
      Tensor<xpu, 3, dtype> poolIndex_e2;

      Tensor<xpu, 3, dtype> nerprime, nerprimeLoss, nerprimeMask;
      Tensor<xpu, 3, dtype> depprime, depprimeLoss, depprimeMask;

      vector< Tensor<xpu, 2, dtype> > relrnn_input_sdp_e1, relrnn_input_sdp_e1_Loss;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e1, relrnn_hidden_bottomup_e1_Loss;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
	  	  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e1, relrnn_hidden_topdown_e1_Loss;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
	  	  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my;

      vector< Tensor<xpu, 2, dtype> > relrnn_input_sdp_e2, relrnn_input_sdp_e2_Loss;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e2, relrnn_hidden_bottomup_e2_Loss;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
	  	  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e2, relrnn_hidden_topdown_e2_Loss;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
	  	  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my;

      Tensor<xpu, 2, dtype> hidden_input, hidden_inputLoss;

      Tensor<xpu, 2, dtype> hidden_output, hidden_outputLoss;

      Tensor<xpu, 2, dtype> output, outputLoss;


      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);

      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      posprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);

      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);
      nerrnn_inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);
      nerrnn_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);

      pool_e1 = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      pool_e1_Loss = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      poolIndex_e1 = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);
      pool_e2 = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      pool_e2_Loss = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      poolIndex_e2 = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);

      nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      nerprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      nerprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);

      depprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      depprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      depprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);

      for(int i=0;i<sdp_e1_size;i++) {
    	  relrnn_input_sdp_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));
    	  relrnn_input_sdp_e1_Loss.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));

		  relrnn_hidden_bottomup_e1_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_Loss.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));

		  relrnn_hidden_topdown_e1_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_Loss.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  relrnn_input_sdp_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));
    	  relrnn_input_sdp_e2_Loss.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));

		  relrnn_hidden_bottomup_e2_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_Loss.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));

		  relrnn_hidden_topdown_e2_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_Loss.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
      }

      hidden_input = NewTensor<xpu>(Shape2(1, _rel_hidden_inputsize), 0.0);
      hidden_inputLoss = NewTensor<xpu>(Shape2(1, _rel_hidden_inputsize), 0.0);

      hidden_output = NewTensor<xpu>(Shape2(1, _rel_hidden_outputsize), 0.0);
      hidden_outputLoss = NewTensor<xpu>(Shape2(1, _rel_hidden_outputsize), 0.0);

      output = NewTensor<xpu>(Shape2(1, _rellabelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _rellabelSize), 0.0);

      // forward
      for (int idx = 0; idx < seq_size; idx++) {
			if(options.dropProb != 0) {
				srand(iter * example_num + count * seq_size + idx);
			}

			_words.GetEmb(example._words[idx], wordprime[idx]);
			if(options.dropProb != 0) {
				dropoutcol(wordprimeMask[idx], options.dropProb);
				wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
			}

		   _pos.GetEmb(example._postags[idx], posprime[idx]);
		   if(options.dropProb != 0) {
			   dropoutcol(posprimeMask[idx], options.dropProb);
			   posprime[idx] = posprime[idx] * posprimeMask[idx];
		   }

		   _dep.GetEmb(example._deps[idx], depprime[idx]);
		   if(options.dropProb != 0) {
			   dropoutcol(depprimeMask[idx], options.dropProb);
			   depprime[idx] = depprime[idx] * depprimeMask[idx];
		   }

		   _ner.GetEmb(example._ners[idx], nerprime[idx]);
		   if(options.dropProb != 0) {
			   dropoutcol(nerprimeMask[idx], options.dropProb);
			   nerprime[idx] = nerprime[idx] * nerprimeMask[idx];
		   }

		   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
      }

      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

      for(int i=0;i<seq_size;i++) {
    	  concat(nerrnn_hidden_left[i], nerrnn_hidden_right[i], nerrnn_hidden_merge[i]);
      }

      avgpool_forward(nerrnn_hidden_merge, pool_e1, poolIndex_e1, example._idx_e1);
      avgpool_forward(nerrnn_hidden_merge, pool_e2, poolIndex_e2, example._idx_e2);

      for(int i=0;i<sdp_e1_size;i++) {
    	  int idx = example._idxOnSDP_E12A[i];
    	  concat(nerrnn_hidden_merge[idx], depprime[idx], nerprime[idx], relrnn_input_sdp_e1[i]);
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  int idx = example._idxOnSDP_E22A[i];
    	  concat(nerrnn_hidden_merge[idx], depprime[idx], nerprime[idx], relrnn_input_sdp_e2[i]);
      }

      _relrnn_bottomup.ComputeForwardScore(relrnn_input_sdp_e1, relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
    		  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my, relrnn_hidden_bottomup_e1);
      _relrnn_topdown.ComputeForwardScore(relrnn_input_sdp_e1, relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
    		  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my, relrnn_hidden_topdown_e1);

      _relrnn_bottomup.ComputeForwardScore(relrnn_input_sdp_e2, relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
    		  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my, relrnn_hidden_bottomup_e2);
      _relrnn_topdown.ComputeForwardScore(relrnn_input_sdp_e2, relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
    		  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my, relrnn_hidden_topdown_e2);


      concat(relrnn_hidden_bottomup_e1[sdp_e1_size-1], relrnn_hidden_bottomup_e2[sdp_e2_size-1],
    		  relrnn_hidden_topdown_e1[0], relrnn_hidden_topdown_e2[0],
			  pool_e1, pool_e2, hidden_input);

      _rel_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

      _rel_output_layer.ComputeForwardScore(hidden_output, output);

      // get delta for each output
      cost += softmax_loss(output, example._relLabels, outputLoss, _eval, example_num);

      // loss backward propagation
      _rel_output_layer.ComputeBackwardLoss(hidden_output, output, outputLoss, hidden_outputLoss);

      _rel_hidden_layer.ComputeBackwardLoss(hidden_input, hidden_output, hidden_outputLoss, hidden_inputLoss);


      unconcat(relrnn_hidden_bottomup_e1_Loss[sdp_e1_size-1], relrnn_hidden_bottomup_e2_Loss[sdp_e2_size-1],
    		  relrnn_hidden_topdown_e1_Loss[0], relrnn_hidden_topdown_e2_Loss[0],
			  pool_e1_Loss, pool_e2_Loss, hidden_inputLoss);


      _relrnn_bottomup.ComputeBackwardLoss(relrnn_input_sdp_e1, relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
    		  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my, relrnn_hidden_bottomup_e1,
			  relrnn_hidden_bottomup_e1_Loss, relrnn_input_sdp_e1_Loss);
      _relrnn_topdown.ComputeBackwardLoss(relrnn_input_sdp_e1, relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
    		  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my, relrnn_hidden_topdown_e1,
			  relrnn_hidden_topdown_e1_Loss, relrnn_input_sdp_e1_Loss);

      _relrnn_bottomup.ComputeBackwardLoss(relrnn_input_sdp_e2, relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
    		  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my, relrnn_hidden_bottomup_e2,
			  relrnn_hidden_bottomup_e2_Loss, relrnn_input_sdp_e2_Loss);
      _relrnn_topdown.ComputeBackwardLoss(relrnn_input_sdp_e2, relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
    		  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my, relrnn_hidden_topdown_e2,
			  relrnn_hidden_topdown_e2_Loss, relrnn_input_sdp_e2_Loss);


      for(int i=0;i<sdp_e1_size;i++) {
    	  int idx = example._idxOnSDP_E12A[i];
    	  unconcat(nerrnn_hidden_mergeLoss[idx], depprimeLoss[idx], nerprimeLoss[idx], relrnn_input_sdp_e1_Loss[i]);
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  int idx = example._idxOnSDP_E22A[i];
    	  unconcat(nerrnn_hidden_mergeLoss[idx], depprimeLoss[idx], nerprimeLoss[idx], relrnn_input_sdp_e2_Loss[i]);
      }

      pool_backward(pool_e1_Loss, poolIndex_e1,  nerrnn_hidden_mergeLoss);
      pool_backward(pool_e2_Loss, poolIndex_e2,  nerrnn_hidden_mergeLoss);

      for(int i=0;i<seq_size;i++) {
    	  unconcat(nerrnn_hidden_leftLoss[i], nerrnn_hidden_rightLoss[i], nerrnn_hidden_mergeLoss[i]);
      }

      _nerrnn_left.ComputeBackwardLoss(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left,
			  nerrnn_hidden_leftLoss, nerrnn_inputLoss);
      _nerrnn_right.ComputeBackwardLoss(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right,
				  nerrnn_hidden_rightLoss, nerrnn_inputLoss);


        for (int idx = 0; idx < seq_size; idx++) {
        	unconcat(wordprimeLoss[idx], posprimeLoss[idx], nerrnn_inputLoss[idx]);

        	if(options.dropProb != 0)
        		wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
        	_words.EmbLoss(example._words[idx], wordprimeLoss[idx]);

			if(options.dropProb != 0)
				posprimeLoss[idx] = posprimeLoss[idx] * posprimeMask[idx];
			_pos.EmbLoss(example._postags[idx], posprimeLoss[idx]);

			if(options.dropProb != 0)
				depprimeLoss[idx] = depprimeLoss[idx] * depprimeMask[idx];
			_dep.EmbLoss(example._deps[idx], depprimeLoss[idx]);

			if(options.dropProb != 0)
				nerprimeLoss[idx] = nerprimeLoss[idx] * nerprimeMask[idx];
			_ner.EmbLoss(example._ners[idx], nerprimeLoss[idx]);

        }


      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);

      FreeSpace(&posprime);
      FreeSpace(&posprimeLoss);
      FreeSpace(&posprimeMask);

      FreeSpace(&nerrnn_input);
      FreeSpace(&nerrnn_inputLoss);

      FreeSpace(&nerrnn_hidden_left_iy);
      FreeSpace(&nerrnn_hidden_left_oy);
      FreeSpace(&nerrnn_hidden_left_fy);
      FreeSpace(&nerrnn_hidden_left_mcy);
      FreeSpace(&nerrnn_hidden_left_cy);
      FreeSpace(&nerrnn_hidden_left_my);
      FreeSpace(&nerrnn_hidden_left);
      FreeSpace(&nerrnn_hidden_leftLoss);

      FreeSpace(&nerrnn_hidden_right_iy);
      FreeSpace(&nerrnn_hidden_right_oy);
      FreeSpace(&nerrnn_hidden_right_fy);
      FreeSpace(&nerrnn_hidden_right_mcy);
      FreeSpace(&nerrnn_hidden_right_cy);
      FreeSpace(&nerrnn_hidden_right_my);
      FreeSpace(&nerrnn_hidden_right);
      FreeSpace(&nerrnn_hidden_rightLoss);

      FreeSpace(&nerrnn_hidden_merge);
      FreeSpace(&nerrnn_hidden_mergeLoss);

      FreeSpace(&pool_e1);
      FreeSpace(&pool_e1_Loss);
      FreeSpace(&poolIndex_e1);
      FreeSpace(&pool_e2);
      FreeSpace(&pool_e2_Loss);
      FreeSpace(&poolIndex_e2);

      FreeSpace(&nerprime);
      FreeSpace(&nerprimeLoss);
      FreeSpace(&nerprimeMask);

      FreeSpace(&depprime);
      FreeSpace(&depprimeLoss);
      FreeSpace(&depprimeMask);

      for(int i=0;i<sdp_e1_size;i++) {
    	  FreeSpace(&(relrnn_input_sdp_e1[i]));
    	  FreeSpace(&(relrnn_input_sdp_e1_Loss[i]));

		  FreeSpace(&(relrnn_hidden_bottomup_e1_iy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_oy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_fy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_mcy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_cy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_my[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_Loss[i]));

		  FreeSpace(&(relrnn_hidden_topdown_e1_iy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_oy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_fy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_mcy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_cy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_my[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_Loss[i]));
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  FreeSpace(&(relrnn_input_sdp_e2[i]));
    	  FreeSpace(&(relrnn_input_sdp_e2_Loss[i]));

		  FreeSpace(&(relrnn_hidden_bottomup_e2_iy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_oy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_fy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_mcy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_cy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_my[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_Loss[i]));

		  FreeSpace(&(relrnn_hidden_topdown_e2_iy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_oy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_fy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_mcy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_cy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_my[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_Loss[i]));
      }

      FreeSpace(&hidden_input);
      FreeSpace(&hidden_inputLoss);

      FreeSpace(&hidden_output);
      FreeSpace(&hidden_outputLoss);

      FreeSpace(&output);
      FreeSpace(&outputLoss);

    }

    return cost;
  }

  int predictRel(const Example& example, vector<dtype>& results) {

      int seq_size = example._words.size();
      int sdp_e1_size = example._idxOnSDP_E12A.size();
      int sdp_e2_size = example._idxOnSDP_E22A.size();

      Tensor<xpu, 3, dtype> wordprime;
      Tensor<xpu, 3, dtype> posprime;

      Tensor<xpu, 3, dtype> nerrnn_input;

      Tensor<xpu, 3, dtype> nerrnn_hidden_left;
      Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	  	  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	  	  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

      Tensor<xpu, 3, dtype> nerrnn_hidden_merge;

      Tensor<xpu, 2, dtype> pool_e1;
      Tensor<xpu, 3, dtype> poolIndex_e1;
      Tensor<xpu, 2, dtype> pool_e2;
      Tensor<xpu, 3, dtype> poolIndex_e2;

      Tensor<xpu, 3, dtype> nerprime;
      Tensor<xpu, 3, dtype> depprime;

      vector< Tensor<xpu, 2, dtype> > relrnn_input_sdp_e1;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e1;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
	  	  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e1;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
	  	  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my;

      vector< Tensor<xpu, 2, dtype> > relrnn_input_sdp_e2;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e2;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
	  	  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e2;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
	  	  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my;

      Tensor<xpu, 2, dtype> hidden_input;

      Tensor<xpu, 2, dtype> hidden_output;

      Tensor<xpu, 2, dtype> output;


      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);

      pool_e1 = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      poolIndex_e1 = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);
      pool_e2 = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      poolIndex_e2 = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);

      nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

      depprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

      for(int i=0;i<sdp_e1_size;i++) {
    	  relrnn_input_sdp_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));

		  relrnn_hidden_bottomup_e1_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));

		  relrnn_hidden_topdown_e1_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  relrnn_input_sdp_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));

		  relrnn_hidden_bottomup_e2_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));

		  relrnn_hidden_topdown_e2_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
      }

      hidden_input = NewTensor<xpu>(Shape2(1, _rel_hidden_inputsize), 0.0);

      hidden_output = NewTensor<xpu>(Shape2(1, _rel_hidden_outputsize), 0.0);

      output = NewTensor<xpu>(Shape2(1, _rellabelSize), 0.0);

      // forward
      for (int idx = 0; idx < seq_size; idx++) {

			_words.GetEmb(example._words[idx], wordprime[idx]);

		   _pos.GetEmb(example._postags[idx], posprime[idx]);

		   _dep.GetEmb(example._deps[idx], depprime[idx]);

		   _ner.GetEmb(example._ners[idx], nerprime[idx]);

		   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
      }

      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

      for(int i=0;i<seq_size;i++) {
    	  concat(nerrnn_hidden_left[i], nerrnn_hidden_right[i], nerrnn_hidden_merge[i]);
      }

      avgpool_forward(nerrnn_hidden_merge, pool_e1, poolIndex_e1, example._idx_e1);
      avgpool_forward(nerrnn_hidden_merge, pool_e2, poolIndex_e2, example._idx_e2);

      for(int i=0;i<sdp_e1_size;i++) {
    	  int idx = example._idxOnSDP_E12A[i];
    	  concat(nerrnn_hidden_merge[idx], depprime[idx], nerprime[idx], relrnn_input_sdp_e1[i]);
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  int idx = example._idxOnSDP_E22A[i];
    	  concat(nerrnn_hidden_merge[idx], depprime[idx], nerprime[idx], relrnn_input_sdp_e2[i]);
      }

      _relrnn_bottomup.ComputeForwardScore(relrnn_input_sdp_e1, relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
    		  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my, relrnn_hidden_bottomup_e1);
      _relrnn_topdown.ComputeForwardScore(relrnn_input_sdp_e1, relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
    		  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my, relrnn_hidden_topdown_e1);

      _relrnn_bottomup.ComputeForwardScore(relrnn_input_sdp_e2, relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
    		  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my, relrnn_hidden_bottomup_e2);
      _relrnn_topdown.ComputeForwardScore(relrnn_input_sdp_e2, relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
    		  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my, relrnn_hidden_topdown_e2);


      concat(relrnn_hidden_bottomup_e1[sdp_e1_size-1], relrnn_hidden_bottomup_e2[sdp_e2_size-1],
    		  relrnn_hidden_topdown_e1[0], relrnn_hidden_topdown_e2[0],
			  pool_e1, pool_e2, hidden_input);

      _rel_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

      _rel_output_layer.ComputeForwardScore(hidden_output, output);


      int optLabel = softmax_predict(output, results);


      //release
      FreeSpace(&wordprime);

      FreeSpace(&posprime);

      FreeSpace(&nerrnn_input);

      FreeSpace(&nerrnn_hidden_left_iy);
      FreeSpace(&nerrnn_hidden_left_oy);
      FreeSpace(&nerrnn_hidden_left_fy);
      FreeSpace(&nerrnn_hidden_left_mcy);
      FreeSpace(&nerrnn_hidden_left_cy);
      FreeSpace(&nerrnn_hidden_left_my);
      FreeSpace(&nerrnn_hidden_left);

      FreeSpace(&nerrnn_hidden_right_iy);
      FreeSpace(&nerrnn_hidden_right_oy);
      FreeSpace(&nerrnn_hidden_right_fy);
      FreeSpace(&nerrnn_hidden_right_mcy);
      FreeSpace(&nerrnn_hidden_right_cy);
      FreeSpace(&nerrnn_hidden_right_my);
      FreeSpace(&nerrnn_hidden_right);

      FreeSpace(&nerrnn_hidden_merge);

      FreeSpace(&pool_e1);
      FreeSpace(&poolIndex_e1);
      FreeSpace(&pool_e2);
      FreeSpace(&poolIndex_e2);

      FreeSpace(&nerprime);

      FreeSpace(&depprime);


      for(int i=0;i<sdp_e1_size;i++) {
    	  FreeSpace(&(relrnn_input_sdp_e1[i]));

		  FreeSpace(&(relrnn_hidden_bottomup_e1_iy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_oy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_fy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_mcy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_cy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_my[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1[i]));

		  FreeSpace(&(relrnn_hidden_topdown_e1_iy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_oy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_fy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_mcy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_cy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_my[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1[i]));
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  FreeSpace(&(relrnn_input_sdp_e2[i]));

		  FreeSpace(&(relrnn_hidden_bottomup_e2_iy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_oy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_fy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_mcy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_cy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_my[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2[i]));

		  FreeSpace(&(relrnn_hidden_topdown_e2_iy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_oy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_fy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_mcy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_cy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_my[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2[i]));
      }

      FreeSpace(&hidden_input);

      FreeSpace(&hidden_output);

      FreeSpace(&output);



    return optLabel;
  }

  inline dtype processNer(const vector<Example>& examples, int iter) {
    _eval.reset();
    int example_num = examples.size();
    dtype cost = 0.0;
    for (int count = 0; count < example_num; count++) {
    	const Example& example = examples[count];

      int seq_size = example._words.size();

      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> posprime, posprimeLoss, posprimeMask;

      Tensor<xpu, 3, dtype> nerrnn_input, nerrnn_inputLoss;

      Tensor<xpu, 3, dtype> nerrnn_hidden_left, nerrnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	  	  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right, nerrnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	  	  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

      Tensor<xpu, 2, dtype> nerprime, nerprimeLoss, nerprimeMask;

      Tensor<xpu, 2, dtype> hidden_input, hidden_inputLoss;

      Tensor<xpu, 2, dtype> hidden_output, hidden_outputLoss;

      Tensor<xpu, 2, dtype> output, outputLoss;


      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);

      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      posprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);

      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);
      nerrnn_inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerprime = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      nerprimeLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
      nerprimeMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);

      hidden_input = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);
      hidden_inputLoss = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);

      hidden_output = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);
      hidden_outputLoss = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);

      output = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);

      // forward
      for (int idx = 0; idx < seq_size; idx++) {
			if(options.dropProb != 0) {
				srand(iter * example_num + count * seq_size + idx);
			}

			_words.GetEmb(example._words[idx], wordprime[idx]);
			if(options.dropProb != 0) {
				dropoutcol(wordprimeMask[idx], options.dropProb);
				wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
			}

		   _pos.GetEmb(example._postags[idx], posprime[idx]);
		   if(options.dropProb != 0) {
			   dropoutcol(posprimeMask[idx], options.dropProb);
			   posprime[idx] = posprime[idx] * posprimeMask[idx];
		   }


		   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
      }

      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

	   _ner.GetEmb(example._prior_ner, nerprime);
	   if(options.dropProb != 0) {
		   dropoutcol(nerprimeMask, options.dropProb);
		   nerprime = nerprime * nerprimeMask;
	   }

      concat(nerrnn_hidden_left[example._current_idx], nerrnn_hidden_right[example._current_idx], nerprime, hidden_input);

      _ner_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

      _ner_olayer_linear.ComputeForwardScore(hidden_output, output);

/*		for(int i=0;i<output.size(1);i++) {
			cout<<output[0][i]<<" ";
		}
		cout<<endl;*/

      // get delta for each output
      cost += softmax_loss(output, example._nerLabels, outputLoss, _eval, example_num);

      // loss backward propagation
      _ner_olayer_linear.ComputeBackwardLoss(hidden_output, output, outputLoss, hidden_outputLoss);

      _ner_hidden_layer.ComputeBackwardLoss(hidden_input, hidden_output, hidden_outputLoss, hidden_inputLoss);

      unconcat(nerrnn_hidden_leftLoss[example._current_idx], nerrnn_hidden_rightLoss[example._current_idx], nerprimeLoss, hidden_inputLoss);

      if(options.dropProb != 0)
    	  nerprimeLoss = nerprimeLoss * nerprimeMask;
      _ner.EmbLoss(example._prior_ner, nerprimeLoss);

      _nerrnn_left.ComputeBackwardLoss(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left,
			  nerrnn_hidden_leftLoss, nerrnn_inputLoss);
      _nerrnn_right.ComputeBackwardLoss(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right,
				  nerrnn_hidden_rightLoss, nerrnn_inputLoss);


        for (int idx = 0; idx < seq_size; idx++) {
        	unconcat(wordprimeLoss[idx], posprimeLoss[idx], nerrnn_inputLoss[idx]);

        	if(options.dropProb != 0)
        		wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
        	_words.EmbLoss(example._words[idx], wordprimeLoss[idx]);

			if(options.dropProb != 0)
				posprimeLoss[idx] = posprimeLoss[idx] * posprimeMask[idx];
			_pos.EmbLoss(example._postags[idx], posprimeLoss[idx]);

        }


      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);

      FreeSpace(&posprime);
      FreeSpace(&posprimeLoss);
      FreeSpace(&posprimeMask);

      FreeSpace(&nerrnn_input);
      FreeSpace(&nerrnn_inputLoss);

      FreeSpace(&nerrnn_hidden_left_iy);
      FreeSpace(&nerrnn_hidden_left_oy);
      FreeSpace(&nerrnn_hidden_left_fy);
      FreeSpace(&nerrnn_hidden_left_mcy);
      FreeSpace(&nerrnn_hidden_left_cy);
      FreeSpace(&nerrnn_hidden_left_my);
      FreeSpace(&nerrnn_hidden_left);
      FreeSpace(&nerrnn_hidden_leftLoss);

      FreeSpace(&nerrnn_hidden_right_iy);
      FreeSpace(&nerrnn_hidden_right_oy);
      FreeSpace(&nerrnn_hidden_right_fy);
      FreeSpace(&nerrnn_hidden_right_mcy);
      FreeSpace(&nerrnn_hidden_right_cy);
      FreeSpace(&nerrnn_hidden_right_my);
      FreeSpace(&nerrnn_hidden_right);
      FreeSpace(&nerrnn_hidden_rightLoss);

      FreeSpace(&nerprime);
      FreeSpace(&nerprimeLoss);
      FreeSpace(&nerprimeMask);

      FreeSpace(&hidden_input);
      FreeSpace(&hidden_inputLoss);

      FreeSpace(&hidden_output);
      FreeSpace(&hidden_outputLoss);

      FreeSpace(&output);
      FreeSpace(&outputLoss);

    }

    return cost;
  }

  inline dtype processNerGlobal(vector<Prediction> & beam, Prediction& gold, int sentence_size, int iter) {
    _eval.reset();

    dtype cost = 0.0;
	// make sure the step size
	int step_size = beam[0].labels.size() < sentence_size ? beam[0].labels.size():sentence_size;

    int example_num = step_size;
	dtype goldPathScoreSum = 0;

	// for training, we need to compute expsum in each prediction of beam
    dtype zg = 0;
	for(int i=0;i<beam.size();i++) {
		beam[i].expsum = exp(beam[i].sum-beam[0].sum);
		zg += beam[i].expsum;
	}


	for(int stepIdx=0;stepIdx<step_size;stepIdx++) {
		const Example& example = gold.examples[stepIdx];

		  int seq_size = example._words.size();

		  Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
		  Tensor<xpu, 3, dtype> posprime, posprimeLoss, posprimeMask;

		  Tensor<xpu, 3, dtype> nerrnn_input, nerrnn_inputLoss;

		  Tensor<xpu, 3, dtype> nerrnn_hidden_left, nerrnn_hidden_leftLoss;
		  Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
			  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
		  Tensor<xpu, 3, dtype> nerrnn_hidden_right, nerrnn_hidden_rightLoss;
		  Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
			  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

		  Tensor<xpu, 2, dtype> nerprime, nerprimeLoss, nerprimeMask;

		  Tensor<xpu, 2, dtype> hidden_input, hidden_inputLoss;

		  Tensor<xpu, 2, dtype> hidden_output, hidden_outputLoss;

		  Tensor<xpu, 2, dtype> output, outputLoss;


		  //initialize
		  wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
		  wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
		  wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);

		  posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
		  posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
		  posprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);

		  nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);
		  nerrnn_inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

		  nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

		  nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		  nerrnn_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

		  nerprime = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
		  nerprimeLoss = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);
		  nerprimeMask = NewTensor<xpu>(Shape2(1, _otherDim), 1.0);

		  hidden_input = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);
		  hidden_inputLoss = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);

		  hidden_output = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);
		  hidden_outputLoss = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);

		  output = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);
		  outputLoss = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);

		  // forward
		  for (int idx = 0; idx < seq_size; idx++) {
				if(options.dropProb != 0) {
					srand(iter * example_num + stepIdx * seq_size + idx);
				}

				_words.GetEmb(example._words[idx], wordprime[idx]);
				if(options.dropProb != 0) {
					dropoutcol(wordprimeMask[idx], options.dropProb);
					wordprime[idx] = wordprime[idx] * wordprimeMask[idx];
				}

			   _pos.GetEmb(example._postags[idx], posprime[idx]);
			   if(options.dropProb != 0) {
				   dropoutcol(posprimeMask[idx], options.dropProb);
				   posprime[idx] = posprime[idx] * posprimeMask[idx];
			   }


			   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
		  }

		  _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
				  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
		  _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
					  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

		   _ner.GetEmb(example._prior_ner, nerprime);
		   if(options.dropProb != 0) {
			   dropoutcol(nerprimeMask, options.dropProb);
			   nerprime = nerprime * nerprimeMask;
		   }

		  concat(nerrnn_hidden_left[example._current_idx], nerrnn_hidden_right[example._current_idx], nerprime, hidden_input);

		  _ner_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

		  _ner_olayer_linear.ComputeForwardScore(hidden_output, output);


		  goldPathScoreSum += output[0][example.nerGoldLabel];
		  globalnormalization_loss(output, example._nerLabels, outputLoss,
				  beam, zg, stepIdx, _eval, example_num);

		  // loss backward propagation
		  _ner_olayer_linear.ComputeBackwardLoss(hidden_output, output, outputLoss, hidden_outputLoss);

		  _ner_hidden_layer.ComputeBackwardLoss(hidden_input, hidden_output, hidden_outputLoss, hidden_inputLoss);

		  unconcat(nerrnn_hidden_leftLoss[example._current_idx], nerrnn_hidden_rightLoss[example._current_idx], nerprimeLoss, hidden_inputLoss);

		  if(options.dropProb != 0)
			  nerprimeLoss = nerprimeLoss * nerprimeMask;
		  _ner.EmbLoss(example._prior_ner, nerprimeLoss);

		  _nerrnn_left.ComputeBackwardLoss(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
				  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left,
				  nerrnn_hidden_leftLoss, nerrnn_inputLoss);
		  _nerrnn_right.ComputeBackwardLoss(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
					  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right,
					  nerrnn_hidden_rightLoss, nerrnn_inputLoss);


			for (int idx = 0; idx < seq_size; idx++) {
				unconcat(wordprimeLoss[idx], posprimeLoss[idx], nerrnn_inputLoss[idx]);

				if(options.dropProb != 0)
					wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
				_words.EmbLoss(example._words[idx], wordprimeLoss[idx]);

				if(options.dropProb != 0)
					posprimeLoss[idx] = posprimeLoss[idx] * posprimeMask[idx];
				_pos.EmbLoss(example._postags[idx], posprimeLoss[idx]);

			}


		  //release
		  FreeSpace(&wordprime);
		  FreeSpace(&wordprimeLoss);
		  FreeSpace(&wordprimeMask);

		  FreeSpace(&posprime);
		  FreeSpace(&posprimeLoss);
		  FreeSpace(&posprimeMask);

		  FreeSpace(&nerrnn_input);
		  FreeSpace(&nerrnn_inputLoss);

		  FreeSpace(&nerrnn_hidden_left_iy);
		  FreeSpace(&nerrnn_hidden_left_oy);
		  FreeSpace(&nerrnn_hidden_left_fy);
		  FreeSpace(&nerrnn_hidden_left_mcy);
		  FreeSpace(&nerrnn_hidden_left_cy);
		  FreeSpace(&nerrnn_hidden_left_my);
		  FreeSpace(&nerrnn_hidden_left);
		  FreeSpace(&nerrnn_hidden_leftLoss);

		  FreeSpace(&nerrnn_hidden_right_iy);
		  FreeSpace(&nerrnn_hidden_right_oy);
		  FreeSpace(&nerrnn_hidden_right_fy);
		  FreeSpace(&nerrnn_hidden_right_mcy);
		  FreeSpace(&nerrnn_hidden_right_cy);
		  FreeSpace(&nerrnn_hidden_right_my);
		  FreeSpace(&nerrnn_hidden_right);
		  FreeSpace(&nerrnn_hidden_rightLoss);

		  FreeSpace(&nerprime);
		  FreeSpace(&nerprimeLoss);
		  FreeSpace(&nerprimeMask);

		  FreeSpace(&hidden_input);
		  FreeSpace(&hidden_inputLoss);

		  FreeSpace(&hidden_output);
		  FreeSpace(&hidden_outputLoss);

		  FreeSpace(&output);
		  FreeSpace(&outputLoss);
	}


	// compute the gold path score


	cost += (log(zg) - goldPathScoreSum) / example_num;

    return cost;
  }

  int predictNer(const Example& example, vector<dtype>& results) {

		int seq_size = example._words.size();

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> posprime;

		Tensor<xpu, 3, dtype> nerrnn_input;

		Tensor<xpu, 3, dtype> nerrnn_hidden_left;
		Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
		Tensor<xpu, 3, dtype> nerrnn_hidden_right;
		Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

		Tensor<xpu, 2, dtype> nerprime;

		Tensor<xpu, 2, dtype> hidden_input;

		Tensor<xpu, 2, dtype> hidden_output;

		Tensor<xpu, 2, dtype> output;


	      //initialize
	      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

	      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

	      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

	      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

	      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

	      nerprime = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);

	      hidden_input = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);

	      hidden_output = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);

	      output = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);


	      // forward
	      for (int idx = 0; idx < seq_size; idx++) {

				_words.GetEmb(example._words[idx], wordprime[idx]);

			   _pos.GetEmb(example._postags[idx], posprime[idx]);

			   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
	      }

	      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
	      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

		   _ner.GetEmb(example._prior_ner, nerprime);

	      concat(nerrnn_hidden_left[example._current_idx], nerrnn_hidden_right[example._current_idx], nerprime, hidden_input);

	      _ner_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

	      _ner_olayer_linear.ComputeForwardScore(hidden_output, output);

		// decode algorithm
		int optLabel = softmax_predict(output, results);


	      //release
	      FreeSpace(&wordprime);

	      FreeSpace(&posprime);

	      FreeSpace(&nerrnn_input);

	      FreeSpace(&nerrnn_hidden_left_iy);
	      FreeSpace(&nerrnn_hidden_left_oy);
	      FreeSpace(&nerrnn_hidden_left_fy);
	      FreeSpace(&nerrnn_hidden_left_mcy);
	      FreeSpace(&nerrnn_hidden_left_cy);
	      FreeSpace(&nerrnn_hidden_left_my);
	      FreeSpace(&nerrnn_hidden_left);

	      FreeSpace(&nerrnn_hidden_right_iy);
	      FreeSpace(&nerrnn_hidden_right_oy);
	      FreeSpace(&nerrnn_hidden_right_fy);
	      FreeSpace(&nerrnn_hidden_right_mcy);
	      FreeSpace(&nerrnn_hidden_right_cy);
	      FreeSpace(&nerrnn_hidden_right_my);
	      FreeSpace(&nerrnn_hidden_right);

	      FreeSpace(&nerprime);

	      FreeSpace(&hidden_input);

	      FreeSpace(&hidden_output);

	      FreeSpace(&output);

		return optLabel;

  }

  int predictNerScore(const Example& example, vector<dtype>& scores) {

		int seq_size = example._words.size();

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> posprime;

		Tensor<xpu, 3, dtype> nerrnn_input;

		Tensor<xpu, 3, dtype> nerrnn_hidden_left;
		Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
		Tensor<xpu, 3, dtype> nerrnn_hidden_right;
		Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

		Tensor<xpu, 2, dtype> nerprime;

		Tensor<xpu, 2, dtype> hidden_input;

		Tensor<xpu, 2, dtype> hidden_output;

		Tensor<xpu, 2, dtype> output;


	      //initialize
	      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

	      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

	      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

	      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

	      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

	      nerprime = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);

	      hidden_input = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);

	      hidden_output = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);

	      output = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);


	      // forward
	      for (int idx = 0; idx < seq_size; idx++) {

				_words.GetEmb(example._words[idx], wordprime[idx]);

			   _pos.GetEmb(example._postags[idx], posprime[idx]);

			   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
	      }

	      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
	      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

		   _ner.GetEmb(example._prior_ner, nerprime);

	      concat(nerrnn_hidden_left[example._current_idx], nerrnn_hidden_right[example._current_idx], nerprime, hidden_input);

	      _ner_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

	      _ner_olayer_linear.ComputeForwardScore(hidden_output, output);

		// decode algorithm
	      int optLabel = -1;
	      for(int i=0;i<_nerlabelSize;i++) {
	    	  scores.push_back(output[0][i]);
	          if (optLabel < 0 || output[0][i] > output[0][optLabel])
	            optLabel = i;
	      }


	      //release
	      FreeSpace(&wordprime);

	      FreeSpace(&posprime);

	      FreeSpace(&nerrnn_input);

	      FreeSpace(&nerrnn_hidden_left_iy);
	      FreeSpace(&nerrnn_hidden_left_oy);
	      FreeSpace(&nerrnn_hidden_left_fy);
	      FreeSpace(&nerrnn_hidden_left_mcy);
	      FreeSpace(&nerrnn_hidden_left_cy);
	      FreeSpace(&nerrnn_hidden_left_my);
	      FreeSpace(&nerrnn_hidden_left);

	      FreeSpace(&nerrnn_hidden_right_iy);
	      FreeSpace(&nerrnn_hidden_right_oy);
	      FreeSpace(&nerrnn_hidden_right_fy);
	      FreeSpace(&nerrnn_hidden_right_mcy);
	      FreeSpace(&nerrnn_hidden_right_cy);
	      FreeSpace(&nerrnn_hidden_right_my);
	      FreeSpace(&nerrnn_hidden_right);

	      FreeSpace(&nerprime);

	      FreeSpace(&hidden_input);

	      FreeSpace(&hidden_output);

	      FreeSpace(&output);

		return optLabel;

  }

  dtype computeScoreRel(const Example& example) {


      int seq_size = example._words.size();
      int sdp_e1_size = example._idxOnSDP_E12A.size();
      int sdp_e2_size = example._idxOnSDP_E22A.size();

      Tensor<xpu, 3, dtype> wordprime;
      Tensor<xpu, 3, dtype> posprime;

      Tensor<xpu, 3, dtype> nerrnn_input;

      Tensor<xpu, 3, dtype> nerrnn_hidden_left;
      Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	  	  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right;
      Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	  	  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

      Tensor<xpu, 3, dtype> nerrnn_hidden_merge;

      Tensor<xpu, 2, dtype> pool_e1;
      Tensor<xpu, 3, dtype> poolIndex_e1;
      Tensor<xpu, 2, dtype> pool_e2;
      Tensor<xpu, 3, dtype> poolIndex_e2;

      Tensor<xpu, 3, dtype> nerprime;
      Tensor<xpu, 3, dtype> depprime;

      vector< Tensor<xpu, 2, dtype> > relrnn_input_sdp_e1;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e1;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
	  	  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e1;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
	  	  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my;

      vector< Tensor<xpu, 2, dtype> > relrnn_input_sdp_e2;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e2;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
	  	  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e2;
      vector< Tensor<xpu, 2, dtype> > relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
	  	  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my;

      Tensor<xpu, 2, dtype> hidden_input;

      Tensor<xpu, 2, dtype> hidden_output;

      Tensor<xpu, 2, dtype> output;


      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

      nerrnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);

      pool_e1 = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      poolIndex_e1 = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);
      pool_e2 = NewTensor<xpu>(Shape2(1, _nerrnn_outputsize*2), 0.0);
      poolIndex_e2 = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize*2), 0.0);

      nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

      depprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

      for(int i=0;i<sdp_e1_size;i++) {
    	  relrnn_input_sdp_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));

		  relrnn_hidden_bottomup_e1_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));

		  relrnn_hidden_topdown_e1_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e1.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  relrnn_input_sdp_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_inputsize), 0.0));

		  relrnn_hidden_bottomup_e2_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_bottomup_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));

		  relrnn_hidden_topdown_e2_iy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_oy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_fy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_mcy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_cy.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2_my.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
		  relrnn_hidden_topdown_e2.push_back(NewTensor<xpu>(Shape2(1, _relrnn_outputsize), 0.0));
      }

      hidden_input = NewTensor<xpu>(Shape2(1, _rel_hidden_inputsize), 0.0);

      hidden_output = NewTensor<xpu>(Shape2(1, _rel_hidden_outputsize), 0.0);

      output = NewTensor<xpu>(Shape2(1, _rellabelSize), 0.0);

      // forward
      for (int idx = 0; idx < seq_size; idx++) {

			_words.GetEmb(example._words[idx], wordprime[idx]);

		   _pos.GetEmb(example._postags[idx], posprime[idx]);

		   _dep.GetEmb(example._deps[idx], depprime[idx]);

		   _ner.GetEmb(example._ners[idx], nerprime[idx]);

		   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
      }

      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

      for(int i=0;i<seq_size;i++) {
    	  concat(nerrnn_hidden_left[i], nerrnn_hidden_right[i], nerrnn_hidden_merge[i]);
      }

      avgpool_forward(nerrnn_hidden_merge, pool_e1, poolIndex_e1, example._idx_e1);
      avgpool_forward(nerrnn_hidden_merge, pool_e2, poolIndex_e2, example._idx_e2);

      for(int i=0;i<sdp_e1_size;i++) {
    	  int idx = example._idxOnSDP_E12A[i];
    	  concat(nerrnn_hidden_merge[idx], depprime[idx], nerprime[idx], relrnn_input_sdp_e1[i]);
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  int idx = example._idxOnSDP_E22A[i];
    	  concat(nerrnn_hidden_merge[idx], depprime[idx], nerprime[idx], relrnn_input_sdp_e2[i]);
      }

      _relrnn_bottomup.ComputeForwardScore(relrnn_input_sdp_e1, relrnn_hidden_bottomup_e1_iy, relrnn_hidden_bottomup_e1_oy, relrnn_hidden_bottomup_e1_fy,
    		  relrnn_hidden_bottomup_e1_mcy, relrnn_hidden_bottomup_e1_cy, relrnn_hidden_bottomup_e1_my, relrnn_hidden_bottomup_e1);
      _relrnn_topdown.ComputeForwardScore(relrnn_input_sdp_e1, relrnn_hidden_topdown_e1_iy, relrnn_hidden_topdown_e1_oy, relrnn_hidden_topdown_e1_fy,
    		  relrnn_hidden_topdown_e1_mcy, relrnn_hidden_topdown_e1_cy, relrnn_hidden_topdown_e1_my, relrnn_hidden_topdown_e1);

      _relrnn_bottomup.ComputeForwardScore(relrnn_input_sdp_e2, relrnn_hidden_bottomup_e2_iy, relrnn_hidden_bottomup_e2_oy, relrnn_hidden_bottomup_e2_fy,
    		  relrnn_hidden_bottomup_e2_mcy, relrnn_hidden_bottomup_e2_cy, relrnn_hidden_bottomup_e2_my, relrnn_hidden_bottomup_e2);
      _relrnn_topdown.ComputeForwardScore(relrnn_input_sdp_e2, relrnn_hidden_topdown_e2_iy, relrnn_hidden_topdown_e2_oy, relrnn_hidden_topdown_e2_fy,
    		  relrnn_hidden_topdown_e2_mcy, relrnn_hidden_topdown_e2_cy, relrnn_hidden_topdown_e2_my, relrnn_hidden_topdown_e2);


      concat(relrnn_hidden_bottomup_e1[sdp_e1_size-1], relrnn_hidden_bottomup_e2[sdp_e2_size-1],
    		  relrnn_hidden_topdown_e1[0], relrnn_hidden_topdown_e2[0],
			  pool_e1, pool_e2, hidden_input);

      _rel_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

      _rel_output_layer.ComputeForwardScore(hidden_output, output);

      dtype cost = softmax_cost(output, example._relLabels);



      //release
      FreeSpace(&wordprime);

      FreeSpace(&posprime);

      FreeSpace(&nerrnn_input);

      FreeSpace(&nerrnn_hidden_left_iy);
      FreeSpace(&nerrnn_hidden_left_oy);
      FreeSpace(&nerrnn_hidden_left_fy);
      FreeSpace(&nerrnn_hidden_left_mcy);
      FreeSpace(&nerrnn_hidden_left_cy);
      FreeSpace(&nerrnn_hidden_left_my);
      FreeSpace(&nerrnn_hidden_left);

      FreeSpace(&nerrnn_hidden_right_iy);
      FreeSpace(&nerrnn_hidden_right_oy);
      FreeSpace(&nerrnn_hidden_right_fy);
      FreeSpace(&nerrnn_hidden_right_mcy);
      FreeSpace(&nerrnn_hidden_right_cy);
      FreeSpace(&nerrnn_hidden_right_my);
      FreeSpace(&nerrnn_hidden_right);

      FreeSpace(&nerrnn_hidden_merge);

      FreeSpace(&pool_e1);
      FreeSpace(&poolIndex_e1);
      FreeSpace(&pool_e2);
      FreeSpace(&poolIndex_e2);

      FreeSpace(&nerprime);

      FreeSpace(&depprime);


      for(int i=0;i<sdp_e1_size;i++) {
    	  FreeSpace(&(relrnn_input_sdp_e1[i]));

		  FreeSpace(&(relrnn_hidden_bottomup_e1_iy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_oy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_fy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_mcy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_cy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1_my[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e1[i]));

		  FreeSpace(&(relrnn_hidden_topdown_e1_iy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_oy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_fy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_mcy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_cy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1_my[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e1[i]));
      }

      for(int i=0;i<sdp_e2_size;i++) {
    	  FreeSpace(&(relrnn_input_sdp_e2[i]));

		  FreeSpace(&(relrnn_hidden_bottomup_e2_iy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_oy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_fy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_mcy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_cy[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2_my[i]));
		  FreeSpace(&(relrnn_hidden_bottomup_e2[i]));

		  FreeSpace(&(relrnn_hidden_topdown_e2_iy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_oy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_fy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_mcy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_cy[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2_my[i]));
		  FreeSpace(&(relrnn_hidden_topdown_e2[i]));
      }

      FreeSpace(&hidden_input);

      FreeSpace(&hidden_output);

      FreeSpace(&output);



    return cost;

  }

  dtype computeScoreNer(const Example& example) {

		int seq_size = example._words.size();

		Tensor<xpu, 3, dtype> wordprime;
		Tensor<xpu, 3, dtype> posprime;

		Tensor<xpu, 3, dtype> nerrnn_input;

		Tensor<xpu, 3, dtype> nerrnn_hidden_left;
		Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
		Tensor<xpu, 3, dtype> nerrnn_hidden_right;
		Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

		Tensor<xpu, 2, dtype> nerprime;

		Tensor<xpu, 2, dtype> hidden_input;

		Tensor<xpu, 2, dtype> hidden_output;

		Tensor<xpu, 2, dtype> output;


	      //initialize
	      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

	      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

	      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

	      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

	      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
	      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

	      nerprime = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);

	      hidden_input = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);

	      hidden_output = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);

	      output = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);


	      // forward
	      for (int idx = 0; idx < seq_size; idx++) {

				_words.GetEmb(example._words[idx], wordprime[idx]);

			   _pos.GetEmb(example._postags[idx], posprime[idx]);

			   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
	      }

	      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
	    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
	      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
	          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

		   _ner.GetEmb(example._prior_ner, nerprime);

	      concat(nerrnn_hidden_left[example._current_idx], nerrnn_hidden_right[example._current_idx], nerprime, hidden_input);

	      _ner_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

	      _ner_olayer_linear.ComputeForwardScore(hidden_output, output);

		// decode algorithm
	      dtype cost = softmax_cost(output, example._nerLabels);

	      //release
	      FreeSpace(&wordprime);

	      FreeSpace(&posprime);

	      FreeSpace(&nerrnn_input);

	      FreeSpace(&nerrnn_hidden_left_iy);
	      FreeSpace(&nerrnn_hidden_left_oy);
	      FreeSpace(&nerrnn_hidden_left_fy);
	      FreeSpace(&nerrnn_hidden_left_mcy);
	      FreeSpace(&nerrnn_hidden_left_cy);
	      FreeSpace(&nerrnn_hidden_left_my);
	      FreeSpace(&nerrnn_hidden_left);

	      FreeSpace(&nerrnn_hidden_right_iy);
	      FreeSpace(&nerrnn_hidden_right_oy);
	      FreeSpace(&nerrnn_hidden_right_fy);
	      FreeSpace(&nerrnn_hidden_right_mcy);
	      FreeSpace(&nerrnn_hidden_right_cy);
	      FreeSpace(&nerrnn_hidden_right_my);
	      FreeSpace(&nerrnn_hidden_right);

	      FreeSpace(&nerprime);

	      FreeSpace(&hidden_input);

	      FreeSpace(&hidden_output);

	      FreeSpace(&output);

		return cost;

}

  dtype computeScoreNerGlobal(vector<Prediction> & beam, Prediction& gold, int sentence_size) {
	  dtype cost = 0;

	  int step_size = beam[0].labels.size() < sentence_size ? beam[0].labels.size():sentence_size;
	  dtype goldPathScoreSum = 0;
	  int example_num = step_size;

	    dtype zg = 0;
		for(int i=0;i<beam.size();i++) {
			beam[i].expsum = exp(beam[i].sum-beam[0].sum);
			zg += beam[i].expsum;
		}

		for(int stepIdx=0;stepIdx<step_size;stepIdx++) {
			const Example& example = gold.examples[stepIdx];

			int seq_size = example._words.size();

			Tensor<xpu, 3, dtype> wordprime;
			Tensor<xpu, 3, dtype> posprime;

			Tensor<xpu, 3, dtype> nerrnn_input;

			Tensor<xpu, 3, dtype> nerrnn_hidden_left;
			Tensor<xpu, 3, dtype> nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
			  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my;
			Tensor<xpu, 3, dtype> nerrnn_hidden_right;
			Tensor<xpu, 3, dtype> nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
			  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my;

			Tensor<xpu, 2, dtype> nerprime;

			Tensor<xpu, 2, dtype> hidden_input;

			Tensor<xpu, 2, dtype> hidden_output;

			Tensor<xpu, 2, dtype> output;


		      //initialize
		      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);

		      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);

		      nerrnn_input = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_inputsize), 0.0);

		      nerrnn_hidden_left_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_left_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_left_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_left_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_left_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_left_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

		      nerrnn_hidden_right_iy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_right_oy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_right_fy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_right_mcy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_right_cy = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_right_my = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);
		      nerrnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _nerrnn_outputsize), 0.0);

		      nerprime = NewTensor<xpu>(Shape2(1, _otherDim), 0.0);

		      hidden_input = NewTensor<xpu>(Shape2(1, _ner_hidden_inputsize), 0.0);

		      hidden_output = NewTensor<xpu>(Shape2(1, _ner_hidden_outputsize), 0.0);

		      output = NewTensor<xpu>(Shape2(1, _nerlabelSize), 0.0);


		      // forward
		      for (int idx = 0; idx < seq_size; idx++) {

					_words.GetEmb(example._words[idx], wordprime[idx]);

				   _pos.GetEmb(example._postags[idx], posprime[idx]);

				   concat(wordprime[idx], posprime[idx], nerrnn_input[idx]);
		      }

		      _nerrnn_left.ComputeForwardScore(nerrnn_input, nerrnn_hidden_left_iy, nerrnn_hidden_left_oy, nerrnn_hidden_left_fy,
		    		  nerrnn_hidden_left_mcy, nerrnn_hidden_left_cy, nerrnn_hidden_left_my, nerrnn_hidden_left);
		      _nerrnn_right.ComputeForwardScore(nerrnn_input, nerrnn_hidden_right_iy, nerrnn_hidden_right_oy, nerrnn_hidden_right_fy,
		          		  nerrnn_hidden_right_mcy, nerrnn_hidden_right_cy, nerrnn_hidden_right_my, nerrnn_hidden_right);

			   _ner.GetEmb(example._prior_ner, nerprime);

		      concat(nerrnn_hidden_left[example._current_idx], nerrnn_hidden_right[example._current_idx], nerprime, hidden_input);

		      _ner_hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

		      _ner_olayer_linear.ComputeForwardScore(hidden_output, output);

		      goldPathScoreSum += output[0][example.nerGoldLabel];
		      //dtype cost = softmax_cost(output, example._nerLabels);

		      //release
		      FreeSpace(&wordprime);

		      FreeSpace(&posprime);

		      FreeSpace(&nerrnn_input);

		      FreeSpace(&nerrnn_hidden_left_iy);
		      FreeSpace(&nerrnn_hidden_left_oy);
		      FreeSpace(&nerrnn_hidden_left_fy);
		      FreeSpace(&nerrnn_hidden_left_mcy);
		      FreeSpace(&nerrnn_hidden_left_cy);
		      FreeSpace(&nerrnn_hidden_left_my);
		      FreeSpace(&nerrnn_hidden_left);

		      FreeSpace(&nerrnn_hidden_right_iy);
		      FreeSpace(&nerrnn_hidden_right_oy);
		      FreeSpace(&nerrnn_hidden_right_fy);
		      FreeSpace(&nerrnn_hidden_right_mcy);
		      FreeSpace(&nerrnn_hidden_right_cy);
		      FreeSpace(&nerrnn_hidden_right_my);
		      FreeSpace(&nerrnn_hidden_right);

		      FreeSpace(&nerprime);

		      FreeSpace(&hidden_input);

		      FreeSpace(&hidden_output);

		      FreeSpace(&output);
	    }

		cost += (log(zg) - goldPathScoreSum) / example_num;

		return cost;

  }

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
	  _ner_olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _ner_hidden_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _nerrnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _nerrnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	  _rel_output_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  _rel_hidden_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);

  _relrnn_bottomup.updateAdaGrad(nnRegular, adaAlpha, adaEps);
  _relrnn_topdown.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	_pos.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	_dep.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _ner.updateAdaGrad(nnRegular, adaAlpha, adaEps);

  }



  void checkgradNer(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScoreNer(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScoreNer(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgradNerGlobal(vector<Prediction> & beam, Prediction& gold, int sentence_size, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = computeScoreNerGlobal(beam, gold, sentence_size);


    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = computeScoreNerGlobal(beam, gold, sentence_size);

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
/*    int step_size = beam[0].labels.size() < sentence_size ? beam[0].labels.size():sentence_size;
    mockGrad = mockGrad / step_size;*/
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgradRel(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScoreRel(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScoreRel(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }


  void checkgradNer(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    if(indexes.size() == 0) return;
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScoreNer(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScoreNer(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgradNerGlobal(vector<Prediction> & beam, Prediction& gold, int sentence_size, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    if(indexes.size() == 0) return;
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = computeScoreNerGlobal(beam, gold, sentence_size);


    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = computeScoreNerGlobal(beam, gold, sentence_size);

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
/*    int step_size = beam[0].labels.size() < sentence_size ? beam[0].labels.size():sentence_size;
    mockGrad = mockGrad / step_size;*/
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgradRel(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    if(indexes.size() == 0) return;
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScoreRel(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScoreRel(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgradsNer(const vector<Example>& examples, int iter) {

    checkgradNer(examples, _ner_olayer_linear._W, _ner_olayer_linear._gradW, "_ner_olayer_linear._W", iter);
    checkgradNer(examples, _ner_olayer_linear._b, _ner_olayer_linear._gradb, "_ner_olayer_linear._b", iter);

    checkgradNer(examples, _ner_hidden_layer._W, _ner_hidden_layer._gradW, "_ner_hidden_layer._W", iter);
    checkgradNer(examples, _ner_hidden_layer._b, _ner_hidden_layer._gradb, "_ner_hidden_layer._b", iter);

    checkgradNer(examples, _nerrnn_left._lstm_output._W1, _nerrnn_left._lstm_output._gradW1, "_nerrnn_left._lstm_output._W1", iter);
    checkgradNer(examples, _nerrnn_left._lstm_output._W2, _nerrnn_left._lstm_output._gradW2, "_nerrnn_left._lstm_output._W2", iter);
    checkgradNer(examples, _nerrnn_left._lstm_output._W3, _nerrnn_left._lstm_output._gradW3, "_nerrnn_left._lstm_output._W3", iter);
    checkgradNer(examples, _nerrnn_left._lstm_output._b, _nerrnn_left._lstm_output._gradb, "_nerrnn_left._lstm_output._b", iter);
    checkgradNer(examples, _nerrnn_left._lstm_input._W1, _nerrnn_left._lstm_input._gradW1, "_nerrnn_left._lstm_input._W1", iter);
    checkgradNer(examples, _nerrnn_left._lstm_input._W2, _nerrnn_left._lstm_input._gradW2, "_nerrnn_left._lstm_input._W2", iter);
    checkgradNer(examples, _nerrnn_left._lstm_input._W3, _nerrnn_left._lstm_input._gradW3, "_nerrnn_left._lstm_input._W3", iter);
    checkgradNer(examples, _nerrnn_left._lstm_input._b, _nerrnn_left._lstm_input._gradb, "_nerrnn_left._lstm_input._b", iter);
    checkgradNer(examples, _nerrnn_left._lstm_forget._W1, _nerrnn_left._lstm_forget._gradW1, "_nerrnn_left._lstm_forget._W1", iter);
    checkgradNer(examples, _nerrnn_left._lstm_forget._W2, _nerrnn_left._lstm_forget._gradW2, "_nerrnn_left._lstm_forget._W2", iter);
    checkgradNer(examples, _nerrnn_left._lstm_forget._W3, _nerrnn_left._lstm_forget._gradW3, "_nerrnn_left._lstm_forget._W3", iter);
    checkgradNer(examples, _nerrnn_left._lstm_forget._b, _nerrnn_left._lstm_forget._gradb, "_nerrnn_left._lstm_forget._b", iter);
    checkgradNer(examples, _nerrnn_left._lstm_cell._WL, _nerrnn_left._lstm_cell._gradWL, "_nerrnn_left._lstm_cell._WL", iter);
    checkgradNer(examples, _nerrnn_left._lstm_cell._WR, _nerrnn_left._lstm_cell._gradWR, "_nerrnn_left._lstm_cell._WR", iter);
    checkgradNer(examples, _nerrnn_left._lstm_cell._b, _nerrnn_left._lstm_cell._gradb, "_nerrnn_left._lstm_cell._b", iter);

    checkgradNer(examples, _nerrnn_right._lstm_output._W1, _nerrnn_right._lstm_output._gradW1, "_nerrnn_right._lstm_output._W1", iter);
    checkgradNer(examples, _nerrnn_right._lstm_output._W2, _nerrnn_right._lstm_output._gradW2, "_nerrnn_right._lstm_output._W2", iter);
    checkgradNer(examples, _nerrnn_right._lstm_output._W3, _nerrnn_right._lstm_output._gradW3, "_nerrnn_right._lstm_output._W3", iter);
    checkgradNer(examples, _nerrnn_right._lstm_output._b, _nerrnn_right._lstm_output._gradb, "_nerrnn_right._lstm_output._b", iter);
    checkgradNer(examples, _nerrnn_right._lstm_input._W1, _nerrnn_right._lstm_input._gradW1, "_nerrnn_right._lstm_input._W1", iter);
    checkgradNer(examples, _nerrnn_right._lstm_input._W2, _nerrnn_right._lstm_input._gradW2, "_nerrnn_right._lstm_input._W2", iter);
    checkgradNer(examples, _nerrnn_right._lstm_input._W3, _nerrnn_right._lstm_input._gradW3, "_nerrnn_right._lstm_input._W3", iter);
    checkgradNer(examples, _nerrnn_right._lstm_input._b, _nerrnn_right._lstm_input._gradb, "_nerrnn_right._lstm_input._b", iter);
    checkgradNer(examples, _nerrnn_right._lstm_forget._W1, _nerrnn_right._lstm_forget._gradW1, "_nerrnn_right._lstm_forget._W1", iter);
    checkgradNer(examples, _nerrnn_right._lstm_forget._W2, _nerrnn_right._lstm_forget._gradW2, "_nerrnn_right._lstm_forget._W2", iter);
    checkgradNer(examples, _nerrnn_right._lstm_forget._W3, _nerrnn_right._lstm_forget._gradW3, "_nerrnn_right._lstm_forget._W3", iter);
    checkgradNer(examples, _nerrnn_right._lstm_forget._b, _nerrnn_right._lstm_forget._gradb, "_nerrnn_right._lstm_forget._b", iter);
    checkgradNer(examples, _nerrnn_right._lstm_cell._WL, _nerrnn_right._lstm_cell._gradWL, "_nerrnn_right._lstm_cell._WL", iter);
    checkgradNer(examples, _nerrnn_right._lstm_cell._WR, _nerrnn_right._lstm_cell._gradWR, "_nerrnn_right._lstm_cell._WR", iter);
    checkgradNer(examples, _nerrnn_right._lstm_cell._b, _nerrnn_right._lstm_cell._gradb, "_nerrnn_right._lstm_cell._b", iter);

    checkgradNer(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgradNer(examples, _ner._E, _ner._gradE, "_ner._E", iter, _ner._indexers);
    checkgradNer(examples, _dep._E, _dep._gradE, "_dep._E", iter, _dep._indexers);
    checkgradNer(examples, _pos._E, _pos._gradE, "_pos._E", iter, _pos._indexers);
  }

  void checkgradsNerGlobal(vector<Prediction> & beam, Prediction& gold, int sentence_size, int iter) {

    checkgradNerGlobal(beam, gold, sentence_size, _ner_olayer_linear._W, _ner_olayer_linear._gradW, "_ner_olayer_linear._W", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _ner_olayer_linear._b, _ner_olayer_linear._gradb, "_ner_olayer_linear._b", iter);

    checkgradNerGlobal(beam, gold, sentence_size, _ner_hidden_layer._W, _ner_hidden_layer._gradW, "_ner_hidden_layer._W", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _ner_hidden_layer._b, _ner_hidden_layer._gradb, "_ner_hidden_layer._b", iter);

    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_output._W1, _nerrnn_left._lstm_output._gradW1, "_nerrnn_left._lstm_output._W1", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_output._W2, _nerrnn_left._lstm_output._gradW2, "_nerrnn_left._lstm_output._W2", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_output._W3, _nerrnn_left._lstm_output._gradW3, "_nerrnn_left._lstm_output._W3", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_output._b, _nerrnn_left._lstm_output._gradb, "_nerrnn_left._lstm_output._b", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_input._W1, _nerrnn_left._lstm_input._gradW1, "_nerrnn_left._lstm_input._W1", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_input._W2, _nerrnn_left._lstm_input._gradW2, "_nerrnn_left._lstm_input._W2", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_input._W3, _nerrnn_left._lstm_input._gradW3, "_nerrnn_left._lstm_input._W3", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_input._b, _nerrnn_left._lstm_input._gradb, "_nerrnn_left._lstm_input._b", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_forget._W1, _nerrnn_left._lstm_forget._gradW1, "_nerrnn_left._lstm_forget._W1", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_forget._W2, _nerrnn_left._lstm_forget._gradW2, "_nerrnn_left._lstm_forget._W2", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_forget._W3, _nerrnn_left._lstm_forget._gradW3, "_nerrnn_left._lstm_forget._W3", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_forget._b, _nerrnn_left._lstm_forget._gradb, "_nerrnn_left._lstm_forget._b", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_cell._WL, _nerrnn_left._lstm_cell._gradWL, "_nerrnn_left._lstm_cell._WL", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_cell._WR, _nerrnn_left._lstm_cell._gradWR, "_nerrnn_left._lstm_cell._WR", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_left._lstm_cell._b, _nerrnn_left._lstm_cell._gradb, "_nerrnn_left._lstm_cell._b", iter);

    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_output._W1, _nerrnn_right._lstm_output._gradW1, "_nerrnn_right._lstm_output._W1", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_output._W2, _nerrnn_right._lstm_output._gradW2, "_nerrnn_right._lstm_output._W2", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_output._W3, _nerrnn_right._lstm_output._gradW3, "_nerrnn_right._lstm_output._W3", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_output._b, _nerrnn_right._lstm_output._gradb, "_nerrnn_right._lstm_output._b", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_input._W1, _nerrnn_right._lstm_input._gradW1, "_nerrnn_right._lstm_input._W1", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_input._W2, _nerrnn_right._lstm_input._gradW2, "_nerrnn_right._lstm_input._W2", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_input._W3, _nerrnn_right._lstm_input._gradW3, "_nerrnn_right._lstm_input._W3", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_input._b, _nerrnn_right._lstm_input._gradb, "_nerrnn_right._lstm_input._b", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_forget._W1, _nerrnn_right._lstm_forget._gradW1, "_nerrnn_right._lstm_forget._W1", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_forget._W2, _nerrnn_right._lstm_forget._gradW2, "_nerrnn_right._lstm_forget._W2", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_forget._W3, _nerrnn_right._lstm_forget._gradW3, "_nerrnn_right._lstm_forget._W3", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_forget._b, _nerrnn_right._lstm_forget._gradb, "_nerrnn_right._lstm_forget._b", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_cell._WL, _nerrnn_right._lstm_cell._gradWL, "_nerrnn_right._lstm_cell._WL", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_cell._WR, _nerrnn_right._lstm_cell._gradWR, "_nerrnn_right._lstm_cell._WR", iter);
    checkgradNerGlobal(beam, gold, sentence_size, _nerrnn_right._lstm_cell._b, _nerrnn_right._lstm_cell._gradb, "_nerrnn_right._lstm_cell._b", iter);

    checkgradNerGlobal(beam, gold, sentence_size, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgradNerGlobal(beam, gold, sentence_size, _ner._E, _ner._gradE, "_ner._E", iter, _ner._indexers);
    checkgradNerGlobal(beam, gold, sentence_size, _dep._E, _dep._gradE, "_dep._E", iter, _dep._indexers);
    checkgradNerGlobal(beam, gold, sentence_size, _pos._E, _pos._gradE, "_pos._E", iter, _pos._indexers);
  }

  void checkgradsRel(const vector<Example>& examples, int iter) {

    checkgradRel(examples, _rel_output_layer._W, _rel_output_layer._gradW, "_rel_output_layer._W", iter);
    checkgradRel(examples, _rel_output_layer._b, _rel_output_layer._gradb, "_rel_output_layer._b", iter);

    checkgradRel(examples, _rel_hidden_layer._W, _rel_hidden_layer._gradW, "_rel_hidden_layer._W", iter);
    checkgradRel(examples, _rel_hidden_layer._b, _rel_hidden_layer._gradb, "_rel_hidden_layer._b", iter);

    checkgradRel(examples, _relrnn_bottomup._lstm_output._W1, _relrnn_bottomup._lstm_output._gradW1, "_relrnn_bottomup._lstm_output._W1", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_output._W2, _relrnn_bottomup._lstm_output._gradW2, "_relrnn_bottomup._lstm_output._W2", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_output._W3, _relrnn_bottomup._lstm_output._gradW3, "_relrnn_bottomup._lstm_output._W3", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_output._b, _relrnn_bottomup._lstm_output._gradb, "_relrnn_bottomup._lstm_output._b", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_input._W1, _relrnn_bottomup._lstm_input._gradW1, "_relrnn_bottomup._lstm_input._W1", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_input._W2, _relrnn_bottomup._lstm_input._gradW2, "_relrnn_bottomup._lstm_input._W2", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_input._W3, _relrnn_bottomup._lstm_input._gradW3, "_relrnn_bottomup._lstm_input._W3", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_input._b, _relrnn_bottomup._lstm_input._gradb, "_relrnn_bottomup._lstm_input._b", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_forget._W1, _relrnn_bottomup._lstm_forget._gradW1, "_relrnn_bottomup._lstm_forget._W1", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_forget._W2, _relrnn_bottomup._lstm_forget._gradW2, "_relrnn_bottomup._lstm_forget._W2", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_forget._W3, _relrnn_bottomup._lstm_forget._gradW3, "_relrnn_bottomup._lstm_forget._W3", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_forget._b, _relrnn_bottomup._lstm_forget._gradb, "_relrnn_bottomup._lstm_forget._b", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_cell._WL, _relrnn_bottomup._lstm_cell._gradWL, "_relrnn_bottomup._lstm_cell._WL", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_cell._WR, _relrnn_bottomup._lstm_cell._gradWR, "_relrnn_bottomup._lstm_cell._WR", iter);
    checkgradRel(examples, _relrnn_bottomup._lstm_cell._b, _relrnn_bottomup._lstm_cell._gradb, "_relrnn_bottomup._lstm_cell._b", iter);

    checkgradRel(examples, _relrnn_topdown._lstm_output._W1, _relrnn_topdown._lstm_output._gradW1, "_relrnn_topdown._lstm_output._W1", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_output._W2, _relrnn_topdown._lstm_output._gradW2, "_relrnn_topdown._lstm_output._W2", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_output._W3, _relrnn_topdown._lstm_output._gradW3, "_relrnn_topdown._lstm_output._W3", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_output._b, _relrnn_topdown._lstm_output._gradb, "_relrnn_topdown._lstm_output._b", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_input._W1, _relrnn_topdown._lstm_input._gradW1, "_relrnn_topdown._lstm_input._W1", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_input._W2, _relrnn_topdown._lstm_input._gradW2, "_relrnn_topdown._lstm_input._W2", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_input._W3, _relrnn_topdown._lstm_input._gradW3, "_relrnn_topdown._lstm_input._W3", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_input._b, _relrnn_topdown._lstm_input._gradb, "_relrnn_topdown._lstm_input._b", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_forget._W1, _relrnn_topdown._lstm_forget._gradW1, "_relrnn_topdown._lstm_forget._W1", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_forget._W2, _relrnn_topdown._lstm_forget._gradW2, "_relrnn_topdown._lstm_forget._W2", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_forget._W3, _relrnn_topdown._lstm_forget._gradW3, "_relrnn_topdown._lstm_forget._W3", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_forget._b, _relrnn_topdown._lstm_forget._gradb, "_relrnn_topdown._lstm_forget._b", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_cell._WL, _relrnn_topdown._lstm_cell._gradWL, "_relrnn_topdown._lstm_cell._WL", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_cell._WR, _relrnn_topdown._lstm_cell._gradWR, "_relrnn_topdown._lstm_cell._WR", iter);
    checkgradRel(examples, _relrnn_topdown._lstm_cell._b, _relrnn_topdown._lstm_cell._gradb, "_relrnn_topdown._lstm_cell._b", iter);


    checkgradRel(examples, _nerrnn_left._lstm_output._W1, _nerrnn_left._lstm_output._gradW1, "_nerrnn_left._lstm_output._W1", iter);
    checkgradRel(examples, _nerrnn_left._lstm_output._W2, _nerrnn_left._lstm_output._gradW2, "_nerrnn_left._lstm_output._W2", iter);
    checkgradRel(examples, _nerrnn_left._lstm_output._W3, _nerrnn_left._lstm_output._gradW3, "_nerrnn_left._lstm_output._W3", iter);
    checkgradRel(examples, _nerrnn_left._lstm_output._b, _nerrnn_left._lstm_output._gradb, "_nerrnn_left._lstm_output._b", iter);
    checkgradRel(examples, _nerrnn_left._lstm_input._W1, _nerrnn_left._lstm_input._gradW1, "_nerrnn_left._lstm_input._W1", iter);
    checkgradRel(examples, _nerrnn_left._lstm_input._W2, _nerrnn_left._lstm_input._gradW2, "_nerrnn_left._lstm_input._W2", iter);
    checkgradRel(examples, _nerrnn_left._lstm_input._W3, _nerrnn_left._lstm_input._gradW3, "_nerrnn_left._lstm_input._W3", iter);
    checkgradRel(examples, _nerrnn_left._lstm_input._b, _nerrnn_left._lstm_input._gradb, "_nerrnn_left._lstm_input._b", iter);
    checkgradRel(examples, _nerrnn_left._lstm_forget._W1, _nerrnn_left._lstm_forget._gradW1, "_nerrnn_left._lstm_forget._W1", iter);
    checkgradRel(examples, _nerrnn_left._lstm_forget._W2, _nerrnn_left._lstm_forget._gradW2, "_nerrnn_left._lstm_forget._W2", iter);
    checkgradRel(examples, _nerrnn_left._lstm_forget._W3, _nerrnn_left._lstm_forget._gradW3, "_nerrnn_left._lstm_forget._W3", iter);
    checkgradRel(examples, _nerrnn_left._lstm_forget._b, _nerrnn_left._lstm_forget._gradb, "_nerrnn_left._lstm_forget._b", iter);
    checkgradRel(examples, _nerrnn_left._lstm_cell._WL, _nerrnn_left._lstm_cell._gradWL, "_nerrnn_left._lstm_cell._WL", iter);
    checkgradRel(examples, _nerrnn_left._lstm_cell._WR, _nerrnn_left._lstm_cell._gradWR, "_nerrnn_left._lstm_cell._WR", iter);
    checkgradRel(examples, _nerrnn_left._lstm_cell._b, _nerrnn_left._lstm_cell._gradb, "_nerrnn_left._lstm_cell._b", iter);

    checkgradRel(examples, _nerrnn_right._lstm_output._W1, _nerrnn_right._lstm_output._gradW1, "_nerrnn_right._lstm_output._W1", iter);
    checkgradRel(examples, _nerrnn_right._lstm_output._W2, _nerrnn_right._lstm_output._gradW2, "_nerrnn_right._lstm_output._W2", iter);
    checkgradRel(examples, _nerrnn_right._lstm_output._W3, _nerrnn_right._lstm_output._gradW3, "_nerrnn_right._lstm_output._W3", iter);
    checkgradRel(examples, _nerrnn_right._lstm_output._b, _nerrnn_right._lstm_output._gradb, "_nerrnn_right._lstm_output._b", iter);
    checkgradRel(examples, _nerrnn_right._lstm_input._W1, _nerrnn_right._lstm_input._gradW1, "_nerrnn_right._lstm_input._W1", iter);
    checkgradRel(examples, _nerrnn_right._lstm_input._W2, _nerrnn_right._lstm_input._gradW2, "_nerrnn_right._lstm_input._W2", iter);
    checkgradRel(examples, _nerrnn_right._lstm_input._W3, _nerrnn_right._lstm_input._gradW3, "_nerrnn_right._lstm_input._W3", iter);
    checkgradRel(examples, _nerrnn_right._lstm_input._b, _nerrnn_right._lstm_input._gradb, "_nerrnn_right._lstm_input._b", iter);
    checkgradRel(examples, _nerrnn_right._lstm_forget._W1, _nerrnn_right._lstm_forget._gradW1, "_nerrnn_right._lstm_forget._W1", iter);
    checkgradRel(examples, _nerrnn_right._lstm_forget._W2, _nerrnn_right._lstm_forget._gradW2, "_nerrnn_right._lstm_forget._W2", iter);
    checkgradRel(examples, _nerrnn_right._lstm_forget._W3, _nerrnn_right._lstm_forget._gradW3, "_nerrnn_right._lstm_forget._W3", iter);
    checkgradRel(examples, _nerrnn_right._lstm_forget._b, _nerrnn_right._lstm_forget._gradb, "_nerrnn_right._lstm_forget._b", iter);
    checkgradRel(examples, _nerrnn_right._lstm_cell._WL, _nerrnn_right._lstm_cell._gradWL, "_nerrnn_right._lstm_cell._WL", iter);
    checkgradRel(examples, _nerrnn_right._lstm_cell._WR, _nerrnn_right._lstm_cell._gradWR, "_nerrnn_right._lstm_cell._WR", iter);
    checkgradRel(examples, _nerrnn_right._lstm_cell._b, _nerrnn_right._lstm_cell._gradb, "_nerrnn_right._lstm_cell._b", iter);

    checkgradRel(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgradRel(examples, _ner._E, _ner._gradE, "_ner._E", iter, _ner._indexers);
    checkgradRel(examples, _dep._E, _dep._gradE, "_dep._E", iter, _dep._indexers);
    checkgradRel(examples, _pos._E, _pos._gradE, "_pos._E", iter, _pos._indexers);
  }


  inline void globalnormalization_loss(Tensor<xpu, 2, dtype> output, const vector<int> &answer, Tensor<xpu, 2, dtype> loutput,
		  vector<Prediction> & beam, dtype zg, int step, Metric & eval, int batchsize = 1) {
    int dim1 = output.size(0), dim2 = output.size(1);
    int odim1 = loutput.size(0), odim2 = loutput.size(1);
    int labelsize = answer.size();

    loutput = 0.0;

    int optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
        if (optLabel < 0 || output[0][i] > output[0][optLabel])
          optLabel = i;
    }

    if (answer[optLabel] == 1)
      eval.correct_label_count++;
    eval.overall_label_count++;

    for (int i = 0; i < dim2; ++i) {
    	for(int beamIdx=0;beamIdx<beam.size();beamIdx++) {
    		if(i == beam[beamIdx].labels[step])
    			loutput[0][i] += beam[beamIdx].expsum;
    	}

    	loutput[0][i] /= zg;

    	loutput[0][i] += -answer[i];

        loutput[0][i] /= batchsize;
    }



  }


};

#endif /* SRC_PoolGRNNClassifier_H_ */
