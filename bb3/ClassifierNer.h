
#ifndef SRC_ClassifierNer_H_
#define SRC_ClassifierNer_H_

#include <iostream>

#include <assert.h>
#include "N3L.h"
#include "NerExample.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class ClassifierNer {
public:
	ClassifierNer() {

  }
  ~ClassifierNer() {

  }

public:
  LookupTable<xpu> _words;
  LookupTable<xpu> _pos;
  LookupTable<xpu> _ner;

  int _wordDim;
  int _otherDim;
  int _nerrnn_inputsize;
  int _nerrnn_outputsize;
  int _hidden_inputsize;
  int _hidden_outputsize;
  int _labelSize;


  LSTM<xpu> _nerrnn_left;
  LSTM<xpu> _nerrnn_right;
  UniLayer<xpu> _hidden_layer;
  UniLayer<xpu> _olayer_linear;

  Metric _eval;

  Options options;


public:

  inline void init(Options options) {

	  this->options = options;

	  _wordDim = options.wordEmbSize;
	  _otherDim = options.otherEmbSize;
	  _nerrnn_inputsize = _wordDim+_otherDim; // word+pos
	  _nerrnn_outputsize = options.rnnHiddenSize;
	  _hidden_inputsize = _nerrnn_outputsize*2 + _otherDim; // birnn+ner
	  _hidden_outputsize = options.hiddenSize;
	  _labelSize = MAX_ENTITY;


	  _nerrnn_left.initial(_nerrnn_outputsize, _nerrnn_inputsize, true, 10);
	  _nerrnn_right.initial(_nerrnn_outputsize, _nerrnn_inputsize, false, 20);
	  _hidden_layer.initial(_hidden_outputsize, _hidden_inputsize, true, 30, 0);
	  _olayer_linear.initial(_labelSize, _hidden_outputsize, true, 40, 2);


    	cout<<"ClassifierNer initial"<<endl;
  }

  inline void release() {
    _words.release();
    _pos.release();
    _ner.release();

    _nerrnn_left.release();
    _nerrnn_left.release();
    _hidden_layer.release();
    _olayer_linear.release();
  }

  inline dtype process(const vector<NerExample>& examples, int iter) {
    _eval.reset();
    int example_num = examples.size();
    dtype cost = 0.0;
    for (int count = 0; count < example_num; count++) {
    	const NerExample& example = examples[count];

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

      hidden_input = NewTensor<xpu>(Shape2(1, _hidden_inputsize), 0.0);
      hidden_inputLoss = NewTensor<xpu>(Shape2(1, _hidden_inputsize), 0.0);

      hidden_output = NewTensor<xpu>(Shape2(1, _hidden_outputsize), 0.0);
      hidden_outputLoss = NewTensor<xpu>(Shape2(1, _hidden_outputsize), 0.0);

      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

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

      _hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

      _olayer_linear.ComputeForwardScore(hidden_output, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(hidden_output, output, outputLoss, hidden_outputLoss);

      _hidden_layer.ComputeBackwardLoss(hidden_input, hidden_output, hidden_outputLoss, hidden_inputLoss);

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

  int predict(const NerExample& example) {

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

	      hidden_input = NewTensor<xpu>(Shape2(1, _hidden_inputsize), 0.0);

	      hidden_output = NewTensor<xpu>(Shape2(1, _hidden_outputsize), 0.0);

	      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);


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

	      _hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

	      _olayer_linear.ComputeForwardScore(hidden_output, output);

		// decode algorithm
	      vector<dtype> results;
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

  dtype computeScore(const NerExample& example) {

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

	      hidden_input = NewTensor<xpu>(Shape2(1, _hidden_inputsize), 0.0);

	      hidden_output = NewTensor<xpu>(Shape2(1, _hidden_outputsize), 0.0);

	      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);


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

	      _hidden_layer.ComputeForwardScore(hidden_input, hidden_output);

	      _olayer_linear.ComputeForwardScore(hidden_output, output);

		// decode algorithm
	      dtype cost = softmax_cost(output, example.m_labels);

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

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
	  _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _hidden_layer.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _nerrnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _nerrnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);


    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	_pos.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _ner.updateAdaGrad(nnRegular, adaAlpha, adaEps);

  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<NerExample>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
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
      NerExample oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      NerExample oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<NerExample>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols, idThirds;
    idRows.clear();
    idCols.clear();
    idThirds.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int i = 0; i < Wd.size(1); i++)
      idCols.push_back(i);
    for (int i = 0; i < Wd.size(2); i++)
      idThirds.push_back(i);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());
    random_shuffle(idThirds.begin(), idThirds.end());

    int check_i = idRows[0], check_j = idCols[0], check_k = idThirds[0];

    dtype orginValue = Wd[check_i][check_j][check_k];

    Wd[check_i][check_j][check_k] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      NerExample oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j][check_k] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      NerExample oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j][check_k];

    printf("Iteration %d, Checking gradient for %s[%d][%d][%d]:\t", iter, mark.c_str(), check_i, check_j, check_k);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j][check_k] = orginValue;
  }

  void checkgrad(const vector<NerExample>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
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
      NerExample oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      NerExample oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<NerExample>& examples, int iter) {

    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

    checkgrad(examples, _hidden_layer._W, _hidden_layer._gradW, "_hidden_layer._W", iter);
    checkgrad(examples, _hidden_layer._b, _hidden_layer._gradb, "_hidden_layer._b", iter);

    checkgrad(examples, _nerrnn_left._lstm_output._W1, _nerrnn_left._lstm_output._gradW1, "_nerrnn_left._lstm_output._W1", iter);
    checkgrad(examples, _nerrnn_left._lstm_output._W2, _nerrnn_left._lstm_output._gradW2, "_nerrnn_left._lstm_output._W2", iter);
    checkgrad(examples, _nerrnn_left._lstm_output._W3, _nerrnn_left._lstm_output._gradW3, "_nerrnn_left._lstm_output._W3", iter);
    checkgrad(examples, _nerrnn_left._lstm_output._b, _nerrnn_left._lstm_output._gradb, "_nerrnn_left._lstm_output._b", iter);
    checkgrad(examples, _nerrnn_left._lstm_input._W1, _nerrnn_left._lstm_input._gradW1, "_nerrnn_left._lstm_input._W1", iter);
    checkgrad(examples, _nerrnn_left._lstm_input._W2, _nerrnn_left._lstm_input._gradW2, "_nerrnn_left._lstm_input._W2", iter);
    checkgrad(examples, _nerrnn_left._lstm_input._W3, _nerrnn_left._lstm_input._gradW3, "_nerrnn_left._lstm_input._W3", iter);
    checkgrad(examples, _nerrnn_left._lstm_input._b, _nerrnn_left._lstm_input._gradb, "_nerrnn_left._lstm_input._b", iter);
    checkgrad(examples, _nerrnn_left._lstm_forget._W1, _nerrnn_left._lstm_forget._gradW1, "_nerrnn_left._lstm_forget._W1", iter);
    checkgrad(examples, _nerrnn_left._lstm_forget._W2, _nerrnn_left._lstm_forget._gradW2, "_nerrnn_left._lstm_forget._W2", iter);
    checkgrad(examples, _nerrnn_left._lstm_forget._W3, _nerrnn_left._lstm_forget._gradW3, "_nerrnn_left._lstm_forget._W3", iter);
    checkgrad(examples, _nerrnn_left._lstm_forget._b, _nerrnn_left._lstm_forget._gradb, "_nerrnn_left._lstm_forget._b", iter);
    checkgrad(examples, _nerrnn_left._lstm_cell._WL, _nerrnn_left._lstm_cell._gradWL, "_nerrnn_left._lstm_cell._WL", iter);
    checkgrad(examples, _nerrnn_left._lstm_cell._WR, _nerrnn_left._lstm_cell._gradWR, "_nerrnn_left._lstm_cell._WR", iter);
    checkgrad(examples, _nerrnn_left._lstm_cell._b, _nerrnn_left._lstm_cell._gradb, "_nerrnn_left._lstm_cell._b", iter);

    checkgrad(examples, _nerrnn_right._lstm_output._W1, _nerrnn_right._lstm_output._gradW1, "_nerrnn_right._lstm_output._W1", iter);
    checkgrad(examples, _nerrnn_right._lstm_output._W2, _nerrnn_right._lstm_output._gradW2, "_nerrnn_right._lstm_output._W2", iter);
    checkgrad(examples, _nerrnn_right._lstm_output._W3, _nerrnn_right._lstm_output._gradW3, "_nerrnn_right._lstm_output._W3", iter);
    checkgrad(examples, _nerrnn_right._lstm_output._b, _nerrnn_right._lstm_output._gradb, "_nerrnn_right._lstm_output._b", iter);
    checkgrad(examples, _nerrnn_right._lstm_input._W1, _nerrnn_right._lstm_input._gradW1, "_nerrnn_right._lstm_input._W1", iter);
    checkgrad(examples, _nerrnn_right._lstm_input._W2, _nerrnn_right._lstm_input._gradW2, "_nerrnn_right._lstm_input._W2", iter);
    checkgrad(examples, _nerrnn_right._lstm_input._W3, _nerrnn_right._lstm_input._gradW3, "_nerrnn_right._lstm_input._W3", iter);
    checkgrad(examples, _nerrnn_right._lstm_input._b, _nerrnn_right._lstm_input._gradb, "_nerrnn_right._lstm_input._b", iter);
    checkgrad(examples, _nerrnn_right._lstm_forget._W1, _nerrnn_right._lstm_forget._gradW1, "_nerrnn_right._lstm_forget._W1", iter);
    checkgrad(examples, _nerrnn_right._lstm_forget._W2, _nerrnn_right._lstm_forget._gradW2, "_nerrnn_right._lstm_forget._W2", iter);
    checkgrad(examples, _nerrnn_right._lstm_forget._W3, _nerrnn_right._lstm_forget._gradW3, "_nerrnn_right._lstm_forget._W3", iter);
    checkgrad(examples, _nerrnn_right._lstm_forget._b, _nerrnn_right._lstm_forget._gradb, "_nerrnn_right._lstm_forget._b", iter);
    checkgrad(examples, _nerrnn_right._lstm_cell._WL, _nerrnn_right._lstm_cell._gradWL, "_nerrnn_right._lstm_cell._WL", iter);
    checkgrad(examples, _nerrnn_right._lstm_cell._WR, _nerrnn_right._lstm_cell._gradWR, "_nerrnn_right._lstm_cell._WR", iter);
    checkgrad(examples, _nerrnn_right._lstm_cell._b, _nerrnn_right._lstm_cell._gradb, "_nerrnn_right._lstm_cell._b", iter);

    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
    checkgrad(examples, _ner._E, _ner._gradE, "_ner._E", iter, _ner._indexers);
    checkgrad(examples, _pos._E, _pos._gradE, "_pos._E", iter, _pos._indexers);
  }


};

#endif /* SRC_PoolGRNNClassifier_H_ */
