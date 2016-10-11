
#ifndef CNN_H_
#define CNN_H_

// A standard convolutional neural network

#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "UniLayer.h"
#include "Attention_ZhouACL2016.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class CNN {
public:
  UniLayer<xpu> _kernel;
  int _outputsize;
  int _inputsize;
  int _windowsize;
  int _kernelinputsize;
  int _poolType;

  Tensor<xpu, 2, dtype> _null, _nullLoss, _eg2null;

  AttentionZhouACL2016<xpu> _att;

public:
  CNN() {
	  _outputsize = 0;
	  _inputsize= 0;
	  _windowsize= 0;
	  _kernelinputsize= 0;
	  _poolType = 0;
  }

  // outputsize is the dimension of this CNN
  // inputsize is the dimension of one element 'x' of the sequence
  // if windowsize is 1, the input of kernel will be x[i-1],x[i],x[i+1]
  // poolType: 0-max, 5-att
  inline void initial(int outputsize, int inputsize, int windowsize, int poolType=0, int funcType = 0, int seed = 0) {
	  int kernelInputsize = inputsize + 2*windowsize*inputsize;

	  _kernel.initial(outputsize, kernelInputsize, true, seed, funcType);

    _null = NewTensor<xpu>(Shape2(1, inputsize), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, inputsize), d_zero);
    _eg2null = NewTensor<xpu>(Shape2(1, inputsize), d_zero);

    _outputsize = outputsize;
    _inputsize = inputsize;
    _windowsize = windowsize;
    _kernelinputsize = kernelInputsize;
    _poolType = poolType;

    if(_poolType == 5) {
    	_att.initial(_outputsize, seed+13);
    }
  }

  inline void release() {
	  _kernel.release();

    FreeSpace(&_null);
    FreeSpace(&_nullLoss);
    FreeSpace(&_eg2null);

    if(_poolType == 5) {
    	_att.release();
    }
  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> y, Tensor<xpu, 3, dtype> kernelInputs, Tensor<xpu, 3, dtype> kernelOutputs, Tensor<xpu, 3, dtype> poolIndex) {
    y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    	// convolution


      for (int idx = 0; idx < seq_size; idx++) {

    	  int windowBegin = idx-_windowsize;
    	  int windowEnd = idx+_windowsize;

    	  int kernelIdx = 0;
    	  for(int i=windowBegin;i<=windowEnd;i++) {
    		  if(i<0 || i>=seq_size) {
    			  for(int j=0;j<_inputsize;j++) {
    				  kernelInputs[idx][0][kernelIdx] = _null[0][j];
    				  kernelIdx++;
    			  }
    		  } else {
    			  for(int j=0;j<_inputsize;j++) {
    				  kernelInputs[idx][0][kernelIdx] = x[i][0][j];
    				  kernelIdx++;
    			  }
    		  }
    	  }

    	  _kernel.ComputeForwardScore(kernelInputs[idx], kernelOutputs[idx]);

      }

      maxpool_forward(kernelOutputs, y, poolIndex);

  }

  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> y,
		  Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx,
		  Tensor<xpu, 3, dtype> kernelInputs, Tensor<xpu, 3, dtype> kernelOutputs,
		  Tensor<xpu, 3, dtype> poolIndex, bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (bclear)
      lx = 0.0;

    Tensor<xpu, 3, dtype> l_kernelOutputs= NewTensor<xpu>(Shape3(seq_size, 1, _outputsize), d_zero);
    pool_backward(ly, poolIndex, l_kernelOutputs);

    Tensor<xpu, 3, dtype> l_kernelInputs = NewTensor<xpu>(Shape3(seq_size, 1, _kernelinputsize), d_zero);
    for (int idx = 0; idx < seq_size; idx++) {

    	_kernel.ComputeBackwardLoss(kernelInputs[idx], kernelOutputs[idx], l_kernelOutputs[idx], l_kernelInputs[idx]);

  	  int windowBegin = idx-_windowsize;
  	  int windowEnd = idx+_windowsize;

  	  int kernelIdx = 0;
  	  for(int i=windowBegin;i<=windowEnd;i++) {
  		  if(i<0 || i>=seq_size) {
  			  for(int j=0;j<_inputsize;j++) {
  				  _nullLoss[0][j] += l_kernelInputs[idx][0][kernelIdx];
  				  kernelIdx++;
  			  }
  		  } else {
  			  for(int j=0;j<_inputsize;j++) {
  				  lx[i][0][j] += l_kernelInputs[idx][0][kernelIdx];
  				  kernelIdx++;
  			  }
  		  }
  	  }



    }

    FreeSpace(&l_kernelOutputs);
    FreeSpace(&l_kernelInputs);

  }

  // if attention, use this one
  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> y,
		  Tensor<xpu, 3, dtype> kernelInputs, Tensor<xpu, 3, dtype> kernelOutputs,
		  Tensor<xpu, 3, dtype> poolIndex,
		  Tensor<xpu, 3, dtype> M,
		  Tensor<xpu, 2, dtype> omegaM, Tensor<xpu, 2, dtype> exp_omegaM,
		  Tensor<xpu, 2, dtype> alpha, Tensor<xpu, 2, dtype> r
		  ) {
    y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    	// convolution


      for (int idx = 0; idx < seq_size; idx++) {

    	  int windowBegin = idx-_windowsize;
    	  int windowEnd = idx+_windowsize;

    	  int kernelIdx = 0;
    	  for(int i=windowBegin;i<=windowEnd;i++) {
    		  if(i<0 || i>=seq_size) {
    			  for(int j=0;j<_inputsize;j++) {
    				  kernelInputs[idx][0][kernelIdx] = _null[0][j];
    				  kernelIdx++;
    			  }
    		  } else {
    			  for(int j=0;j<_inputsize;j++) {
    				  kernelInputs[idx][0][kernelIdx] = x[i][0][j];
    				  kernelIdx++;
    			  }
    		  }
    	  }

    	  _kernel.ComputeForwardScore(kernelInputs[idx], kernelOutputs[idx]);

      }

      _att.ComputeForwardScore(kernelOutputs, M, omegaM, exp_omegaM, alpha, r, y);

  }

  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> y,
		  Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx,
		  Tensor<xpu, 3, dtype> kernelInputs, Tensor<xpu, 3, dtype> kernelOutputs,
		  Tensor<xpu, 3, dtype> poolIndex,
		  Tensor<xpu, 3, dtype> M, Tensor<xpu, 2, dtype> omegaM, Tensor<xpu, 2, dtype> exp_omegaM,
		  Tensor<xpu, 2, dtype> alpha, Tensor<xpu, 2, dtype> r,
		  bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (bclear)
      lx = 0.0;

    Tensor<xpu, 3, dtype> l_kernelOutputs= NewTensor<xpu>(Shape3(seq_size, 1, _outputsize), d_zero);
    _att.ComputeBackwardLoss(kernelOutputs, M, omegaM, exp_omegaM, alpha, r,
    		  y, ly, l_kernelOutputs);

    Tensor<xpu, 3, dtype> l_kernelInputs = NewTensor<xpu>(Shape3(seq_size, 1, _kernelinputsize), d_zero);
    for (int idx = 0; idx < seq_size; idx++) {

    	_kernel.ComputeBackwardLoss(kernelInputs[idx], kernelOutputs[idx], l_kernelOutputs[idx], l_kernelInputs[idx]);

  	  int windowBegin = idx-_windowsize;
  	  int windowEnd = idx+_windowsize;

  	  int kernelIdx = 0;
  	  for(int i=windowBegin;i<=windowEnd;i++) {
  		  if(i<0 || i>=seq_size) {
  			  for(int j=0;j<_inputsize;j++) {
  				  _nullLoss[0][j] += l_kernelInputs[idx][0][kernelIdx];
  				  kernelIdx++;
  			  }
  		  } else {
  			  for(int j=0;j<_inputsize;j++) {
  				  lx[i][0][j] += l_kernelInputs[idx][0][kernelIdx];
  				  kernelIdx++;
  			  }
  		  }
  	  }



    }

    FreeSpace(&l_kernelOutputs);
    FreeSpace(&l_kernelInputs);

  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
	  _nullLoss = _nullLoss + _null * regularizationWeight;
    _eg2null = _eg2null + _nullLoss * _nullLoss;
    _null = _null - _nullLoss * adaAlpha / F<nl_sqrt>(_eg2null + adaEps);

    _kernel.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    if(_poolType == 5) {
    	_att.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    }

    clearGrad();
  }

  inline void clearGrad() {
    _nullLoss = 0;
  }

};

#endif /* CNN_H_ */
