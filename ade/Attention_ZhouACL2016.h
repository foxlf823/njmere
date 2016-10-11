
#ifndef SRC_AttentionZhouACL2016_H_
#define SRC_AttentionZhouACL2016_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "Pooling.h"
#include "UniLayer.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// This is re-implementment of the work of Zhou et al. (2016) in ACL.
// Attention-Based Bidirectional LSTM Networks for Relation Classification

template<typename xpu>
class AttentionZhouACL2016 {

public:
	Tensor<xpu, 2, dtype> _omega;
	Tensor<xpu, 2, dtype> _gradOmega;
	Tensor<xpu, 2, dtype> _eg2Omega;

	int _dw = 0;


  AttentionZhouACL2016() {
  }

  inline void initial(int inputAndOutputSize, int seed = 0) {
	  _dw = inputAndOutputSize;


	dtype bound = sqrt(6.0 / (_dw + 1));

	_omega = NewTensor<xpu>(Shape2(1, _dw), d_zero);
	_gradOmega = NewTensor<xpu>(Shape2(1, _dw), d_zero);
	_eg2Omega = NewTensor<xpu>(Shape2(1, _dw), d_zero);


	random(_omega, -1.0 * bound, 1.0 * bound, seed);
  }


  inline void release() {
	    FreeSpace(&_omega);
	    FreeSpace(&_gradOmega);
	    FreeSpace(&_eg2Omega);
  }


public:

  // assume that all the vector is row vector, so the formulas should be adjusted.
  inline void ComputeForwardScore(std::vector<Tensor<xpu, 2, dtype> >& H, std::vector<Tensor<xpu, 2, dtype> >& M,
		  Tensor<xpu, 2, dtype> omegaM, Tensor<xpu, 2, dtype> exp_omegaM,
		  Tensor<xpu, 2, dtype> alpha, Tensor<xpu, 2, dtype> r,
		  Tensor<xpu, 2, dtype> hStar) {

    int seq_size = H.size();
    if(seq_size == 0) return;



    for(int i=0;i<seq_size;i++) {
    	M[i] = F<nl_tanh>(H[i]); // M = tanh(H)
    	for(int j=0;j<_dw;j++)
    		omegaM[0][i] += _omega[0][j]*M[i][0][j]; // dot(_omega, M[i].T()); // t(w)*M
    }

    // alpha = softmax(t(w)*M)
/*    int optLabel = -1;
    for (int i = 0; i < seq_size; ++i) {
        if (optLabel < 0 || omegaM[0][i] > omegaM[0][optLabel])
          optLabel = i;
    }*/
    dtype sum = 0;
    //dtype maxScore = omegaM[0][optLabel];
    for (int i = 0; i < seq_size; ++i) {
    	exp_omegaM[0][i] = exp(omegaM[0][i] /*- maxScore*/);
    	sum += exp_omegaM[0][i];
    }
    for (int i = 0; i < seq_size; ++i) {
    	alpha[0][i] = exp_omegaM[0][i]/sum;
    }

    // r = H*t(alpha)
    for(int i = 0; i < _dw; ++i) {
    	for(int j=0; j<seq_size; j++) {
    		r[0][i] += H[j][0][i]*alpha[0][j];
    	}
    }

    // h* = tanh(r)
    hStar = F<nl_tanh>(r);

/*    for(int i = 0; i < _dw; ++i) {
    	for(int j=0;j<seq_size;j++)
    		hStar[0][i] += H[j][0][i];
    }*/

  }



  inline void ComputeBackwardLoss(std::vector<Tensor<xpu, 2, dtype> >& H, std::vector<Tensor<xpu, 2, dtype> >& M,
		  Tensor<xpu, 2, dtype> omegaM, Tensor<xpu, 2, dtype> exp_omegaM,
		  Tensor<xpu, 2, dtype> alpha, Tensor<xpu, 2, dtype> r,
		  Tensor<xpu, 2, dtype> hStar, Tensor<xpu, 2, dtype> lhStar,
		  std::vector<Tensor<xpu, 2, dtype> >& lH, bool bclear = false) {

	  int seq_size = H.size();
	  if(seq_size == 0) return;

    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
    	  lH[idx] = 0.0;
      }
    }



    Tensor<xpu, 2, dtype> lr = NewTensor<xpu>(Shape2(r.size(0), r.size(1)), d_zero);
    lr += lhStar * F<nl_dtanh>(hStar);

    Tensor<xpu, 2, dtype> lalpha = NewTensor<xpu>(Shape2(alpha.size(0), alpha.size(1)), d_zero);
    for(int i = 0; i < _dw; ++i) {
    	for(int j=0; j<seq_size; j++) {
    		lH[j][0][i] += lr[0][i] * alpha[0][j];
    		lalpha[0][j] += lr[0][i] * H[j][0][i];
    	}
    }

    Tensor<xpu, 2, dtype> lOmegaM = NewTensor<xpu>(Shape2(omegaM.size(0), omegaM.size(1)), d_zero);
    for (int i = 0; i < seq_size; ++i) {
    	for(int j=0;j<seq_size;j++) {
    		if(j==i)
    			lOmegaM[0][i] += lalpha[0][j] * alpha[0][j] * (1-alpha[0][j]);
    		else
    			lOmegaM[0][i] += lalpha[0][j] * (-alpha[0][j]*alpha[0][i]);
    	}
    }

    std::vector<Tensor<xpu, 2, dtype> > lM(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      lM[idx] = NewTensor<xpu>(Shape2(M[idx].size(0), M[idx].size(1)), d_zero);
    }
    for(int i=0;i<seq_size;i++) {
		_gradOmega += lOmegaM[0][i] * M[i];
		lM[i] += lOmegaM[0][i] * _omega;
		lH[i] += lM[i] * F<nl_dtanh>(M[i]);
	}


    FreeSpace(&lr);
    FreeSpace(&lalpha);
    FreeSpace(&lOmegaM);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(lM[idx]));
    }

/*    for(int i = 0; i < _dw; ++i) {
    	for(int j=0;j<seq_size;j++)
    		lH[j][0][i] += lhStar[0][i];
    }*/



  }


  inline void ComputeForwardScore(Tensor<xpu, 3, dtype>  H, Tensor<xpu, 3, dtype>  M,
		  Tensor<xpu, 2, dtype> omegaM, Tensor<xpu, 2, dtype> exp_omegaM,
		  Tensor<xpu, 2, dtype> alpha, Tensor<xpu, 2, dtype> r,
		  Tensor<xpu, 2, dtype> hStar) {

    int seq_size = H.size(0);
    if(seq_size == 0) return;



    for(int i=0;i<seq_size;i++) {
    	M[i] = F<nl_tanh>(H[i]); // M = tanh(H)
    	for(int j=0;j<_dw;j++)
    		omegaM[0][i] += _omega[0][j]*M[i][0][j]; // dot(_omega, M[i].T()); // t(w)*M
    }

    // alpha = softmax(t(w)*M)
    dtype sum = 0;
    for (int i = 0; i < seq_size; ++i) {
    	exp_omegaM[0][i] = exp(omegaM[0][i] );
    	sum += exp_omegaM[0][i];
    }
    for (int i = 0; i < seq_size; ++i) {
    	alpha[0][i] = exp_omegaM[0][i]/sum;
    }

    // r = H*t(alpha)
    for(int i = 0; i < _dw; ++i) {
    	for(int j=0; j<seq_size; j++) {
    		r[0][i] += H[j][0][i]*alpha[0][j];
    	}
    }

    // h* = tanh(r)
    hStar = F<nl_tanh>(r);


  }



  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> H, Tensor<xpu, 3, dtype> M,
		  Tensor<xpu, 2, dtype> omegaM, Tensor<xpu, 2, dtype> exp_omegaM,
		  Tensor<xpu, 2, dtype> alpha, Tensor<xpu, 2, dtype> r,
		  Tensor<xpu, 2, dtype> hStar, Tensor<xpu, 2, dtype> lhStar,
		  Tensor<xpu, 3, dtype> lH, bool bclear = false) {

	  int seq_size = H.size(0);
	  if(seq_size == 0) return;

    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
    	  lH[idx] = 0.0;
      }
    }



    Tensor<xpu, 2, dtype> lr = NewTensor<xpu>(Shape2(r.size(0), r.size(1)), d_zero);
    lr += lhStar * F<nl_dtanh>(hStar);

    Tensor<xpu, 2, dtype> lalpha = NewTensor<xpu>(Shape2(alpha.size(0), alpha.size(1)), d_zero);
    for(int i = 0; i < _dw; ++i) {
    	for(int j=0; j<seq_size; j++) {
    		lH[j][0][i] += lr[0][i] * alpha[0][j];
    		lalpha[0][j] += lr[0][i] * H[j][0][i];
    	}
    }

    Tensor<xpu, 2, dtype> lOmegaM = NewTensor<xpu>(Shape2(omegaM.size(0), omegaM.size(1)), d_zero);
    for (int i = 0; i < seq_size; ++i) {
    	for(int j=0;j<seq_size;j++) {
    		if(j==i)
    			lOmegaM[0][i] += lalpha[0][j] * alpha[0][j] * (1-alpha[0][j]);
    		else
    			lOmegaM[0][i] += lalpha[0][j] * (-alpha[0][j]*alpha[0][i]);
    	}
    }

    std::vector<Tensor<xpu, 2, dtype> > lM(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      lM[idx] = NewTensor<xpu>(Shape2(M[idx].size(0), M[idx].size(1)), d_zero);
    }
    for(int i=0;i<seq_size;i++) {
		_gradOmega += lOmegaM[0][i] * M[i];
		lM[i] += lOmegaM[0][i] * _omega;
		lH[i] += lM[i] * F<nl_dtanh>(M[i]);
	}


    FreeSpace(&lr);
    FreeSpace(&lalpha);
    FreeSpace(&lOmegaM);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(lM[idx]));
    }


  }



  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradOmega = _gradOmega + _omega * regularizationWeight;
    _eg2Omega= _eg2Omega + _gradOmega * _gradOmega;
    _omega = _omega - _gradOmega * adaAlpha / F<nl_sqrt>(_eg2Omega + adaEps);

    clearGrad();
  }

  inline void clearGrad() {
    _gradOmega = 0;
  }


};

#endif
