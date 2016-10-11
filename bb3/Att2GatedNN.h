
#ifndef SRC_Att2GatedNN_H_
#define SRC_Att2GatedNN_H_

#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// two attention targets, three inputs
template<typename xpu>
class Att2GatedNN {
public:
  TriLayer<xpu> _reset1;
  TriLayer<xpu> _reset2;
  TriLayer<xpu> _reset3;
  TriLayer<xpu> _recursive_tilde;


  Tensor<xpu, 2, dtype> nx1;
  Tensor<xpu, 2, dtype> nx2;
  Tensor<xpu, 2, dtype> nx3;

  Tensor<xpu, 2, dtype> lrx1;
  Tensor<xpu, 2, dtype> lrx2;
  Tensor<xpu, 2, dtype> lrx3;

  Tensor<xpu, 2, dtype> lnx1;
  Tensor<xpu, 2, dtype> lnx2;
  Tensor<xpu, 2, dtype> lnx3;



public:
  Att2GatedNN() {
  }
  virtual ~Att2GatedNN() {
    // TODO Auto-generated destructor stub
  }

  inline void initial(int dimension, int attDim, int seed = 0) {
    _reset1.initial(dimension, dimension, attDim, attDim, false, seed, 1);
    _reset2.initial(dimension, dimension, attDim, attDim, false, seed + 10, 1);
    _reset3.initial(dimension, dimension, attDim, attDim, false, seed + 20, 1);
    _recursive_tilde.initial(dimension, dimension, dimension, dimension, false, seed + 70, 0);

    nx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    nx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    nx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);


    lrx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lrx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lrx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    lnx1 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lnx2 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lnx3 = NewTensor<xpu>(Shape2(1, dimension), d_zero);
  }


  inline void release() {
    _reset1.release();
    _reset2.release();
    _reset3.release();

    _recursive_tilde.release();

    FreeSpace(&nx1);
    FreeSpace(&nx2);
    FreeSpace(&nx3);

    FreeSpace(&lnx1);
    FreeSpace(&lnx2);
    FreeSpace(&lnx3);

    FreeSpace(&lrx1);
    FreeSpace(&lrx2);
    FreeSpace(&lrx3);
  }



public:

  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3,
		  Tensor<xpu, 2, dtype> a1, Tensor<xpu, 2, dtype> a2,
      Tensor<xpu, 2, dtype> rx1, Tensor<xpu, 2, dtype> rx2, Tensor<xpu, 2, dtype> rx3,
      Tensor<xpu, 2, dtype> y) {

    nx1 = 0.0;
    nx2 = 0.0;
    nx3 = 0.0;

    _reset1.ComputeForwardScore(x1, a1, a2, rx1);
    _reset2.ComputeForwardScore(x2, a1, a2, rx2);
    _reset3.ComputeForwardScore(x3, a1, a2, rx3);


    nx1 = rx1 * x1;
    nx2 = rx2 * x2;
    nx3 = rx3 * x3;

    _recursive_tilde.ComputeForwardScore(nx1, nx2, nx3, y);


  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3,
		  Tensor<xpu, 2, dtype> a1, Tensor<xpu, 2, dtype> a2,
      Tensor<xpu, 2, dtype> rx1, Tensor<xpu, 2, dtype> rx2, Tensor<xpu, 2, dtype> rx3,
      Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lx1, Tensor<xpu, 2, dtype> lx2, Tensor<xpu, 2, dtype> lx3,
	  	  Tensor<xpu, 2, dtype> la1, Tensor<xpu, 2, dtype> la2,
      bool bclear = false) {
    if (bclear){
      lx1 = 0.0; lx2 = 0.0; lx3 = 0.0; la1 = 0.0; la2 = 0.0;
    }

    nx1 = 0.0;
    nx2 = 0.0;
    nx3 = 0.0;


    lrx1 = 0.0;
    lrx2 = 0.0;
    lrx3 = 0.0;

    lnx1 = 0.0;
    lnx2 = 0.0;
    lnx3 = 0.0;

    nx1 = rx1 * x1;
    nx2 = rx2 * x2;
    nx3 = rx3 * x3;

    _recursive_tilde.ComputeBackwardLoss(nx1, nx2, nx3, y, ly, lnx1, lnx2, lnx3);

    lrx1 += lnx1 * x1;
    lx1 += lnx1 * rx1;

    lrx2 += lnx2 * x2;
    lx2 += lnx2 * rx2;

    lrx3 += lnx3 * x3;
    lx3 += lnx3 * rx3;

    _reset1.ComputeBackwardLoss(x1, a1, a2, rx1, lrx1, lx1, la1, la2);
    _reset2.ComputeBackwardLoss(x2, a1, a2, rx2, lrx2, lx2, la1, la2);
    _reset3.ComputeBackwardLoss(x3, a1, a2, rx3, lrx3, lx3, la1, la2);
  }


  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _reset1.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _reset2.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _reset3.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    _recursive_tilde.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }



};



#endif /* SRC_AttRecursiveGatedNN_H_ */
