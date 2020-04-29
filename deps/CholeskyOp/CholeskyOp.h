#include "adept.h"
#include "adept_arrays.h"
using namespace adept;
void factorize_forward(double *l, const double *a, int n){
  // https://rosettacode.org/wiki/Cholesky_decomposition
  for (int i=0;i<n;i++){
    double a11 = a[0], a21 = a[1], a31 = a[2],
           a12 = a[3], a22 = a[4], a32 = a[5],
           a13 = a[6], a23 = a[7], a33 = a[8];
    double l11 = sqrt(a11);
    double l21 = a21/l11;
    double l31 = a31/l11;
    double l22 = sqrt(a22-l21*l21);
    double l32 = (a32-l31*l21)/l22;
    double l33 = sqrt(a33-(l31*l31+l32*l32));
    l[0] = l11;
    l[1] = l22;
    l[2] = l33;
    l[3] = l21;
    l[4] = l31;
    l[5] = l32;
    a += 9;
    l += 6;
  }
}

void factorize_backward(
  double * grad_a,
  const double * grad_l, 
  const double *l, const double *a, int n){
  Stack stack;
  for (int i=0;i<n;i++){
    adouble a11 = a[0], a21 = a[1], a31 = a[2],
           a12 = a[3], a22 = a[4], a32 = a[5],
           a13 = a[6], a23 = a[7], a33 = a[8];
    stack.new_recording();
    adouble l11 = sqrt(a11);
    adouble l21 = a21/l11;
    adouble l31 = a31/l11;
    adouble l22 = sqrt(a22-l21*l21);
    adouble l32 = (a32-l31*l21)/l22;
    adouble l33 = sqrt(a33-(l31*l31+l32*l32));
    adouble L =  grad_l[0] * l11+grad_l[1] * l22+grad_l[2] * l33+
        grad_l[3] * l21+grad_l[4] * l31+grad_l[5] * l32;
    L.set_gradient(1.0);
    stack.compute_adjoint();
    grad_a[0] = a11.get_gradient();
    grad_a[1] = a21.get_gradient();
    grad_a[2] = a31.get_gradient();
    grad_a[3] = a12.get_gradient();
    grad_a[4] = a22.get_gradient();
    grad_a[5] = a32.get_gradient();
    grad_a[6] = a13.get_gradient();
    grad_a[7] = a23.get_gradient();
    grad_a[8] = a33.get_gradient();
    a += 9;
    grad_a += 9;
    l += 6;
    grad_l += 6;
  }
}

void outerproduct_forward(double *a, const double *l, int n){
  // https://rosettacode.org/wiki/Cholesky_decomposition
  for (int i=0;i<n;i++){
    Matrix33 L;
    double l11 = l[0];
    double l22 = l[1];
    double l33 = l[2];
    double l21 = l[3];
    double l31 = l[4];
    double l32 = l[5];
    L << l11, 0.0, 0.0,
         l21, l22, 0.0,
         l31, l32, l33;
    Matrix33 A = L ** L.T();
    a[0] = A(0,0); a[1] = A(0,1); a[2] = A(0,2);
    a[3] = A(1,0); a[4] = A(1,1); a[5] = A(1,2);
    a[6] = A(2,0); a[7] = A(2,1); a[8] = A(2,2);
    a += 9;
    l += 6;
  }
}

void outerproduct_backward(
  double * grad_l,
  const double * grad_a,
   const double *a,  const double *l, int n){
  Stack stack;
  for (int i=0;i<n;i++){
    aMatrix33 L;
    adouble l11 = l[0];
    adouble l22 = l[1];
    adouble l33 = l[2];
    adouble l21 = l[3];
    adouble l31 = l[4];
    adouble l32 = l[5];
    stack.new_recording();
    L(0, 0) = l11; L(0, 1) = 0.0; L(0,2) = 0.0;
    L(1, 0) = l21; L(1, 1) = l22; L(1,2) = 0.0;
    L(2, 0) = l31; L(2, 1) = l32; L(2,2) = l33;
    aMatrix33 A = L ** L.T();
    adouble obj = grad_a[0] * A(0,0)+ grad_a[1] * A(0,1)+ grad_a[2] * A(0,2)+
        grad_a[3] * A(1,0)+ grad_a[4] * A(1,1)+ grad_a[5] * A(1,2)+
        grad_a[6] * A(2,0)+ grad_a[7] * A(2,1)+ grad_a[8] * A(2,2);
    obj.set_gradient(1.0);
    stack.compute_adjoint();
    grad_l[0] = l11.get_gradient();
    grad_l[1] = l22.get_gradient();
    grad_l[2] = l33.get_gradient();
    grad_l[3] = l21.get_gradient();
    grad_l[4] = l31.get_gradient();
    grad_l[5] = l32.get_gradient();
    a += 9;
    l += 6;
    grad_a += 9;
    grad_l += 6;
  }
}