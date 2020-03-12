#include <math.h>       /* asinh, exp, cosh */

void forward(double *u, const double *a, const double *uold, const double v0, const double *psi, const double *sigman, const double *sigmazx,
       double eta, double deltat, int n){
// TODO:
  int max_iter = 1000;
  double tol = 1e-8;
  double dx, dFdx;
  for (int i=0; i<n; i++) {
    u[i] = uold[i];
    int iter;
    for (iter=0; iter<max_iter; iter++) {
      double y = (u[i]-uold[i]) / deltat / 2.0 / v0 * exp(psi[i]/a[i]);
      double F = a[i] * asinh(y) * sigman[i] - sigmazx[i] + eta * (u[i]-uold[i]) / deltat;
      dFdx = a[i] / sqrt(1+y*y) / deltat / 2.0 / v0 * exp(psi[i]/a[i]) * sigman[i] + eta/deltat;
      dx = F / dFdx;
      // if (i==0)
      //   printf("%f, %f\n", dFdx, dx);
      if (abs(dx)/abs(u[i]) < tol){
        // printf("Variable %d, iter = %d,  residual = %f, dx = %f\n", i, iter, F, dx);
        break;
      }
      u[i] -= dx;
    }
    // printf("Variable %d, sigman[i] = %e, %e, error = %f, iter = %d, dFdx = %f\n", i, sigman[i], sigmazx[i], dx, iter, dFdx);
  }
}

void backward(
        double * grad_a, double *grad_uold, double *grad_psi, double *grad_sigman, double *grad_sigmazx, const double *grad_u, 
        const double *u, const double *a, const double *uold, const double v0, const double *psi, 
        const double *sigman, const double *sigmazx,
       double eta, double deltat, int n){

    for(int i=0;i<n;i++){
       
       double y = (u[i]-uold[i])/deltat / 2 / v0 * exp(psi[i]/a[i]) ; 
       double dFdx = a[i]/sqrt(1+y*y) / deltat / 2 / v0 * exp(psi[i]/a[i]) * sigman[i] + eta/deltat;
       double dFdu = - a[i]/sqrt(1+y*y) / deltat / 2 / v0 * exp(psi[i]/a[i]) * sigman[i] - eta/deltat;
       double dFda = asinh(y) * sigman[i] + 
              a[i]/sqrt(1+y*y) * (u[i]-uold[i])/deltat / 2 / v0 * (-psi[i]/a[i]/a[i])* exp(psi[i]/a[i]) * sigman[i];
       double dFdpsi = a[i]/sqrt(1+y*y)*(u[i]-uold[i])/deltat / 2 / v0 / a[i]* exp(psi[i]/a[i]) * sigman[i];
       double dFdtau = -1.0;
       double dFdsigma = a[i] * asinh(y);
       
       grad_a[i] = - grad_u[i] / dFdx * dFda;
       grad_uold[i] = - grad_u[i] / dFdx * dFdu;
       grad_psi[i] = - grad_u[i] / dFdx * dFdpsi;
       grad_sigman[i] = - grad_u[i] / dFdx * dFdsigma;
       grad_sigmazx[i] = - grad_u[i] / dFdx * dFdtau;
       
    }

}