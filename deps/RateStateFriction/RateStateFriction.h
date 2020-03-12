#include <math.h>       /* asinh, exp, cosh */

void forward(double *u, const double *a, const double *uold, const double v0, const double *psi, const double *sigmazx, const double *sigmazy,
       double eta, double deltat, int n){
// TODO:
  int max_iter = 1000;
  double tol = 1e-16;
  for (int i=0; i<n; i++) {
    u[i] = uold[i];
    for (int iter=0; iter<max_iter; iter++) {
      double y = (u[i]-uold[i]) / deltat / 2.0 / v0 * exp(psi[i]/a[i]);
      double F = a[i] * asinh(y) * sigmazx[i] - sigmazy[i] + eta * (u[i]-uold[i]) / deltat;
      double dFdx = a[i] / sqrt(1+y*y) / deltat / 2.0 / v0 * exp(psi[i]/a[i]) * sigmazx[i] + eta/deltat;
      double dx = F / dFdx;
      if (abs(dx) < tol){
        // printf("Variable %d, iter = %d,  residual = %f, dx = %f\n", i, iter, F, dx);
        break;
      }
      u[i] -= dx;
    }
    // printf("Variable %d, iter = %d\n", i, it);
  }
}

void backward(
        double * grad_a, double *grad_uold, double *grad_psi, double *grad_sigmazx, double *grad_sigmazy, const double *grad_u, 
        const double *u, const double *a, const double *uold, const double v0, const double *psi, 
        const double *sigmazx, const double *sigmazy,
       double eta, double deltat, int n){

    for(int i=0;i<n;i++){
       
       double y = (u[i]-uold[i])/deltat / 2 / v0 * exp(psi[i]/a[i]) ; 
       double dFdx = a[i]/sqrt(1+y*y) / deltat / 2 / v0 * exp(psi[i]/a[i]) * sigmazx[i] + eta/deltat;
       double dFdu = - a[i]/sqrt(1+y*y) / deltat / 2 / v0 * exp(psi[i]/a[i]) * sigmazx[i] - eta/deltat;
       double dFda = asinh(y) * sigmazx[i] + 
              a[i]/sqrt(1+y*y) * (u[i]-uold[i])/deltat / 2 / v0 * (-psi[i]/a[i]/a[i])* exp(psi[i]/a[i]) * sigmazx[i];
       double dFdpsi = a[i]/sqrt(1+y*y)*(u[i]-uold[i])/deltat / 2 / v0 / a[i]* exp(psi[i]/a[i]) * sigmazx[i];
       double dFdtau = -1.0;
       double dFdsigma = a[i] * asinh(y);
       
       grad_a[i] = - grad_u[i] / dFdx * dFda;
       grad_uold[i] = - grad_u[i] / dFdx * dFdu;
       grad_psi[i] = - grad_u[i] / dFdx * dFdpsi;
       grad_sigmazx[i] = - grad_u[i] / dFdx * dFdsigma;
       grad_sigmazy[i] = - grad_u[i] / dFdx * dFdtau;
       
    }

}