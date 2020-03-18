#include <math.h>       /* asinh, exp, cosh */


// o/sqrt(a^2+o^2) = 1/sqrt(a^2 o^{-2} + 1)

// log (so+sqrt(1+s^2*o^2)) = log o + log( s + sqrt(s^2 + o^{-2}))
double asinh_exp(double s, double inv_o, double log_o){ 
  // std::cout << log(s + sqrt(s*s + inv_o*inv_o)) << std::endl;
  return log_o + log(s + sqrt(s*s + inv_o*inv_o));
}


void forward(double *u, const double *a, const double *uold, const double v0, const double *psi, const double *sigma, const double *tau,
       double eta, double deltat, int n){
// TODO:
  int max_iter = 1000;
  double tol = 1e-8;
  double dx, dFdx;
  for (int i=0; i<n; i++) {
    u[i] = uold[i] * 2; // avoid u = uold; 
    // u[i] = uold[i] + deltat * tau[i]/eta; // constrained by maximum velocity v ~ tau/eta.
    int iter;
    for (iter=0; iter<max_iter; iter++) {
      double inv_o = exp(-psi[i]/a[i]);
      double log_o = psi[i]/a[i];
      double s = (u[i]-uold[i]) / deltat / 2.0 / v0;

      // double y = (u[i]-uold[i]) / deltat / 2.0 / v0 * exp(psi[i]/a[i]);
      // double F_ = a[i] * asinh(y) * sigma[i] - tau[i] + eta * (u[i]-uold[i]) / deltat;
      double F = a[i] * asinh_exp(s, inv_o, log_o) * sigma[i] - tau[i] + eta * (u[i]-uold[i]) / deltat;
      // std::cout << F-F_ << std::endl;

      // double dFdx_ = a[i]  / deltat / 2.0 / v0 * exp(psi[i]/a[i]) / sqrt(1+y*y) * sigma[i] + eta/deltat;
      dFdx = a[i]  / deltat / 2.0 / v0 * 1.0/sqrt(s*s + inv_o*inv_o) * sigma[i] + eta/deltat;

      // std::cout << "dF " << dFdx-dFdx_ << std::endl;
      dx = F / dFdx;
      // if (i==0)
      //   printf("%f, %f\n", dFdx, dx);
      if (abs(dx)/abs(u[i]) < tol){
        // printf("Variable %d, iter = %d,  residual = %f, dx = %f\n", i, iter, F, dx);
        break;
      }
      u[i] -= dx;
    }
    // printf("Variable %d, sigma[i] = %e, %e, error = %f, iter = %d, dFdx = %f\n", i, sigma[i], tau[i], dx, iter, dFdx);
  }
}

void backward(
        double * grad_a, double *grad_uold, double *grad_psi, double *grad_sigman, double *grad_sigmazx, const double *grad_u, 
        const double *u, const double *a, const double *uold, const double v0, const double *psi, 
        const double *sigman, const double *sigmazx,
       double eta, double deltat, int n){

    for(int i=0;i<n;i++){

      double inv_o = exp(-psi[i]/a[i]);
      double log_o = psi[i]/a[i];
      double s = (u[i]-uold[i]) / deltat / 2.0 / v0;
       
      //  double y = (u[i]-uold[i])/deltat / 2 / v0 * exp(psi[i]/a[i]) ; 
      //  double dFdx = a[i]/sqrt(1+y*y) / deltat / 2 / v0 * exp(psi[i]/a[i]) * sigman[i] + eta/deltat;
      //  double dFdu = - a[i]/sqrt(1+y*y) / deltat / 2 / v0 * exp(psi[i]/a[i]) * sigman[i] - eta/deltat;
      //  double dFda = asinh(y) * sigman[i] + 
      //         a[i]/sqrt(1+y*y) * (u[i]-uold[i])/deltat / 2 / v0 * (-psi[i]/a[i]/a[i])* exp(psi[i]/a[i]) * sigman[i];
      //  double dFdpsi = a[i]/sqrt(1+y*y)*(u[i]-uold[i])/deltat / 2 / v0 / a[i]* exp(psi[i]/a[i]) * sigman[i];
      //  double dFdtau = -1.0;
      //  double dFdsigma = a[i] * asinh(y);

       double dFdx = a[i] / deltat / 2 / v0 * 1.0/sqrt(s*s + inv_o*inv_o) * sigman[i] + eta/deltat;
       double dFdu = - a[i] / deltat / 2 / v0 * 1.0/sqrt(s*s + inv_o*inv_o) * sigman[i] - eta/deltat;
       double dFda = asinh_exp(s, inv_o, log_o) * sigman[i] + 
              a[i] * (u[i]-uold[i])/deltat / 2 / v0 * (-psi[i]/a[i]/a[i])* 1.0/sqrt(s*s + inv_o*inv_o) * sigman[i];
       double dFdpsi = a[i]*(u[i]-uold[i])/deltat / 2 / v0 / a[i]* 1.0/sqrt(s*s + inv_o*inv_o) * sigman[i];
       double dFdtau = -1.0;
       double dFdsigma = a[i] * asinh_exp(s, inv_o, log_o);
       
       grad_a[i] = - grad_u[i] / dFdx * dFda;
       grad_uold[i] = - grad_u[i] / dFdx * dFdu;
       grad_psi[i] = - grad_u[i] / dFdx * dFdpsi;
       grad_sigman[i] = - grad_u[i] / dFdx * dFdsigma;
       grad_sigmazx[i] = - grad_u[i] / dFdx * dFdtau;
       
    }

}