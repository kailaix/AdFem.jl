#include <math.h>       /* asinh, exp, cosh */


// o/sqrt(a^2+o^2) = 1/sqrt(a^2 o^{-2} + 1)

// log (so+sqrt(1+s^2*o^2)) = log o + log( s + sqrt(s^2 + o^{-2}))
double asinh_exp(double s, double inv_o, double log_o){ 
  // std::cout << log(s + sqrt(s*s + inv_o*inv_o)) << std::endl;
  return log_o + log(s+ sqrt(s*s + inv_o*inv_o));
}

double f(double u, double psi, double a, double uold, 
    double deltat, double eta, double v0, double tau, double sigma){
  double inv_o = exp(-psi/a);
  double log_o = psi/a;
  double s = (u-uold) / deltat / 2.0 / v0;
//   printf("%e %e\n", s, asinh_exp(s, inv_o, log_o));
  double F = a * asinh_exp(s, inv_o, log_o) * sigma - tau + eta * (u-uold) / deltat;
  return F; 
}

double Bisection(double xR, double xL, double psi, 
        double a, double uold, double deltat, double eta, double v0, double tau, double sigma){
    double fL = f(xL, psi, a, uold, deltat, eta, v0, tau, sigma);
    double fR = f(xR, psi, a, uold, deltat, eta, v0, tau, sigma);
    int maxIter = 200;
    double xTol = 1e-10;
    double fTol = 1e-10;
    if (fL*fR>0){
        printf("Initial guess not valid\n");
        exit(1);
        return -1; 
    }

    double xM = (xL+xR)/2;
    double fM = f(xM, psi, a, uold, deltat, eta, v0, tau, sigma);
    double xErr = abs(xM - xL);
    double fErr = abs(fM);
    int numIter = 1;

    // printf("iter = 1, fL=%e, fM=%e, fR=%e\n", fL, fM, fR);
    while (true){
        // printf("iter = %d, x = (%e, %e), f = (%e, %e)\n", numIter, xL, xR, fL, fR);
        if (numIter>maxIter || (xErr<xTol && fErr <= fTol) )
            break;
        if (fL*fM<=0){
            xR = xM;
            fR = fM;
        }      
        else {
            xL = xM; 
            fL = fM;
        }
        
        xM = (xL+xR)/2;
        fM = f(xM, psi, a, uold, deltat, eta, v0, tau, sigma);
        xErr = abs(xM-xL);
        fErr = abs(fM);
        numIter += 1;
    }
    if (numIter>maxIter){
        printf("WARNING: Bisection Does Find Optimal Solution!!!\n");
    }
    // printf("Number of iterations = %d\n", numIter);
    return xM; 
}

void forward(double *u, const double *a, const double *uold, const double v0, const double *psi, const double *sigma, const double *tau,
       double eta, double deltat, int n){
    for (int i=0; i<n; i++) {
        double xL = uold[i];
        double xR = uold[i] + deltat * tau[i]/eta;
        u[i] = Bisection(xR, xL, psi[i], a[i], uold[i], deltat, eta, v0, tau[i], sigma[i]);
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