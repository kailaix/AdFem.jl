#include <math.h>       /* asinh, exp, cosh */
#include <assert.h>

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
  double s = u / 2.0 / v0;
  double F = a * asinh_exp(s, inv_o, log_o) * sigma - tau + eta * u;
  return F; 
}

double Bisection(double xR, double xL, double psi, 
        double a, double uold, double deltat, double eta, double v0, double tau, double sigma){

    assert(psi>0);
    assert(a>0);
    assert(eta>0);
    assert(v0>0);
    assert(tau>0);
    assert(sigma>0);
    assert(uold>0);

    double fL = f(xL, psi, a, uold, deltat, eta, v0, tau, sigma);
    double fR = f(xR, psi, a, uold, deltat, eta, v0, tau, sigma);
    int maxIter = 1000000;
    double xTol = 1e-10;
    double fTol = 1e-10;
    if (fL*fR>0){
        printf("Initial guess (%e, %e) (resulting f = (%e, %e)) not valid\n", xL, xR, fL, fR);
        // exit(1);
        return -1; 
    }

    // double xM = (xL+xR)/2;
    // double fM = f(xM, psi, a, uold, deltat, eta, v0, tau, sigma);
    double fM = fL;
    double xM = xL;
    double xErr = abs(xM - xL);
    double fErr = abs(fM);
    int numIter = 1;

    // assumption: fL < 0, fR > 0

    // assert(fL<0);
    // assert(fR>0);

    // printf("iter = 1, fL=%e, fM=%e, fR=%e\n", fL, fM, fR);
    while (true){
        
        if (numIter>maxIter || (xErr<xTol && fErr <= fTol) )
            break;
        xM = (xL+xR)/2;
        fM = f(xM, psi, a, uold, deltat, eta, v0, tau, sigma);

        // printf("iter = %d, x = (%e, %e, %e), f = (%e, %e, %e)\n", numIter, xL-2, xM-2, xR-2, fL, fM, fR);
        if (fL*fM<0){
            xR = xM;
            fR = fM;
        }      
        else if (fL*fM>0) {
            xL = xM; 
            fL = fM;
        }
        else{
            break;
        }
        
        xErr = abs(xR-xL);
        fErr = abs(fM);
        numIter += 1;
    }
    // if (numIter>maxIter){
    //     printf("WARNING: Bisection Does Find Optimal Solution, use Newton's method for refinement\n");
    //     int iter;
    //     for (iter=0; iter<100; iter++) {
    //         double inv_o = exp(-psi/a);
    //         double log_o = psi/a;
    //         double s = (xM-uold) / deltat / 2.0 / v0;

    //         double F = a * asinh_exp(s, inv_o, log_o) * sigma - tau + eta * (xM-uold) / deltat;

    //         double dFdx = a  / deltat / 2.0 / v0 * 1.0/sqrt(s*s + inv_o*inv_o) * sigma + eta/deltat;
            
    //         // std::cout << "dF " << dFdx-dFdx_ << std::endl;
    //         double dx = F / dFdx;

    //         if (abs(dx) < 1e-10){
    //             break;
    //         }
    //         xM -= dx;
    //     }
    //     printf("Newton's iteration: %d\n", iter);
    // }
    // printf("Number of iterations = %d\n", numIter);
    return xM; 
}

void forward(double *u, const double *a, const double *uold, const double v0, const double *psi, const double *sigma, const double *tau,
       double eta, double deltat, int n){
    for (int i=0; i<n; i++) {
        double xL = 0.0;
        double xR = tau[i]/eta;
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