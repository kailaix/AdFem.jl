#include <math.h>       /* asinh, exp, cosh */
#include <assert.h>
#include <limits>
double asinh_exp(double s, double inv_o, double log_o){ 
  return log_o + log(s+ sqrt(s*s + inv_o*inv_o));
}

double f(double u, double psi, double a, double eta, double v0, double tau, double sigma){
  double inv_o = exp(-psi/a);
  double log_o = psi/a;
  double s = u / 2.0 / v0;
  double F = a * asinh_exp(s, inv_o, log_o) * sigma - tau + eta * u;
  return F; 
}

double Bisection(double xR, double xL, double psi, 
        double a, double eta, double v0, double tau, double sigma){

    assert(psi>0);
    assert(a>0);
    assert(eta>0);
    assert(v0>0);
    assert(tau>0);
    assert(sigma>0);

    double fL = f(xL, psi, a, eta, v0, tau, sigma);
    double fR = f(xR, psi, a, eta, v0, tau, sigma);
    int maxIter = 1000000;
    double xTol = 1e-10;
    double fTol = 1e-10;

    if (abs(fL)<fTol) return xL;
    if (abs(fR)<fTol) return xR;
    if (fL*fR>0){
        printf("Initial guess (%e, %e) (resulting f = (%e, %e)) not valid\n", xL, xR, fL, fR);
        // exit(1);
        return std::numeric_limits<double>::quiet_NaN(); 
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
        fM = f(xM, psi, a, eta, v0, tau, sigma);

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
        // xM = (xL+xR)/2;
        // int iter;
        // for (iter=0; iter<100; iter++) {
        //     double inv_o = exp(-psi/a);
        //     double log_o = psi/a;
        //     double s = xM / 2.0 / v0;

        //     double F = a * asinh_exp(s, inv_o, log_o) * sigma - tau + eta * xM;

        //     double dFdx = a  / 2.0 / v0 * 1.0/sqrt(s*s + inv_o*inv_o) * sigma + eta;
            
        //     // std::cout << "dF " << dFdx-dFdx_ << std::endl;
        //     double dx = F / dFdx;

        //     if (abs(dx) < 1e-10){
        //         break;
        //     }
        //     xM -= dx;
        // }
        // printf("Newton's iteration: %d\n", iter);
    // }
    // printf("Number of iterations = %d\n", numIter);
    return xM; 
}

void forward(double *u, const double *a, const double v0, const double *psi, const double *sigma, const double *tau,
       double eta, int n){
    for (int i=0; i<n; i++) {
        double xL = 0.0;
        double xR = tau[i]/eta;
        u[i] = Bisection(xR, xL, psi[i], a[i], eta, v0, tau[i], sigma[i]);
  }
}

void backward(
        double * grad_a, double *grad_psi, double *grad_sigma, double *grad_tau, double *grad_eta, 
        double * grad_v0, 
        const double *grad_u, 
        const double *u, const double *a, const double v0, const double *psi, 
        const double *sigma, const double *tau,
       double eta, int n){
    grad_v0[0] = 0.0;
    grad_eta[0] = 0.0;
    for(int i=0;i<n;i++){

      double inv_o = exp(-psi[i]/a[i]);
      double log_o = psi[i]/a[i];
      double s = u[i] / 2.0 / v0;
       
       double dFdx = a[i] / 2 / v0 * 1.0/sqrt(s*s + inv_o*inv_o) * sigma[i] + eta;
       double dFda = asinh_exp(s, inv_o, log_o) * sigma[i] + 
              a[i] * u[i] / 2 / v0 * (-psi[i]/a[i]/a[i])* 1.0/sqrt(s*s + inv_o*inv_o) * sigma[i];
       double dFdpsi = a[i]*u[i] / 2 / v0 / a[i]* 1.0/sqrt(s*s + inv_o*inv_o) * sigma[i];
       double dFdtau = -1.0;
       double dFdsigma = a[i] * asinh_exp(s, inv_o, log_o);
       double dFdeta = u[i];
       double dFdv0 = -a[i]*sigma[i]/v0/sqrt(1+4*v0*v0*inv_o*inv_o);
       
       grad_a[i] = - grad_u[i] / dFdx * dFda;
       grad_psi[i] = - grad_u[i] / dFdx * dFdpsi;
       grad_sigma[i] = - grad_u[i] / dFdx * dFdsigma;
       grad_tau[i] = - grad_u[i] / dFdx * dFdtau;
       grad_v0[0] -= grad_u[i]/dFdx * dFdv0;
       grad_eta[0] -= grad_u[i]/dFdx * dFdeta;
    }

}