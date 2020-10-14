#include "had.h"

using namespace had;
namespace had { threadDefine ADGraph* g_ADGraph = 0; }

void ViscoelasticityStressUpdateForward(
        double *sigma2,
        const double *epsilon1, const double *epsilon2, const double *sigma1, 
        const double *mu_arr, const double *eta_arr, const double *lambda_arr, double dt, int start, int end){
    for(int i = start; i < end; i++){
        double mu = mu_arr[i], eta = eta_arr[i], lambda = lambda_arr[i];
        double t11 = 1 + 2.0/3.0*mu*eta*dt;
        double t12 = -1.0/3.0*mu*eta*dt;
        double J = t11*t11 - t12*t12;
        double a = t11/J, b = -t12/J;
        double c = 1.0/(1.0 + mu * eta * dt);
        double d = a * (2*mu+lambda) + b * lambda, e = b * (2*mu+lambda) + a * lambda, f = c * mu;
        sigma2[3*i] = d * (epsilon2[3*i]-epsilon1[3*i]) + e * (epsilon2[3*i+1]-epsilon1[3*i+1]) + a * sigma1[3*i] + b * sigma1[3*i+1];
        sigma2[3*i+1] = e * (epsilon2[3*i]-epsilon1[3*i]) + d * (epsilon2[3*i+1]-epsilon1[3*i+1]) + b * sigma1[3*i] + a * sigma1[3*i+1];
        sigma2[3*i+2] = f * (epsilon2[3*i+2]-epsilon1[3*i+2]) + c*sigma1[3*i+2];
    }
}

void ViscoelasticityStressUpdateBackward(
        double *grad_epsilon1, double *grad_epsilon2, double *grad_sigma1, 
        double *grad_mu, double *grad_eta, double *grad_lambda,
        const double *grad_sigma2,
        const double *epsilon1, const double *epsilon2, const double *sigma1, 
        const double *mu_arr, const double *eta_arr, const double *lambda_arr, double dt, int start, int end){
    ADGraph adGraph;
    for(int i = start; i < end; i++){
        AReal mu(mu_arr[i]), eta(eta_arr[i]), lambda(lambda_arr[i]);
        AReal eps11(epsilon1[3*i]), eps12(epsilon1[3*i+1]), eps13(epsilon1[3*i+2]),
                eps21(epsilon2[3*i]), eps22(epsilon2[3*i+1]), eps23(epsilon2[3*i+2]);
        AReal sig1(sigma1[3*i]), sig2(sigma1[3*i+1]), sig3(sigma1[3*i+2]);
        AReal t11 = 1 + 2.0/3.0*mu*eta*dt;
        AReal t12 = -1.0/3.0*mu*eta*dt;
        AReal J = t11*t11 - t12*t12;
        AReal a = t11/J, b = -t12/J;
        AReal c = 1.0/(1.0 + mu * eta * dt);
        AReal d = a * (2*mu+lambda) + b * lambda, e = b * (2*mu+lambda) + a * lambda, f = c * mu;
        AReal s1 = d * (eps21-eps11) + e * (eps22-eps12) + a * sig1 + b * sig2;
        AReal s2 = e * (eps21-eps11) + d * (eps22-eps12) + b * sig1 + a * sig2;
        AReal s3 = f * (eps23-eps13) + c*sig3;
        AReal L = s1 * grad_sigma2[3*i] + s2 * grad_sigma2[3*i+1] + s3 * grad_sigma2[3*i+2];
        SetAdjoint(L, 1.0);
        PropagateAdjoint();
        grad_mu[i] = GetAdjoint(mu);
        grad_eta[i] = GetAdjoint(eta);
        grad_lambda[i] = GetAdjoint(lambda);
        grad_sigma1[3*i] = GetAdjoint(sig1);
        grad_sigma1[3*i+1] = GetAdjoint(sig2);
        grad_sigma1[3*i+2] = GetAdjoint(sig3);
        grad_epsilon1[3*i] = GetAdjoint(eps11);
        grad_epsilon1[3*i+1] = GetAdjoint(eps12);
        grad_epsilon1[3*i+2] = GetAdjoint(eps13);
        grad_epsilon2[3*i] = GetAdjoint(eps21);
        grad_epsilon2[3*i+1] = GetAdjoint(eps22);
        grad_epsilon2[3*i+2] = GetAdjoint(eps23);
        adGraph.Clear();
    }
}



extern "C" void ViscoelasticityStressUpdateForwardJulia(double *sigma2,
        const double *epsilon1, const double *epsilon2, const double *sigma1, 
        const double *mu_arr, const double *eta_arr, const double *lambda_arr, double dt, int ng){
    ViscoelasticityStressUpdateForward(sigma2, epsilon1, epsilon2, sigma1, mu_arr, eta_arr, lambda_arr, dt, 0, ng);
}