#include "had.h"

using namespace had; 

void PlaneStrainMatrix_forward(double *out, const double *E, const double *nu, int N){
    for (int i = 0; i < N; i++){
        double s = E[i] * (1- nu[i]) / (1 + nu[i]) / (1 - 2*nu[i]);
        out[9*i] = s;
        out[9*i+1] = s * nu[i]/(1-nu[i]);
        out[9*i+2] = s * nu[i]/(1-nu[i]);
        out[9*i+3] = s * nu[i]/(1-nu[i]);
        out[9*i+4] = s;
        out[9*i+5] = s * nu[i]/(1-nu[i]);
        out[9*i+6] = s * nu[i]/(1-nu[i]);
        out[9*i+7] = s * nu[i]/(1-nu[i]);
        out[9*i+8] = s;
    }
}

void PlaneStrainMatrix_backward(
    double *grad_nu, double *grad_E, 
    const double *grad_out, 
    const double *E, const double *nu, int N
){
    ADGraph adGraph;
    for (int i = 0; i < N; i++){
        AReal Ei = E[i], nui = nu[i];
        AReal s = Ei * (1- nui) / (1 + nui) / (1 - 2*nui);
        AReal L = grad_out[9*i] * s + 
                    grad_out[9*i+1] * s * nui/(1-nui) + 
                    grad_out[9*i+2] * s * nui/(1-nui) + 
                    grad_out[9*i+3] * s * nui/(1-nui) + 
                    grad_out[9*i+4] * s + 
                    grad_out[9*i+5] * s * nui/(1-nui) + 
                    grad_out[9*i+6] * s * nui/(1-nui) + 
                    grad_out[9*i+7] * s * nui/(1-nui) + 
                    grad_out[9*i+8] * s;
        SetAdjoint(L, 1.0);
        PropagateAdjoint();
        grad_E[i] = GetAdjoint(Ei);
        grad_nu[i] = GetAdjoint(nui);
        adGraph.Clear();
    }
}

void PlaneStressMatrix_forward(double *out, const double *E, const double *nu, int N){
    for (int i = 0; i < N; i++){
        double s = E[i] / (1 + nu[i]) / (1 - 2*nu[i]);
        out[9*i] = s * (1-nu[i]);
        out[9*i+1] = s * nu[i];
        out[9*i+2] = 0.0;
        out[9*i+3] = s * nu[i];
        out[9*i+4] = s * (1-nu[i]);
        out[9*i+5] = 0.0;
        out[9*i+6] = 0.0;
        out[9*i+7] = 0.0;
        out[9*i+8] = s * (1-2*nu[i])/2.0;
    }
}

void PlaneStressMatrix_backward(double *grad_nu, double *grad_E, 
    const double *grad_out, 
    const double *E, const double *nu, int N){
    ADGraph adGraph;
    for (int i = 0; i < N; i++){
        AReal Ei = E[i], nui = nu[i];
        AReal s = Ei / (1 + nui) / (1 - 2*nui);
        AReal L = grad_out[9*i] * s * (1-nui) +
                    grad_out[9*i+1] * s * nui +
                    grad_out[9*i+3] * s * nui +
                    grad_out[9*i+4] * s * (1-nui) +
                    grad_out[9*i+8] * s * (1-2*nui)/2.0;
        SetAdjoint(L, 1.0);
        PropagateAdjoint();
        grad_E[i] = GetAdjoint(Ei);
        grad_nu[i] = GetAdjoint(nui);
        adGraph.Clear();
    }
}