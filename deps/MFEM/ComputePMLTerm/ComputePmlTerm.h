#include "../Common.h"
namespace MFEM{
    void ComputePmlTermForward(
                double * k1,
                double * k2,
                double * k3,
                double * k4,
                const double * u,
                const double * betap,
                const double * c,
                const double * nv){
        int elem_ndof = mmesh.elem_ndof;
        MatrixXd n(2,1);
        MatrixXd N(2,2), P(2,2);
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                int k = i*elem->ngauss + j;
                double beta_val = betap[k];
                double c_val = c[k];
                n << nv[2*k], nv[2*k+1];
                N = n*n.transpose();
                P = MatrixXd::Identity(2,2) - N;
                for (int p = 0; p < elem_ndof; p++){
                    for (int q = 0; q < elem_ndof; q++){
                        k1[elem->dof[p]] += \
                           ( (N(0,0)*elem->hx(q, j) + N(0,1)*elem->hy(q, j))*(N(0,0)*elem->hx(p, j) + N(0,1)*elem->hy(p, j)) + \
                                (N(1,0)*elem->hx(q, j) + N(1,1)*elem->hy(q, j))*(N(1,0)*elem->hx(p, j) + N(1,1)*elem->hy(p, j)) ) \
                            * u[elem->dof[q]] * c_val* elem->w[j];
                        
                        k2[elem->dof[p]] += \
                           ((N(0,0)*elem->hx(q, j) + N(0,1)*elem->hy(q, j))*n(0,0) + (N(1,0)*elem->hx(q, j) + N(1, 1)*elem->hy(q, j))*n(1,0) ) * elem->h(p, j) \
                            * beta_val * c_val * u[elem->dof[q]] * elem->w[j];
                        
                        k3[elem->dof[p]] += \
                           ( (N(0,0)*elem->hx(q, j) + N(0,1)*elem->hy(q, j))*(P(0,0)*elem->hx(p, j) + P(0,1)*elem->hy(p, j)) + \
                                (N(1,0)*elem->hx(q, j) + N(1,1)*elem->hy(q, j))*(P(1,0)*elem->hx(p, j) + P(1,1)*elem->hy(p, j)) ) \
                            * u[elem->dof[q]] * c_val* elem->w[j];
                        
                        k3[elem->dof[p]] += \
                           ( (P(0,0)*elem->hx(q, j) + P(0,1)*elem->hy(q, j))*(N(0,0)*elem->hx(p, j) + N(0,1)*elem->hy(p, j)) + \
                                (P(1,0)*elem->hx(q, j) + P(1,1)*elem->hy(q, j))*(N(1,0)*elem->hx(p, j) + N(1,1)*elem->hy(p, j)) ) \
                            * u[elem->dof[q]] * c_val* elem->w[j];
                    
                        k4[elem->dof[p]] += \
                           ( (P(0,0)*elem->hx(q, j) + P(0,1)*elem->hy(q, j))*(P(0,0)*elem->hx(p, j) + P(0,1)*elem->hy(p, j)) + \
                                (P(1,0)*elem->hx(q, j) + P(1,1)*elem->hy(q, j))*(P(1,0)*elem->hx(p, j) + P(1,1)*elem->hy(p, j)) ) \
                            * u[elem->dof[q]] * c_val* elem->w[j];
                        
                    }
                }
            }
        }
    }


}