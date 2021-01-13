#include "../Common.h"

namespace MFEM{
    void ComputePmlElasticTermForward(
                double * k1,
                double * k2,
                double * k3,
                double * k4,
                const double * u,
                const double * betap, 
                const double * Ep,
                const double * nup, 
                const double * nv){
        int elem_ndof = mmesh.elem_ndof;
        MatrixXd n(2,1);
        MatrixXd N(2,2), P(2,2);
        MatrixXd M(4,4);
        VectorXd un(4), dun(2), up(4), dup(2);
        VectorXd temp(2);
        for(int i = 0; i<mmesh.nelem; i++){
            NNFEM_Element * elem = mmesh.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                int k = i*elem->ngauss + j;
                double beta_val = betap[k];
                double lambda = Ep[k];
                double mu = nup[k];
                double a = 2*mu + lambda;
                double b = lambda;
                double c = mu;
                M << a, 0.0, 0.0, b, 
                    0.0, c, c, 0.0,
                    0.0, c, c, 0.0,
                    b, 0.0, 0.0, a;
                // M  << a, 0.0, 0.0, 0.0,
                //       0.0, a, 0.0, 0.0,
                //       0.0, 0.0, a, 0.0,
                //       0.0, 0.0, 0.0, a;
                    

                
                n << nv[2*k], nv[2*k+1];
                N = n*n.transpose();
                P = MatrixXd::Identity(2,2) - N;
                for (int p = 0; p < elem_ndof; p++){
                    for (int q = 0; q < elem_ndof; q++){
                        temp << elem->hx(q, j), elem->hy(q, j);
                        temp = N * temp * elem->w[j];
                        un << temp[0] * u[elem->dof[q]], temp[1] * u[elem->dof[q]], 
                              temp[0] * u[elem->dof[q]+mmesh.ndof], temp[1] * u[elem->dof[q]+mmesh.ndof];

                        temp << elem->hx(q, j), elem->hy(q, j);
                        temp = P * temp * elem->w[j];
                        up << temp[0] * u[elem->dof[q]], temp[1] * u[elem->dof[q]], 
                              temp[0] * u[elem->dof[q]+mmesh.ndof], temp[1] * u[elem->dof[q]+mmesh.ndof];

                        un = M * un; 
                        up = M * up;
                        dun << elem->hx(p, j), elem->hy(p, j);
                        dup << elem->hx(p, j), elem->hy(p, j);                        
                        dun = N * dun; 
                        dup = P * dup;

                        
                        k1[elem->dof[p]] += dun[0] * un[0] + dun[1] * un[1];
                        k1[elem->dof[p]+mmesh.ndof] += dun[0] * un[2] + dun[1] * un[3];
                        

                        k2[elem->dof[p]] += beta_val * (un[0] * n(0, 0) + un[1] * n(1, 0)) * elem->h(p, j);
                        k2[elem->dof[p]+mmesh.ndof] += beta_val * (un[2] * n(0, 0) + un[3] * n(1, 0)) * elem->h(p, j);
                        
                        k3[elem->dof[p]] += dun[0] * up[0] + dun[1] * up[1];
                        k3[elem->dof[p]+mmesh.ndof] += dun[0] * up[2] + dun[1] * up[3];

                        k3[elem->dof[p]] += dup[0] * un[0] + dup[1] * un[1];
                        k3[elem->dof[p]+mmesh.ndof] += dup[0] * un[2] + dup[1] * un[3];


                        k4[elem->dof[p]] += dup[0] * up[0] + dup[1] * up[1];
                        k4[elem->dof[p]+mmesh.ndof] +=  dup[0] * up[2] + dup[1] * up[3];
                        
                    }
                }
            }
        }
    }
    
}