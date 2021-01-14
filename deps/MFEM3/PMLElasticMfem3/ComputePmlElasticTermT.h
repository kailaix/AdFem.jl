#include "../Common.h"

namespace MFEM{
    void ComputePmlElasticTermTForward(
                double * k1,
                double * k2,
                double * k3,
                double * k4,
                const double * u,
                const double * betap, 
                const double * Ep,
                const double * nup, 
                const double * nv){
        int elem_ndof = mmesh3.elem_ndof;
        MatrixXd n(3,1);
        MatrixXd N(3,3), P(3,3);
        MatrixXd M(9,9);
        VectorXd un(9), dun(3), up(9), dup(3);
        VectorXd temp(3);
        for(int i = 0; i<mmesh3.nelem; i++){
            NNFEM_Element3 * elem = mmesh3.elements[i];
            for (int j = 0; j< elem->ngauss; j++){
                int k = i*elem->ngauss + j;
                double beta_val = betap[k];
                double lambda = Ep[k];
                double mu = nup[k];
                double a = 2*mu + lambda;
                double b = lambda;
                double c = mu;
                M << a,0,0,0,b,0,0,0,b,
                    0,c,0,c,0,0,0,0,0,
                    0,0,c,0,0,0,c,0,0,
                    0,c,0,c,0,0,0,0,0,
                    b,0,0,0,a,0,0,0,b,
                    0,0,0,0,0,c,0,c,0,
                    0,0,c,0,0,0,c,0,0,
                    0,0,0,0,0,c,0,c,0,
                    b,0,0,0,b,0,0,0,a;
                    

                n << nv[3*k], nv[3*k+1], nv[3*k+2];
                N = n*n.transpose();
                P = MatrixXd::Identity(3,3) - N;
                for (int p = 0; p < elem_ndof; p++){
                    for (int q = 0; q < elem_ndof; q++){
                        temp << elem->hx(q, j), elem->hy(q, j), elem->hz(q, j);
                        temp = N * temp * elem->w[j];
                        un << temp[0] * u[elem->dof[q]], temp[1] * u[elem->dof[q]], temp[2] * u[elem->dof[q]], 
                              temp[0] * u[elem->dof[q]+mmesh3.ndof], temp[1] * u[elem->dof[q]+mmesh3.ndof], temp[2] * u[elem->dof[q]+mmesh3.ndof],
                              temp[0] * u[elem->dof[q]+2*mmesh3.ndof], temp[1] * u[elem->dof[q]+2*mmesh3.ndof], temp[2] * u[elem->dof[q]+2*mmesh3.ndof];

                        temp << elem->hx(q, j), elem->hy(q, j), elem->hz(q, j);
                        temp = P * temp * elem->w[j];
                        up << temp[0] * u[elem->dof[q]], temp[1] * u[elem->dof[q]], temp[2] * u[elem->dof[q]], 
                              temp[0] * u[elem->dof[q]+mmesh3.ndof], temp[1] * u[elem->dof[q]+mmesh3.ndof], temp[2] * u[elem->dof[q]+mmesh3.ndof],
                              temp[0] * u[elem->dof[q]+2*mmesh3.ndof], temp[1] * u[elem->dof[q]+2*mmesh3.ndof], temp[2] * u[elem->dof[q]+2*mmesh3.ndof];

                        un = M * un; 
                        up = M * up;
                        dun << elem->hx(p, j), elem->hy(p, j), elem->hz(p, j);
                        dup << elem->hx(p, j), elem->hy(p, j), elem->hz(p, j);                        
                        dun = N * dun; 
                        dup = P * dup;

                        
                        k1[elem->dof[p]] += dun[0] * un[0] + dun[1] * un[1] + dun[2] * un[2];
                        k1[elem->dof[p]+mmesh3.ndof] += dun[0] * un[3] + dun[1] * un[4] + dun[2] * un[5];
                        k1[elem->dof[p]+2*mmesh3.ndof] += dun[0] * un[6] + dun[1] * un[7] + dun[2] * un[8];
                        

                        k2[elem->dof[p]] += beta_val * (un[0] * n(0, 0) + un[1] * n(1, 0) + un[2] * n(2, 0)) * elem->h(p, j);
                        k2[elem->dof[p]+mmesh3.ndof] += beta_val * (un[3] * n(0, 0) + un[4] * n(1, 0) + un[5] * n(2, 0)) * elem->h(p, j);
                        k2[elem->dof[p]+2*mmesh3.ndof] += beta_val * (un[6] * n(0, 0) + un[7] * n(1, 0) + un[8] * n(2, 0)) * elem->h(p, j);

                        k3[elem->dof[p]] += dun[0] * up[0] + dun[1] * up[1] + dun[2] * up[2];
                        k3[elem->dof[p]+mmesh3.ndof] += dun[0] * up[3] + dun[1] * up[4] + dun[2] * up[5];
                        k3[elem->dof[p]+2*mmesh3.ndof] += dun[0] * up[6] + dun[1] * up[7] + dun[2] * up[8];

                        k3[elem->dof[p]] += dup[0] * un[0] + dup[1] * un[1] + dup[2] * un[2];
                        k3[elem->dof[p]+mmesh3.ndof] += dup[0] * un[3] + dup[1] * un[4] + dup[2] * un[5];
                        k3[elem->dof[p]+2*mmesh3.ndof] += dup[0] * un[6] + dup[1] * un[7] + dup[2] * un[8];


                        k4[elem->dof[p]] += dup[0] * up[0] + dup[1] * up[1] + dup[2] * up[2];
                        k4[elem->dof[p]+mmesh3.ndof] +=  dup[0] * up[3] + dup[1] * up[4] + dup[2] * up[5];
                        k4[elem->dof[p]+2*mmesh3.ndof] +=  dup[0] * up[6] + dup[1] * up[7] + dup[2] * up[8];

                    }
                }
            }
        }
    }
    
}