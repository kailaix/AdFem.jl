#include "../Common.h"
#include "had.h"
using namespace had;
using namespace std;

/*

NH_forward: 

F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

Ic = tr(C)
J  = ln(det(C))

ic, jc : 2 * ndof 
idi, iv, jdi, jv : ngauss * (2*elem_ndof)^2
*/
namespace had { threadDefine ADGraph* g_ADGraph = 0; }
void NH_forward(double *ic, double *jc, 
    int64 *idi, double *iv, int64 *jdi, double *jv, 
    const double *ui){
    ADGraph adGraph;
    int elem_ndof = mmesh.elem_ndof;
    int k1 = 0, k2 = 0;
    for (int i = 0; i < mmesh.nelem; i++){
        auto elem = mmesh.elements[i];
        for (int j = 0; j < elem->ngauss; j++){

            {
                AReal ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0;
                std::vector<AReal> u(2*elem_ndof);
                for (int s = 0; s<elem_ndof; s++){
                    u[s] = ui[elem->dof[s]];
                    u[s+elem_ndof] = ui[mmesh.ndof + elem->dof[s]];
                }

                for (int r = 0; r < elem_ndof; r++){
                    ux += u[r] * elem->hx(r, j);
                    uy += u[r] * elem->hy(r, j);
                    vx += u[r + elem_ndof] * elem->hx(r, j);
                    vy += u[r + elem_ndof] * elem->hy(r, j);
                }
                AReal c11 = (ux + 1.0) * (ux + 1.0) + vx * vx;
                AReal c12 = (ux + 1.0) * uy + vx * (vy + 1.0);
                AReal c21 = uy * (ux + 1.0) + (vy + 1.0) * vx;
                AReal c22 = (vy + 1.0) * (vy + 1.0) + uy * uy;

                AReal icc = c11 + c22;
                SetAdjoint(icc, 1.0);
                PropagateAdjoint();
                for (int r = 0; r < elem_ndof; r++){
                    ic[elem->dof[r]] += GetAdjoint(u[r]) * elem->w[j];
                    ic[elem->dof[r] + mmesh.ndof] += GetAdjoint(u[r+elem_ndof]) * elem->w[j];
                }

                for(int r = 0; r < 2*elem_ndof; r++){
                    for(int s = 0; s < 2*elem_ndof; s++){
                        double val = GetAdjoint(u[r], u[s]) * elem->w[j];
                        idi[2*k1] = r>=elem_ndof ? elem->dof[r-elem_ndof]+mmesh.ndof : elem->dof[r];
                        idi[2*k1+1] = s>=elem_ndof ? elem->dof[s-elem_ndof]+mmesh.ndof : elem->dof[s];
                        iv[k1] = val;
                        k1++;
                    }
                }

                adGraph.Clear();
            }

            {
                AReal ux = 0.0, uy = 0.0, vx = 0.0, vy = 0.0;
                std::vector<AReal> u(2*elem_ndof);
                for (int s = 0; s<elem_ndof; s++){
                    u[s] = ui[elem->dof[s]];
                    u[s+elem_ndof] = ui[mmesh.ndof + elem->dof[s]];
                }

                for (int r = 0; r < elem_ndof; r++){
                    ux += u[r] * elem->hx(r, j);
                    uy += u[r] * elem->hy(r, j);
                    vx += u[r + elem_ndof] * elem->hx(r, j);
                    vy += u[r + elem_ndof] * elem->hy(r, j);
                }
                AReal c11 = (ux + 1.0) * (ux + 1.0) + vx * vx;
                AReal c12 = (ux + 1.0) * uy + vx * (vy + 1.0);
                AReal c21 = uy * (ux + 1.0) + (vy + 1.0) * vx;
                AReal c22 = (vy + 1.0) * (vy + 1.0) + uy * uy;
                AReal jln = log(c11 * c22 - c12 * c21);
                SetAdjoint(jln, 1.0);
                PropagateAdjoint();
                for (int r = 0; r < elem_ndof; r++){
                    jc[elem->dof[r]] += GetAdjoint(u[r]) * elem->w[j];
                    jc[elem->dof[r] + mmesh.ndof] += GetAdjoint(u[r+elem_ndof]) * elem->w[j];
                }

                for(int r = 0; r < 2*elem_ndof; r++){
                    for(int s = 0; s < 2*elem_ndof; s++){
                        double val = GetAdjoint(u[r], u[s]) * elem->w[j];
                        jdi[2*k2] = r>=elem_ndof ? elem->dof[r-elem_ndof]+mmesh.ndof : elem->dof[r];
                        jdi[2*k2+1] = s>=elem_ndof ? elem->dof[s-elem_ndof]+mmesh.ndof : elem->dof[s];
                        jv[k2] = val;
                        k2 ++;
                    }
                }

                adGraph.Clear();
            }

        }
    }

}


void NH_backward(
    double *grad_ui, 
    const double *grad_ic, const double *grad_jc, 
    const double *grad_iv, const double *grad_jv, 
    const double *ic, const double *jc, const double *ui){
    
}

extern "C" void NH_forward_Julia(double *ic, double *jc, 
    int64 *idi, double *iv, int64 *jdi, double *jv, 
    const double *ui){
    NH_forward(ic, jc, idi, iv, jdi, jv, ui);
}