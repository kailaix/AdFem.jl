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
void NH_forward(double *psi, 
    int64 *indices, double *vv, 
    const double *ui, const double *mu, const double *lamb){
    ADGraph adGraph;
    int elem_ndof = mmesh.elem_ndof;
    int k1 = 0, k2 = 0;
    for (int i = 0; i < mmesh.nelem; i++){
        auto elem = mmesh.elements[i];
        for (int j = 0; j < elem->ngauss; j++){

            
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

            AReal Ic = c11 + c22;
            AReal J = ((1+ux) * (1+vy) - uy * vx)*((1+ux) * (1+vy) - uy * vx);    
            AReal PSI = mu[i * elem->ngauss + j]/2.0 * (Ic - 2.0) - mu[i * elem->ngauss + j] * log(J)/2.0 + lamb[i * elem->ngauss + j]/2.0 * log(J) * log(J)/4.0;

            SetAdjoint(PSI, 1.0);
            PropagateAdjoint();
            for (int r = 0; r < elem_ndof; r++){
                psi[elem->dof[r]] += GetAdjoint(u[r]) * elem->w[j];
                psi[elem->dof[r] + mmesh.ndof] += GetAdjoint(u[r+elem_ndof]) * elem->w[j];
            }

            for(int r = 0; r < 2*elem_ndof; r++){
                for(int s = 0; s < 2*elem_ndof; s++){
                    double val = GetAdjoint(u[r], u[s]) * elem->w[j];
                    indices[2*k1] = r>=elem_ndof ? elem->dof[r-elem_ndof]+mmesh.ndof : elem->dof[r];
                    indices[2*k1+1] = s>=elem_ndof ? elem->dof[s-elem_ndof]+mmesh.ndof : elem->dof[s];
                    vv[k1] = val;
                    k1++;
                }
            }

            adGraph.Clear();

        }
    }

}


extern "C" void NH_forward_Julia(double *psi, 
    int64 *indices, double *vv, 
    const double *ui, const double *mu, const double *lamb){
    NH_forward(psi, indices, vv, ui, mu, lamb);
}