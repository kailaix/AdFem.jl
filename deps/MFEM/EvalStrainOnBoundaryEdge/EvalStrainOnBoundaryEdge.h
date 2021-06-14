#include "../Common.h"
namespace MFEM{
    void EvalStrainOnBoundaryEdgeForward(double *s, const double *u, const int64 *edgeid, 
        int n_edge){
        int k = 0;
        int e, ll, rr;
        Eigen::VectorXd LineIntegralNode, LineIntegralWeights;
        line_integral_gauss_quadrature(LineIntegralNode, LineIntegralWeights, mmesh.lorder);
        int LineIntegralN = LineIntegralNode.size();
        for (int i = 0; i < n_edge; i++){
            int l = edgeid[2*i]-1, r = edgeid[2*i+1]-1;
            std::tie(e, ll, rr) = mmesh.edge_to_elem[std::make_pair(l, r)];
            double u1 = u[l], u2 = u[r], v1 = u[l+mmesh.ndof], v2 = u[r+mmesh.ndof];
            double a1 = mmesh.elements[e]->Coef(0, ll);
            double a2 = mmesh.elements[e]->Coef(0, rr);
            double b1 = mmesh.elements[e]->Coef(1, ll);
            double b2 = mmesh.elements[e]->Coef(1, rr);
            // std::cout << mmesh.elements[e]->Coef << std::endl;
            s[k] = a1 * u1 + a2 * u2;
            s[k+1] = b1 * v1 + b2 * v2;
            s[k+2] = u1 * b1 + u2 * b2 + v1 * a1 + v2 * a2;
            k+=3;
            for (int j = 0; j < LineIntegralN-1; j++){
                s[k] = s[k-3];
                s[k+1] = s[k-2];
                s[k+2] = s[k-1];
                k+=3;
            }
        }
    }


    void EvalStrainOnBoundaryEdgeBackward(
        double *grad_u, const double *grad_s,
        const double *s, const double *u, const int64 *edgeid, 
        int n_edge){
        int k = 0;
        int e, ll, rr;
        Eigen::VectorXd LineIntegralNode, LineIntegralWeights;
        line_integral_gauss_quadrature(LineIntegralNode, LineIntegralWeights, mmesh.lorder);
        int LineIntegralN = LineIntegralNode.size();
        for (int i = 0; i < n_edge; i++){
            int l = edgeid[2*i]-1, r = edgeid[2*i+1]-1;
            std::tie(e, ll, rr) = mmesh.edge_to_elem[std::make_pair(l, r)];
            double u1 = u[l], u2 = u[r], v1 = u[l+mmesh.ndof], v2 = u[r+mmesh.ndof];
            double a1 = mmesh.elements[e]->Coef(0, ll);
            double a2 = mmesh.elements[e]->Coef(0, rr);
            double b1 = mmesh.elements[e]->Coef(1, ll);
            double b2 = mmesh.elements[e]->Coef(1, rr);
            // std::cout << mmesh.elements[e]->Coef << std::endl;
            for (int j = 0; j < LineIntegralN; j++){
                grad_u[l] += grad_s[k] * a1 + grad_s[k+2] * b1;
                grad_u[r] += grad_s[k] * a2 + grad_s[k+2] * b2;
                grad_u[l+mmesh.ndof] += grad_s[k+1] * b1 + grad_s[k+2] * a1;
                grad_u[r+mmesh.ndof] += grad_s[k+1] * b2 + grad_s[k+2] * a2;
                k += 3;
            }
        }
    }
}