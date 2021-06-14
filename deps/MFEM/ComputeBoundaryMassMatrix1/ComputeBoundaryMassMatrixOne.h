#include "../Common.h"

namespace MFEM{
    // idx: 1 based
    void ComputeBoundaryMassMatrixOneForward(
        int64 *ij, double *vv, 
        const double *c, const int64 *idx, int nedge
    ){
        int s = 0;
        int cs = 0;
        const double *nodes = mmesh.nodes.data();
        Eigen::VectorXd LineIntegralNode, LineIntegralWeights;
        line_integral_gauss_quadrature(LineIntegralNode, LineIntegralWeights, mmesh.lorder);
        int LineIntegralN = LineIntegralNode.size();
        for(int e = 0; e < nedge; e++){
            int p1 = idx[2*e] - 1;
            int p2 = idx[2*e+1] - 1;
            double len = sqrt((nodes[p1] - nodes[p2])*(nodes[p1] - nodes[p2]) + \
                         (nodes[p1+mmesh.nnode] - nodes[p2+mmesh.nnode])*(nodes[p1+mmesh.nnode] - nodes[p2+mmesh.nnode]));
            for (int k = 0; k < LineIntegralN; k++){
                ij[2*s] = p1;
                ij[2*s+1] = p1;
                vv[s++] = LineIntegralWeights[k] * c[cs] * LineIntegralNode[k] * LineIntegralNode[k] * len;
                ij[2*s] = p1;
                ij[2*s+1] = p2;
                vv[s++] = LineIntegralWeights[k] * c[cs] * LineIntegralNode[k] * (1-LineIntegralNode[k]) * len;
                ij[2*s] = p2;
                ij[2*s+1] = p1;
                vv[s++] = LineIntegralWeights[k] * c[cs] * LineIntegralNode[k] * (1-LineIntegralNode[k]) * len;
                ij[2*s] = p2;
                ij[2*s+1] = p2;
                vv[s++] = LineIntegralWeights[k] * c[cs] * (1-LineIntegralNode[k]) * (1-LineIntegralNode[k]) * len;
                cs++;
            }
            
        }
    }

    void ComputeBoundaryMassMatrixOneBackward(
        double *grad_c, 
        const double *grad_vv,
        const double *vv, 
        const double *c, const int64 *idx, int nedge
    ){
        int s = 0;
        int cs = 0;
        const double *nodes = mmesh.nodes.data();
        Eigen::VectorXd LineIntegralNode, LineIntegralWeights;
        line_integral_gauss_quadrature(LineIntegralNode, LineIntegralWeights, mmesh.lorder);
        int LineIntegralN = LineIntegralNode.size();

        for(int e = 0; e < nedge; e++){
            int p1 = idx[2*e] - 1;
            int p2 = idx[2*e+1] - 1;
            double len = sqrt((nodes[p1] - nodes[p2])*(nodes[p1] - nodes[p2]) + \
                         (nodes[p1+mmesh.nnode] - nodes[p2+mmesh.nnode])*(nodes[p1+mmesh.nnode] - nodes[p2+mmesh.nnode]));
            for (int k = 0; k < LineIntegralN; k++){
                grad_c[cs] += LineIntegralWeights[k] * grad_vv[s++] * LineIntegralNode[k] * LineIntegralNode[k] * len;
                grad_c[cs] += LineIntegralWeights[k] * grad_vv[s++] * LineIntegralNode[k] * (1-LineIntegralNode[k]) * len;
                grad_c[cs] += LineIntegralWeights[k] * grad_vv[s++] * LineIntegralNode[k] * (1-LineIntegralNode[k]) * len;
                grad_c[cs] += LineIntegralWeights[k] * grad_vv[s++] * (1-LineIntegralNode[k]) * (1-LineIntegralNode[k]) * len;
                cs++;
            }
        }
    }
}