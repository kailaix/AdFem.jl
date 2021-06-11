#include "../Common.h"
namespace MFEM{
    void EvalScalarOnBoundaryEdgeForward(
        double *s, const double *u, const int64 *edgeid, 
        int n_edge
    ){
        int k = 0;
        for (int i = 0; i < n_edge; i++){
            int l = edgeid[2*i]-1, r = edgeid[2*i+1]-1;
            for (int j = 0; j < LineIntegralN; j++){
                s[k++] = LineIntegralNode[j] * u[r] + (1-LineIntegralNode[j]) * u[l];
            }
        }
    }

    void EvalScalarOnBoundaryEdgeBackward(
        double * grad_u, const double * grad_s,
        const double *s, const double *u, const int64 *edgeid, 
        int n_edge
    ){
        int k = 0;
        for (int i = 0; i < n_edge; i++){
            int l = edgeid[2*i]-1, r = edgeid[2*i+1]-1;
            for (int j = 0; j < LineIntegralN; j++){
                grad_u[r] +=  LineIntegralNode[j] * grad_s[k];
                grad_u[l] += (1-LineIntegralNode[j]) * grad_s[k++];
            }
        }
    }
}