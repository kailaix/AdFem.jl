#include "../Common.h"
namespace MFEM{
    void ComputeFemTractionV_forward(
        double *out, const double *t, 
        const int64 *edgeid, int n
    ){
        Eigen::VectorXd xs, w;
        line_integral_gauss_quadrature(xs, w, mmesh.lorder);
        int Ng = xs.size();

        for(int i = 0; i < n; i++){
            int l = edgeid[2*i] - 1, r = edgeid[2*i+1] - 1;
            double len = (mmesh.nodes.row(l) - mmesh.nodes.row(r)).norm();
            for (int k = 0; k < Ng; k++){
                out[l] += w[k] * t[Ng * i + k] * (1-xs[k]) * len;
                out[r] += w[k] * t[Ng * i + k] * xs[k] * len;
            }
        }
    }

    void ComputeFemTraction_backward(
        double * gradt, 
        const double *grad_out, 
        const double *out, const double *t, 
        const int64 *edgeid, int n
    ){
        Eigen::VectorXd xs, w;
        line_integral_gauss_quadrature(xs, w, mmesh.lorder);
        int Ng = xs.size();

        for(int i = 0; i < n; i++){
            int l = edgeid[2*i] - 1, r = edgeid[2*i+1] - 1;
            double len = (mmesh.nodes.row(l) - mmesh.nodes.row(r)).norm();
            for (int k = 0; k < Ng; k++){
                gradt[Ng * i + k] = grad_out[l] * w[k] * (1-xs[k]) * len + grad_out[r] * w[k] * xs[k] * len;
            }
        }
    }
}