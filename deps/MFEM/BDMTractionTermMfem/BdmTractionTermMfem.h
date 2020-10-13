#include "../Common.h"
#include "eigen3/Eigen/Dense"

namespace MFEM{
  void calcShapeFunction(Eigen::MatrixXd &shape, Eigen::MatrixXd & coords, 
            const Eigen::VectorXd &xs,
            double x1, double y1, double x2, double y2, double x3, double y3);

  void ComputeBDMTractionTermMfem_forward(double *out, 
        const double *sign,
        const double *t, 
        const int *dofs,
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order){
        IntegrationRules rule_;
        IntegrationRule rule = rule_.Get(Element::Type::SEGMENT, order);
        int ngauss = rule.GetNPoints();

        Eigen::MatrixXd shape(2, ngauss), coords(ngauss, 2); 
        Eigen::VectorXd w(ngauss), xs(ngauss);
        for (int i = 0; i < ngauss; i++){
            const IntegrationPoint &ip = rule.IntPoint(i);
            w[i] = ip.weight;
            xs[i] = ip.x;
        }
        
        for (int k = 0; k < bdN; k++){
            double x1 = bdnode_x[3*k], x2 = bdnode_x[3*k+1], x3 = bdnode_x[3*k+2];
            double y1 = bdnode_y[3*k], y2 = bdnode_y[3*k+1], y3 = bdnode_y[3*k+2];
            calcShapeFunction(shape, coords, xs, x1, y1, x2, y2, x3, y3);
            double dist = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
            for(int r = 0; r < ngauss; r++){
                double tval = t[k * ngauss + r];
                for(int p = 0; p < 2; p++)
                    out[dofs[k*2+p]] += (dist * w[r]) * shape(p, r) * tval * sign[k];
            }
        }
    }

    // computes the coefficient for sigma * n = g_N 
    void ComputeBDMTractionBoundaryMfem_forward(double *out, // 2bdN
        const double *sign,
        const double *gN, 
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order){
        IntegrationRules rule_;
        IntegrationRule rule = rule_.Get(Element::Type::SEGMENT, order);
        int ngauss = rule.GetNPoints();

        Eigen::MatrixXd shape(2, ngauss), coords(ngauss, 2); 
        Eigen::VectorXd w(ngauss), xs(ngauss);
        for (int i = 0; i < ngauss; i++){
            const IntegrationPoint &ip = rule.IntPoint(i);
            w[i] = ip.weight;
            xs[i] = ip.x;
        }

        Eigen::MatrixXd M(2, 2);
        M.setZero();
        for (int i = 0; i < ngauss; i++){
            M(0, 0) += xs[i] * xs[i] * w[i];
            M(0, 1) += xs[i] * (1-xs[i]) * w[i];
            M(1, 0) += (1-xs[i]) * xs[i] * w[i];
            M(1, 1) += (1-xs[i]) * (1-xs[i]) * w[i];
        }
        Eigen::MatrixXd Minv = M.inverse();
        
        for (int k = 0; k < bdN; k++){
            double x1 = bdnode_x[3*k], x2 = bdnode_x[3*k+1], x3 = bdnode_x[3*k+2];
            double y1 = bdnode_y[3*k], y2 = bdnode_y[3*k+1], y3 = bdnode_y[3*k+2];
            
            // form [2x2] [2] = [2]
            Eigen::Vector2d rhs;
            rhs.setZero();
            for (int r = 0; r < ngauss; r++){
                rhs[0] += gN[k*ngauss+r] * xs[r] * w[r];
                rhs[1] += gN[k*ngauss+r] * (1-xs[r]) * w[r];
            }
            double dist = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
            VectorXd v = Minv * rhs * dist;
            out[2*k] = v[0] * sign[k];
            out[2*k+1] = v[1] * sign[k];
        }
    }
}

extern "C" void ComputeBDMTractionTermMfem_forward_Julia(
        double *out, const double *sign, const double *t, 
        const int *dofs,
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order
){
    MFEM::ComputeBDMTractionTermMfem_forward(out, sign, t, dofs, bdnode_x, bdnode_y, bdN, order);
}

extern "C" void ComputeBDMTractionBoundaryMfem_forward_Julia(double *out, // 2bdN
        const double *sign,
        const double *gN, 
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order){
        MFEM::ComputeBDMTractionBoundaryMfem_forward(out, sign, gN, bdnode_x, bdnode_y, bdN, order);
}