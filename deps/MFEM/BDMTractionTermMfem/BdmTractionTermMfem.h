#include "../Common.h"


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
}

extern "C" void ComputeBDMTractionTermMfem_forward_Julia(
        double *out, const double *sign, const double *t, 
        const int *dofs,
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order
){
    MFEM::ComputeBDMTractionTermMfem_forward(out, sign, t, dofs, bdnode_x, bdnode_y, bdN, order);
}
