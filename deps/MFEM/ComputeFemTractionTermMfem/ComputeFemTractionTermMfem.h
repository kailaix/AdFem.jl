#include "../Common.h"

namespace MFEM{
    void calcShapeFunction(Eigen::MatrixXd &shape, Eigen::MatrixXd & coords, 
            const Eigen::VectorXd &xs,
            double x1, double y1, double x2, double y2, double x3, double y3){
        if (mmesh.degree == 1){
            double J = (x2*y3 - x3*y2) - (x1*y3-x3*y1) + (x1*y2-x2*y1);
            for (int r = 0; r < xs.size(); r++){
                double x = xs[r] * x1 + (1-xs[r]) * x2;
                double y = xs[r] * y1 + (1-xs[r]) * y2;
                coords(r, 0) = x; 
                coords(r, 1) = y;
                shape(0, r) = (x2*y3-x3*y2 + (y2-y3)*x+(x3-x2)*y)/J;
                shape(1, r) = (x3*y1-x1*y3 + (y3-y1)*x+(x1-x3)*y)/J;
            }
        }
        else if (mmesh.degree == 2){
            double x4 = (x1 + x2)/2;
            double y4 = (y1 + y2)/2;
            double x5 = (x2 + x3)/2;
            double y5 = (y2 + y3)/2;
            double x6 = (x1 + x3)/2;
            double y6 = (y1 + y3)/2;
            double x23 = x2 - x3, y23 = y2 - y3; 
            double x31 = x3 - x1, y31 = y3 - y1;  
            double x46 = x4 - x6, y46 = y4 - y6;  
            double x54 = x5 - x4, y54 = y5 - y4;  
            double x13 = x1 - x3, y13 = y1 - y3; 
            double x21 = x2 - x1, y21 = y2 - y1;  
            double x41 = x4 - x1, y41 = y4 - y1;  
            double x16 = x1 - x6, y16 = y1 - y6; 
            double x24 = x2 - x4, y24 = y2 - y4; 
            double x43 = x4 - x3, y43 = y4 - y3; 
            double x26 = x2 - x6, y26 = y2 - y6; 
            for (int r = 0; r < xs.size(); r++){
                double x = xs[r] * x1 + (1-xs[r]) * x2;
                double y = xs[r] * y1 + (1-xs[r]) * y2;
                coords(r, 0) = x; 
                coords(r, 1) = y;
                shape(0, r) = (x23 * (y - y3) - y23*(x-x3)) * (x46*(y-y6) - y46 * (x-x6))/(x23 * y13 - y23 * x13)/(x46 * y16 - y46 * x16);
                shape(1, r) = (x31 * (y- y1) - y31 * (x-x1)) * (x54 * (y-y4) - y54 * (x-x4))/(x31 * y21 - y31 * x21)/(x54 * y24 - y54 * x24);
                shape(2, r) = (x31 * (y - y1) - y31 * (x-x1))*(x23 * (y - y3) - y23 * (x-x3))/(x31 * y41 - y31 * x41)/(x23 * y43 - y23 * x43);
            }
        }
        else if (mmesh.degree == -1){
            double J = (x2*y3 - x3*y2) - (x1*y3-x3*y1) + (x1*y2-x2*y1);
            double el = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
            for (int r = 0; r < xs.size(); r++){
                double x = xs[r] * x1 + (1-xs[r]) * x2;
                double y = xs[r] * y1 + (1-xs[r]) * y2;
                coords(r, 0) = x; 
                coords(r, 1) = y;
                shape(0, r) = (x2*y3-x3*y2 + (y2-y3)*x+(x3-x2)*y)/J/el;
                shape(1, r) = (x3*y1-x1*y3 + (y3-y1)*x+(x1-x3)*y)/J/el;
            }
        }
        else {
            printf("ERROR: degree = %d is not supported\n", mmesh.degree);
        }
    }


    void ComputeFemTractionTermMfem_forward(double *out, const double *t, 
        const int *dofs,
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order){
        IntegrationRules rule_;
        IntegrationRule rule = rule_.Get(Element::Type::SEGMENT, order);
        int ngauss = rule.GetNPoints();

        Eigen::MatrixXd shape(mmesh.degree+1, ngauss), coords(ngauss, 2); 
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
                for(int p = 0; p < mmesh.degree+1; p++)
                    out[dofs[k*(mmesh.degree+1)+p]] += (dist * w[r]) * shape(p, r) * tval;
            }
        }
    }
}


extern "C" int ComputeFemTractionTermMfem_forward_getNGauss(int order){
    IntegrationRules rule_;
    IntegrationRule rule = rule_.Get(Element::Type::SEGMENT, order);
    int ngauss = rule.GetNPoints();
    return ngauss;
}

extern "C" int ComputeFemTractionTermMfem_forward_getGaussPoints(
        double *x, double *y, 
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order){
    IntegrationRules rule_;
    IntegrationRule rule = rule_.Get(Element::Type::SEGMENT, order);
    int ngauss = rule.GetNPoints();

    int shape_size = -1;
    switch (mmesh.degree)
    {
        case 1:
            shape_size = 2;
            break;
        case 2:
            shape_size = 3;
            break;
        case -1:
            shape_size = 2;
        default:
            break;
    }
    Eigen::MatrixXd shape(shape_size, ngauss), coords(ngauss, 2); 
    Eigen::VectorXd w(ngauss), xs(ngauss);
    for (int i = 0; i < ngauss; i++){
        const IntegrationPoint &ip = rule.IntPoint(i);
        w[i] = ip.weight;
        xs[i] = ip.x;
    }

    
    for (int k = 0; k < bdN; k++){
        
        double x1 = bdnode_x[3*k], x2 = bdnode_x[3*k+1], x3 = bdnode_x[3*k+2];
        double y1 = bdnode_y[3*k], y2 = bdnode_y[3*k+1], y3 = bdnode_y[3*k+2];
        MFEM::calcShapeFunction(shape, coords, xs, x1, y1, x2, y2, x3, y3);
        for(int r = 0; r < ngauss; r++){
            x[k * ngauss + r] = coords(r, 0);
            y[k * ngauss + r] = coords(r, 1);
        }
    }
}

extern "C" void ComputeFemTractionTermMfem_forward_Julia(
        double *out, const double *t, 
        const int *dofs,
        const double *bdnode_x,
        const double *bdnode_y, int bdN, int order){
    MFEM::ComputeFemTractionTermMfem_forward(out, t, dofs, bdnode_x, bdnode_y, bdN, order);
}