#include <Eigen/Dense>
#include <Eigen/Dense>

namespace MFEM{
    void SolveSlipLawForward(double *x_, 
        const double *x0, 
        const double *a_, const double *b_, const double *c_, int n){
        Eigen::ArrayXd x(n), res(n), dres(n), delta;
        for(int i = 0; i < n; i++) x[i] = x0[i];
        Eigen::Map<const Eigen::ArrayXd> a(a_, n);
        Eigen::Map<const Eigen::ArrayXd> b(b_, n);
        Eigen::Map<const Eigen::ArrayXd> c(c_, n);
        double tol = 1.0;
        int iter = 0; 
        while (tol > 1e-5 && iter < 50){
            res = x - a * (b * x).asinh() - c;
            tol = res.matrix().norm();
            dres = 1.0 - a * b/(1.0 + (b*x).square()).sqrt();
            x = x - res/dres;
            iter++;
        }
        for(int i = 0; i < n; i++) x_[i] = x[i];
    }

    void SolveSlipLawBackward(
        double *grad_a, double *grad_b, double *grad_c, 
        const double *grad_x, 
        const double *x_, 
        const double *a_, const double *b_, const double *c_, int n){
        Eigen::Map<Eigen::ArrayXd> ga(grad_a, n);
        Eigen::Map<Eigen::ArrayXd> gb(grad_b, n);
        Eigen::Map<Eigen::ArrayXd> gc(grad_c, n);

        Eigen::Map<const Eigen::ArrayXd> x(x_, n);
        Eigen::Map<const Eigen::ArrayXd> a(a_, n);
        Eigen::Map<const Eigen::ArrayXd> b(b_, n);
        Eigen::Map<const Eigen::ArrayXd> c(c_, n);
        Eigen::Map<const Eigen::ArrayXd> u(grad_x, n);
        Eigen::ArrayXd temp = - u / ( 1 - a * b / ( 1 + (b * x).square() ).sqrt() );
        
        ga = - temp * (b * x).asinh();
        gc = - temp;
        gb = - temp * a * x / ( 1 + (b * x).square()).sqrt();
        
    }
}