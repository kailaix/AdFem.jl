namespace MFEM{
    void EvalNormalAndShearStressOnBoundaryEdgeForward(
        double *sn,
        double *st,
        const double *sigma,
        const double *normal,
        int N 
    ){
        for (int i = 0; i < N; i++){
            double sxx = sigma[3*i], syy = sigma[3*i+1], sxy = sigma[3*i+2];
            double n1 = normal[2*i], n2 = normal[2*i+1];
            sn[i] = sxx * n1 * n1 + syy * n2 * n2 + 2 * sxy * n1 * n2;
            st[i] = sxx * n1 * n2 - syy * n1 * n2 + sxy * (n2 * n2 - n1 * n1);
        }
    }


    void EvalNormalAndShearStressOnBoundaryEdgeBackward(
        double *grad_sigma,
        const double *grad_sn,
        const double *grad_st,
        const double *sn,
        const double *st,
        const double *sigma,
        const double *normal,
        int N 
    ){
        for (int i = 0; i < N; i++){
            double sxx = sigma[3*i], syy = sigma[3*i+1], sxy = sigma[3*i+2];
            double n1 = normal[2*i], n2 = normal[2*i+1];
            
            grad_sigma[3*i] = n1 * n1 * grad_sn[i] + n1 * n2 * grad_st[i];
            grad_sigma[3*i+1] = n2 * n2 * grad_sn[i] - n1 * n2 *  grad_st[i];
            grad_sigma[3*i+2] = n1 * n2 * grad_sn[i] + (n2 * n2 - n1 * n1) * grad_st[i];
        }
    }
}