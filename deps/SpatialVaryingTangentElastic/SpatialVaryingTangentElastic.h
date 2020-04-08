void forward(double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    for(int i=0;i<4*m*n;i++){
      hmat[k++] = mu[i];
      hmat[k++] = 0.;
      hmat[k++] = 0.;
      hmat[k++] = mu[i];
    }
}

void backward(
  double *grad_mu, 
  const double *grad_hmat,
  const double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    for(int i=0;i<4*m*n;i++){
      grad_mu[i] = grad_hmat[k] + grad_hmat[k+3];
      k += 4;
    }
}