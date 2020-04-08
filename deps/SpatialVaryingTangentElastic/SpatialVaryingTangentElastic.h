void forward(double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    for(int i=0;i<4*m*n;i++){
      hmat[k++] = mu[i];
      hmat[k++] = 0.;
      hmat[k++] = 0.;
      hmat[k++] = mu[i];
    }
}

void forward2(double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    int offset = 4*m*n;
    for(int i=0;i<4*m*n;i++){
      hmat[k++] = mu[i];
      hmat[k++] = 0.;
      hmat[k++] = 0.;
      hmat[k++] = mu[i+offset];
    }
}

void forward3(double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    int offset = 4*m*n;
    for(int i=0;i<4*m*n;i++){
      hmat[k++] = mu[i];
      hmat[k++] = mu[i+2*offset];
      hmat[k++] = mu[i+2*offset];
      hmat[k++] = mu[i+offset];
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

void backward2(
  double *grad_mu, 
  const double *grad_hmat,
  const double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    int offset = 4*m*n;
    for(int i=0;i<4*m*n;i++){
      grad_mu[i] = grad_hmat[k];
      grad_mu[i + 4*m*n] = grad_hmat[k+3];
      k += 4;
    }
}

void backward3(
  double *grad_mu, 
  const double *grad_hmat,
  const double *hmat, const double *mu, int m, int n, double h){
    int k = 0;
    int offset = 4*m*n;
    for(int i=0;i<4*m*n;i++){
      grad_mu[i] = grad_hmat[k];
      grad_mu[i + 4*m*n] = grad_hmat[k+3];
      grad_mu[i + 2*4*m*n] = grad_hmat[k+1] + grad_hmat[k+2];
      k += 4;
    }
}