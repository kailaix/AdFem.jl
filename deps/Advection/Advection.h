// v1 v2
// v3 v4
void forward(double *a, const double *v, const double *u, int m, int n, double h){
  for(int i=0;i<m*n;i++) a[i] = 0.0;
  for(int j=0;j<n;j++){
    for(int i=0;i<m;i++){
      double vx = v[2*(j*m+i)];
      double vy = v[2*(j*m+i)+1];

      //------------------------------------------
      if(vx>=0){
        if (i==0){
          a[j*m+i] += vx * 2 * h * u[j*m+i];
        }
        else{
          a[j*m+i] += vx * h * (u[j*m+i] - u[j*m+i-1]);
        }
      }
      else{
        if (i==m-1){
          a[j*m+i] += - vx * 2 * h * u[j*m+i];
        }
        else{
          a[j*m+i] += vx * h * (u[j*m+i+1] - u[j*m+i]);
        }
      }

      //------------------------------------------
      if (vy>=0){
        if(j==0){
            a[j*m+i] += vy * 2 * h * u[j*m+i];
        }
        else{
            a[j*m+i] += vy * h * (u[j*m+i] - u[(j-1)*m+i]);
        }
      }else{
        if (j==n-1){
            a[j*m+i] += - vy * 2 * h * u[j*m+i];
        }
        else{
            a[j*m+i] += vy * h * (u[(j+1)*m+i] - u[j*m+i]);
        }
      }
    }
  }
}

void backward(
  double *grad_v, double *grad_u, 
  const double*grad_a, 
  const double *a, const double *v, const double *u, int m, int n, double h){  
      for(int i=0;i<2*m*n;i++) grad_v[i] = 0.0;
      for(int i=0;i<m*n;i++) grad_u[i] = 0.0;
      for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
          double vx = v[2*(j*m+i)];
          double vy = v[2*(j*m+i)+1];

          //------------------------------------------
          if(vx>=0){
            if (i==0){
              grad_v[2*(j*m+i)] += 2 * h * u[j*m+i] * grad_a[j*m+i];
              grad_u[j*m+i] += 2 * h * vx * grad_a[j*m+i];
            }
            else{
              // a[j*m+i] += vx * h * (u[j*m+i] - u[j*m+i-1]);
              grad_v[2*(j*m+i)] += h * (u[j*m+i] - u[j*m+i-1]) * grad_a[j*m+i];
              grad_u[j*m+i] += vx * h * grad_a[j*m+i];
              grad_u[j*m+i-1] += - vx * h * grad_a[j*m+i];
            }
          }
          else{
            if (i==m-1){
              // a[j*m+i] += - vx * 2 * h * u[j*m+i];
              grad_v[2*(j*m+i)] += - 2 * h * u[j*m+i] * grad_a[j*m+i];
              grad_u[j*m+i] += - 2 * h * vx * grad_a[j*m+i];
            }
            else{
              // a[j*m+i] += vx * h * (u[j*m+i+1] - u[j*m+i]);
              grad_v[2*(j*m+i)] += h * (u[j*m+i+1] - u[j*m+i]) * grad_a[j*m+i];
              grad_u[j*m+i+1] += vx * h * grad_a[j*m+i];
              grad_u[j*m+i] += - vx * h * grad_a[j*m+i];
            }
          }

          //------------------------------------------
          if (vy>=0){
            if(j==0){
                // a[j*m+i] += vy * 2 * h * u[j*m+i];
                grad_v[2*(j*m+i)+1] += 2 * h * u[j*m+i] * grad_a[j*m+i];
                grad_u[j*m+i] += 2 * h * vy * grad_a[j*m+i];

            }
            else{
                // a[j*m+i] += vy * h * (u[j*m+i] - u[(j-1)*m+i]);
                grad_v[2*(j*m+i)+1] += h * (u[j*m+i] - u[(j-1)*m+i]) * grad_a[j*m+i];
                grad_u[j*m+i] += h * vy * grad_a[j*m+i];
                grad_u[(j-1)*m+i] += - h * vy * grad_a[j*m+i];
            }
          }else{

            if (j==n-1){
                // a[j*m+i] += - vy * 2 * h * u[j*m+i];
                grad_v[2*(j*m+i)+1] += - 2 * h * u[j*m+i] * grad_a[j*m+i];
                grad_u[j*m+i] += - 2 * h * vy * grad_a[j*m+i];
            }
            else{
                // a[j*m+i] += vy * h * (u[(j+1)*m+i] - u[j*m+i]);
                grad_v[2*(j*m+i)+1] += h * (u[(j+1)*m+i] - u[j*m+i]) * grad_a[j*m+i];
                grad_u[(j+1)*m+i] += h * vy * grad_a[j*m+i];
                grad_u[j*m+i] += - h * vy * grad_a[j*m+i];
            }
          }
        }
      }
}