#include "DataStore.h"
void forward(double *u, const double *rhs, int n){
  Eigen::VectorXd RHS(n);
  for(int i=0;i<n;i++) RHS[i] = rhs[i];
  Eigen::VectorXd x = solver.solve(RHS);
  for(int i=0;i<n;i++) u[i] = x[i];
}

void backward(double *grad_rhs, const double * grad_u, 
  const double *u, const double *rhs, int n){
    Eigen::VectorXd G(n);
    for(int i=0;i<n;i++) G[i] = grad_u[i];
    Eigen::VectorXd x = solver2.solve(G);
    for(int i=0;i<n;i++) grad_rhs[i] = x[i];
}