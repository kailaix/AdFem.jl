#include "DataStore.h"
#include <iostream>
Eigen::SparseLU<SpMat> solver, solver2;
static const double pts[] = {(-1/sqrt(3)+1.0)/2.0, (1/sqrt(3)+1.0)/2.0};

void factorize_matrix(int m, int n, double h, int *ind, int N){
    std::vector<T> triplets;
    std::set<int> indset;
    for (int i=0;i<N;i++) indset.insert(ind[i]);
    Eigen::Matrix<double,4,4> Omega;
    Eigen::Matrix<double,2,4> B;
    Eigen::Matrix<double,2,2> K;
    Eigen::VectorXi idx(4);
    
    // std::cout << Omega << std::endl;
    for(int i=0;i<m;i++){
      for(int j=0;j<n;j++){
        idx << j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1;
        for(int ei=0;ei<2;ei++)
          for(int ej=0;ej<2;ej++){
            int ids = 16*(i+j*m) + 4*(ei+ej*2);
      
            // std::cout << "***\n" << K << std::endl;
            double xi = pts[ei], eta = pts[ej];
            B << -1/h*(1-eta), 1/h*(1-eta), -1/h*eta, 1/h*eta,
                  -1/h*(1-xi), -1/h*xi, 1/h*(1-xi), 1/h*xi;
            Omega = B.transpose() * B * 0.25 * h* h;
            for(int p=0;p<4;p++){
              for(int q=0;q<4;q++){
                if (indset.count(idx[p])>0) continue;
                triplets.push_back(T(idx[p], idx[q], Omega(p,q)));
              }
            }
          }
      }
    }
    
    for(int i=0;i<N;i++){
      triplets.push_back(T(ind[i], ind[i], 1.0));
    }
    SpMat A( (m+1)*(n+1), (m+1)*(n+1) );
    

    A.setFromTriplets(triplets.begin(), triplets.end());
    // std::cout << A << std::endl;

    solver.analyzePattern(A);
    solver.factorize(A);
    printf("A Factorized!");

    SpMat At = A.transpose();
    solver2.analyzePattern(At);
    solver2.factorize(At);
    printf("At Factorized!");
}