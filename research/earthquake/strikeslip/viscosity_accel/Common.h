#include <map>
#include <string>
#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/Dense>

#include <set>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

extern Eigen::SparseLU<SpMat> solver, solvert;
extern std::vector<int> iiK, jjK; 
extern int count;




