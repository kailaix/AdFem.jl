#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/Dense"
#include <vector>
#include <map>
#include <utility>
#include <tuple>
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace mfem;

enum FiniteElementType {P1, P2, BDM1};

class NNFEM_Element{
public:
    NNFEM_Element(int ngauss, int NDOF, FiniteElementType fet);
    // elem_ndof x ngauss matrices
    MatrixXd h; 
    MatrixXd hx;
    MatrixXd hy;
    // 3 x ngauss matrix
    MatrixXd hs; 
    // BDM basis functions and div functions
    MatrixXd BDMx;
    MatrixXd BDMy;
    MatrixXd BDMdiv;     
    VectorXd w;
    double area;
    MatrixXd coord;
    Eigen::Matrix3d Coef; // [a1 a2 a3; b1 b2 b3; c1 c2 c3], ai * x + bi * y + ci
    int nnode;
    int ngauss;
    int dof[6]; // local degrees of freedom, e.g., dof[3] is the global index of 4-th local DOF
    int node[3]; // global indices of local vertices
    int edge[3]; // global indices of local edges
    int ndof;
};

class NNFEM_Mesh{
public:
    long long* init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, int _degree, long long *nedges_ptr);
    long long* init_BDM1(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, long long *nedges_ptr);
    ~NNFEM_Mesh();
    int nelem;
    int nnode;
    int ngauss;
    int ndof; // total number of dofs
    int order; // integration order 
    int degree; // Degree of Polynomials, 1 - P1 element, 2 - P2 element, -1 - BDM1
    int elem_ndof; // 3 for P1, 6 for P2
    MatrixXd GaussPts;
    Eigen::MatrixXd nodes;
    std::vector<NNFEM_Element*> elements;

    // Data preprocessing arrays
    std::map<std::pair<int, int>, std::tuple<int, int, int>> edge_to_elem;
};

const double LineIntegralWeights[] = {
 0.1739274225687269,
 0.32607257743127305,
 0.32607257743127305,
 0.1739274225687269,
};
const double LineIntegralNode[] = {
    0.06943184420297371,
    0.33000947820757187,
    0.6699905217924281,
    0.9305681557970262
};
const int LineIntegralN = 4;

extern "C" int get_LineIntegralN();
extern NNFEM_Mesh mmesh;

