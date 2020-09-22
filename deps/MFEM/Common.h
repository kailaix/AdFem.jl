#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "eigen3/Eigen/Core"
#include <vector>
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace mfem;

class NNFEM_Element{
public:
    NNFEM_Element(int ngauss, int NDOF);
    MatrixXd h;
    MatrixXd hx;
    MatrixXd hy;
    VectorXd w;
    double area;
    MatrixXd coord;
    int nnode;
    int ngauss;
    int dof[6]; // local degrees of freedom, e.g., dof[3] is the global index of 4-th local DOF
    int node[3]; // global indices of local vertices
    int edge[3]; // global indices of local edges
    int ndof;
};

class NNFEM_Mesh{
public:
    void init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, int _degree, long long *edges);
    ~NNFEM_Mesh();
    int nelem;
    int nnode;
    int ngauss;
    int ndof; // total number of dofs
    int order; // integration order 
    int degree; // Degree of Polynomials, 0 - P0 element, 1 - P1 element, 2 - P2 element 
    MatrixXd GaussPts;
    std::vector<NNFEM_Element*> elements;
};

extern NNFEM_Mesh mmesh;

