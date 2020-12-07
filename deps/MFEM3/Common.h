#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "eigen3/Eigen/Core"
#include <vector>
using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using namespace mfem;

enum FiniteElementType3 {P1, P2};

class NNFEM_Element3{
public:
    NNFEM_Element3(int ngauss, int NDOF);

    // elem_ndof x ngauss matrices
    MatrixXd h;  // ndof x ngauss
    MatrixXd hx; // ndof x ngauss
    MatrixXd hy; // ndof x ngauss
    MatrixXd hz; // ndof x ngauss

    // 4 x ngauss
    MatrixXd hs; 
    // ngauss 
    VectorXd w;

    double volume;

    int nnode; // = 4 
    int ngauss; 
    int dof[10]; // local degrees of freedom, e.g., dof[3] is the global index of 4-th local DOF
    int node[4]; // global indices of local vertices
    int edge[6]; // global indices of local edges
    int ndof; // P1: 4, P2: 10
    MatrixXd coord;
};

class NNFEM_Mesh3{
public:
    /* 
        *nedges_ptr =  total number of edges 
        returns the pointer to edges

        vertices : nnode x 2
        element_indices : nelem x 4
        _order : 1 or 2 
        _degree : an integer 

    */
    long long* init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, int _degree, long long *nedges_ptr);
    ~NNFEM_Mesh3();
    int nelem;
    int nnode;
    int ngauss;
    int ndof; // total number of dofs
    int order; // integration order 
    int degree; // Degree of Polynomials, 1 - P1 element, 2 - P2 element, -1 - BDM1
    int elem_ndof; // 4 for P1, 10 for P2
    MatrixXd GaussPts;
    std::vector<NNFEM_Element3*> elements;
};

extern NNFEM_Mesh3 mmesh3;

