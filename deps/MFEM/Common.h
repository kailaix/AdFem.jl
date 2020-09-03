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
    NNFEM_Element(int ngauss);
    VectorXd h;
    VectorXd hx;
    VectorXd hy;
    VectorXd w;
    double area;
    MatrixXd coord;
    int node[3];
    int nnode;
    int ngauss;
};

class NNFEM_Mesh{
public:
    void init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int order = 2);
    ~NNFEM_Mesh();
    int nelem;
    int nnode;
    int ngauss;
    int order;
    MatrixXd GaussPts;
    std::vector<NNFEM_Element*> elements;
};

extern NNFEM_Mesh mmesh;

