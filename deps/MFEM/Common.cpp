#include "Common.h"
#include <memory>
using namespace mfem;
using Eigen::VectorXd;
using Eigen::MatrixXd;

NNFEM_Mesh mmesh;

double heron(const MatrixXd& coord){
    double a = (coord.row(0) - coord.row(1)).norm();
    double b = (coord.row(2) - coord.row(1)).norm();
    double c = (coord.row(0) - coord.row(2)).norm();
    double s = (a+b+c)/2.0;
    return sqrt(s*(s-a)*(s-b)*(s-c));
}

void NNFEM_Mesh::init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order)
    {
    order = _order;
    nelem = num_elements;
    nnode = num_vertices;
    std::unique_ptr<int> attributes(new int[num_elements]);
    for(int i = 0; i< num_elements; i++) attributes.get()[i] = 1;
    Mesh mesh(vertices, num_vertices, element_indices, Geometry::Type::TRIANGLE,
        (int*)attributes.get(), num_elements, (int *)nullptr, Geometry::Type::SEGMENT, (int *)nullptr, 0, 1, 2);

    IntegrationRules rule_;
    IntegrationRule rule = rule_.Get(Element::Type::TRIANGLE, order);
    ngauss = rule.GetNPoints() * num_elements;
    auto w = rule.GetWeights();
    FiniteElementCollection *fec = new H1_FECollection(1, 2);
    FiniteElementSpace fespace(&mesh, fec);
    Vector shape(3);
    Vector gauss_pts(2);
    DenseMatrix dshape(3, 2); // 3 is the number of basis functions
    GaussPts.resize(ngauss, 2);
    int i_gp = 0;
    for (int e = 0; e<mesh.GetNE(); e++){

        // constructing the element
        auto element = new NNFEM_Element(rule.GetNPoints());
        memcpy(element->node, element_indices + 3 * e, sizeof(int)*3);
        int *idx = mesh.GetElement(e)->GetVertices();
        double *coord1 = mesh.GetVertex(idx[0]);
        double *coord2 = mesh.GetVertex(idx[1]);
        double *coord3 = mesh.GetVertex(idx[2]);
        element->coord << coord1[0], coord1[1],
                            coord2[0], coord2[1],
                            coord3[0], coord3[1];
        element->area = heron(element->coord);


        const FiniteElement *fe = fespace.GetFE(e);
        ElementTransformation* eltrans = fespace.GetElementTransformation(e);
        for (int i = 0; i < rule.GetNPoints(); i++){
            const IntegrationPoint &ip = rule.IntPoint(i);

            // set element weight
            element->w[i] = ip.weight * element->area/0.5;

            

            eltrans->SetIntPoint(&ip);
            fe->CalcPhysShape(*eltrans, shape); // shape function
            fe->CalcPhysDShape(*eltrans, dshape); // Dshape function
            // printf("shape = (%f, %f, %f)\n", shape[0], shape[1], shape[2]);
            for (int k = 0; k<3;k++){
                element->h(k, i) = shape[k];
                element->hx(k, i) = dshape(k, 0);
                element->hy(k, i) = dshape(k, 1);
            }

            // collect Gauss points
            double x1 = coord1[0], y1 = coord1[1], x2 = coord2[0], y2 = coord2[1], x3 = coord3[0], y3 = coord3[1];
            GaussPts(i_gp, 0) = x1 * shape[0] + x2 * shape[1] + x3 * shape[2];
            GaussPts(i_gp, 1) = y1 * shape[0] + y2 * shape[1] + y3 * shape[2];
            i_gp++;
            
        }
        
        elements.push_back(element);
    }
    
}

NNFEM_Mesh::~NNFEM_Mesh(){
    for(int i = 0; i<nelem; i++){
        delete elements[i];
    }
}

NNFEM_Element::NNFEM_Element(int ngauss): ngauss(ngauss){
    h.resize(3, ngauss);
    hx.resize(3, ngauss);
    hy.resize(3, ngauss);
    w.resize(ngauss);
    coord.resize(ngauss, 2);
    nnode = 3;
}


extern "C" void init_nnfem_mesh(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int order){
    if (mmesh.elements.size()>0){
        printf("WARNING: Internal mesh is being overwritten!\n");
        for(int i = 0; i< mmesh.nelem; i++) delete mmesh.elements[i];
        mmesh.elements.clear();
    }
    mmesh.init(vertices, num_vertices, element_indices, num_elements, order);
}

extern "C" int mfem_get_ngauss(){
    return mmesh.ngauss;
}

extern "C" void mfem_get_gauss(double *x, double *y){
   memcpy(x, mmesh.GaussPts.data(), mmesh.ngauss * sizeof(double));
   memcpy(y, mmesh.GaussPts.data() + mmesh.ngauss, mmesh.ngauss * sizeof(double));
}

extern "C" void mfem_get_area(double *a){
   for(int i = 0; i<mmesh.nelem; i++)
    a[i] = mmesh.elements[i]->area;
}