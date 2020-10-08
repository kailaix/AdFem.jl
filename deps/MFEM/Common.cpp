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

// _order: integration order
// _degree: element degrees, 1 for linear element, 2 for quadratic element
long long*  NNFEM_Mesh::init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, int _degree, long long *nedges_ptr)
    {
        order = _order;
        degree = _degree;
        nelem = num_elements;
        nnode = num_vertices;
        std::unique_ptr<int> attributes(new int[num_elements]);
        for(int i = 0; i< num_elements; i++) attributes.get()[i] = 1;

        Mesh mesh(vertices, num_vertices, element_indices, Geometry::Type::TRIANGLE,
            (int*)attributes.get(), num_elements, (int *)nullptr, Geometry::Type::SEGMENT, (int *)nullptr, 0, 2, 2);
        mesh.FinalizeTriMesh(1, 0, true);

        IntegrationRules rule_;
        IntegrationRule rule = rule_.Get(Element::Type::TRIANGLE, order);
        ngauss = rule.GetNPoints() * num_elements;
        auto w = rule.GetWeights();
        H1_FECollection *fec = new H1_FECollection(_degree, 2);
        FiniteElementSpace fespace(&mesh, fec);
        
        elem_ndof = (_degree == 1) ? 3 : 6; // _degree == 1 or 2
        int nedges = mesh.GetNEdges();
        *nedges_ptr = nedges;
        ndof = (_degree==1) ? nnode: (nnode + nedges);
        Vector shape(elem_ndof);
        DenseMatrix dshape(elem_ndof, 2); // elem_ndof is the number of basis functions
        GaussPts.resize(ngauss, 2);
        
        int i_gp = 0;
        Array<int> edges_, cor, vtx;
        for (int e = 0; e<mesh.GetNE(); e++){
            // constructing the element
            auto element = new NNFEM_Element(rule.GetNPoints(), elem_ndof);

            for(int k = 0; k<3; k++){
                mesh.GetElementVertices(e, vtx);
                element->node[k] = vtx[k];
                element->dof[k] = vtx[k];
            } 
                
            for(int k = 0; k<3; k++){
                mesh.GetElementEdges(e, edges_, cor);
                element->edge[k] = edges_[k];
                element->dof[k+3] = edges_[k] + nnode;
            }
                
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

                // collect Gauss points
                double x1 = coord1[0], y1 = coord1[1], x2 = coord2[0], y2 = coord2[1], x3 = coord3[0], y3 = coord3[1];
                element->hs(0, i) = 1-ip.x-ip.y; 
                element->hs(1, i) = ip.x; 
                element->hs(2, i) = ip.y;

                GaussPts(i_gp, 0) = x1 * element->hs(0, i) + x2 * element->hs(1, i) + x3 * element->hs(2, i);
                GaussPts(i_gp, 1) = y1 * element->hs(0, i) + y2 * element->hs(1, i) + y3 * element->hs(2, i);
                i_gp++;
                

                // set element weight
                element->w[i] = ip.weight * element->area/0.5;

                eltrans->SetIntPoint(&ip);
                fe->CalcPhysShape(*eltrans, shape); // shape function
                fe->CalcPhysDShape(*eltrans, dshape); // shape derivative functions

                for (int k = 0; k<elem_ndof;k++){
                    element->h(k, i) = shape[k];
                    element->hx(k, i) = dshape(k, 0);
                    element->hy(k, i) = dshape(k, 1);
                }
                
            }
            
            elements.push_back(element);
        }

        Array<int> vert;
        long long * edges = new long long[2*nedges];
        for (int i = 0; i < nedges; i++){
            mesh.GetEdgeVertices(i, vert);
            edges[i] = std::min(vert[0], vert[1]) + 1;
            edges[nedges+i] = std::max(vert[0], vert[1]) + 1;
        }
        return edges;
    
}


long long*  NNFEM_Mesh::init_BDM1(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, long long *nedges_ptr)
    {
        order = _order;
        nelem = num_elements;
        nnode = num_vertices;
        std::unique_ptr<int> attributes(new int[num_elements]);
        for(int i = 0; i< num_elements; i++) attributes.get()[i] = 1;

        Mesh mesh(vertices, num_vertices, element_indices, Geometry::Type::TRIANGLE,
            (int*)attributes.get(), num_elements, (int *)nullptr, Geometry::Type::SEGMENT, (int *)nullptr, 0, 2, 2);
        mesh.FinalizeTriMesh(1, 0, true);

        IntegrationRules rule_;
        IntegrationRule rule = rule_.Get(Element::Type::TRIANGLE, order);
        ngauss = rule.GetNPoints() * num_elements;
        auto w = rule.GetWeights();
        elem_ndof = 6;
        int nedges = mesh.GetNEdges();
        *nedges_ptr = nedges;
        ndof = 2 * nedges;
        GaussPts.resize(ngauss, 2);
        
        int i_gp = 0;
        Array<int> edges_, cor, vtx;
        for (int e = 0; e<mesh.GetNE(); e++){
            // constructing the element
            auto element = new NNFEM_Element(rule.GetNPoints(), elem_ndof);

            for(int k = 0; k<3; k++){
                mesh.GetElementVertices(e, vtx);
                element->node[k] = vtx[k]; // vertex dof 
            } 
            
            mesh.GetElementEdges(e, edges_, cor);
            for(int k = 0; k<3; k++){
                mesh.GetEdgeVertices(edges_[k], vtx);
                element->edge[k] = edges_[k];
                if (vtx[0]>vtx[1]){
                    element->dof[2*k] = edges_[k];
                    element->dof[2*k + 1] = edges_[k] + nedges;
                }else{
                    element->dof[2*k] = edges_[k] + nedges;
                    element->dof[2*k + 1] = edges_[k];
                }
            }
                
            int *idx = mesh.GetElement(e)->GetVertices();
            double *coord1 = mesh.GetVertex(idx[0]);
            double *coord2 = mesh.GetVertex(idx[1]);
            double *coord3 = mesh.GetVertex(idx[2]);
            element->coord << coord1[0], coord1[1],
                                coord2[0], coord2[1],
                                coord3[0], coord3[1];
            element->area = heron(element->coord);

            for (int i = 0; i < rule.GetNPoints(); i++){
                const IntegrationPoint &ip = rule.IntPoint(i);

                // collect Gauss points
                double x1 = coord1[0], y1 = coord1[1], x2 = coord2[0], y2 = coord2[1], x3 = coord3[0], y3 = coord3[1];
                element->hs(0, i) = 1-ip.x-ip.y; 
                element->hs(1, i) = ip.x; 
                element->hs(2, i) = ip.y;

                GaussPts(i_gp, 0) = x1 * element->hs(0, i) + x2 * element->hs(1, i) + x3 * element->hs(2, i);
                GaussPts(i_gp, 1) = y1 * element->hs(0, i) + y2 * element->hs(1, i) + y3 * element->hs(2, i);
                i_gp++;
                

                // set element weight
                element->w[i] = ip.weight * element->area/0.5;

                for (int k = 0; k<elem_ndof;k++){
                    element->BDMx(k, i) = (x2-x3)*ip.x + (x1-x3)*ip.y + x3;
                    element->BDMy(k, i) = (y2-y3)*ip.x + (y1-y3)*ip.y + y3;
                    
                }
                double J = (x2-x3)*(y1-y3) - (x1-x3)*(y2-y3);
                double divxi = (y1-y3+x3-x1)/J, diveta = (y3-y2+x2-x3)/J;
                double g1 = 0.5 - sqrt(3.0)/6, g2 = 0.5 + sqrt(3.0)/6;
                element->BDMdiv(0, i) = sqrt(2.0)/(g2 - g1) * (g2 * divxi + (g2-1)*diveta);
                element->BDMdiv(1, i) = sqrt(2.0)/(g1 - g2) * (g1 * divxi + (g1-1)*diveta);
                element->BDMdiv(2, i) = 1.0/(g2-g1) * (g2 * divxi + (g2-1)*diveta);
                element->BDMdiv(3, i) = 1.0/(g1-g2) * (g1 * divxi + (g1-1)*diveta);
                element->BDMdiv(4, i) = 1.0/(g2-g1) * ((g2-1) * divxi + g2*diveta);
                element->BDMdiv(5, i) = 1.0/(g1-g2) * ((g1-1) * divxi + g1*diveta);
                
            }
            
            elements.push_back(element);
        }

        Array<int> vert;
        long long * edges = new long long[2*nedges];
        for (int i = 0; i < nedges; i++){
            mesh.GetEdgeVertices(i, vert);
            edges[i] = std::min(vert[0], vert[1]) + 1;
            edges[nedges+i] = std::max(vert[0], vert[1]) + 1;
        }
        return edges;
}

NNFEM_Mesh::~NNFEM_Mesh(){
    for(int i = 0; i<nelem; i++){
        delete elements[i];
    }
}

NNFEM_Element::NNFEM_Element(int ngauss, int ndof): ngauss(ngauss), ndof(ndof), nnode(3){
    h.resize(ndof, ngauss);
    hx.resize(ndof, ngauss);
    hy.resize(ndof, ngauss);
    hs.resize(3, ngauss);
    w.resize(ngauss);
    coord.resize(ngauss, 2);
}
