#include "Common.h"
#include <memory>
using namespace mfem;
using Eigen::VectorXd;
using Eigen::MatrixXd;

NNFEM_Mesh3 mmesh3;

long long*  NNFEM_Mesh3::init(double *vertices, int num_vertices, 
                int *element_indices, int num_elements, int _order, int _degree, long long *nedges_ptr)
    {
        order = _order;
        degree = _degree;
        nelem = num_elements;
        nnode = num_vertices;
        std::unique_ptr<int> attributes(new int[num_elements]);
        for(int i = 0; i< num_elements; i++) attributes.get()[i] = 1;

        Mesh mesh(vertices, num_vertices, element_indices, Geometry::Type::TETRAHEDRON,
            (int*)attributes.get(), num_elements, 
            (int *)nullptr, Geometry::Type::TRIANGLE, (int *)nullptr, 0, 3, 3);
        mesh.FinalizeTetMesh(1, 0, true);

        IntegrationRules rule_;
        IntegrationRule rule = rule_.Get(Element::Type::TETRAHEDRON, order);
        ngauss = rule.GetNPoints() * num_elements;
        auto w = rule.GetWeights();

        H1_FECollection *fec1 = new H1_FECollection(1, 3);
        FiniteElementSpace fespace1(&mesh, fec1);

        H1_FECollection *fec = new H1_FECollection(degree, 3);
        FiniteElementSpace fespace(&mesh, fec);

        
        
        
        int nedges = mesh.GetNEdges();
        *nedges_ptr = nedges;
        
        switch (degree)
        {
        case 1:
            elem_ndof = 4;
            ndof = nnode;
            break;
        case 2:
            elem_ndof = 10;
            ndof = nnode + nedges;
            break;
        default:
            throw "degree must be equal to 1 or 2.\n";
            break;
        }


        Vector shape(elem_ndof);
        DenseMatrix dshape(elem_ndof, 3); // elem_ndof is the number of basis functions
        GaussPts.resize(ngauss, 3);
        
        int i_gp = 0;
        Array<int> edges_, cor, vtx;
        for (int e = 0; e<mesh.GetNE(); e++){
            // constructing the element
            auto element = new NNFEM_Element3(rule.GetNPoints(), elem_ndof);

            for(int k = 0; k<4; k++){
                mesh.GetElementVertices(e, vtx);
                element->node[k] = vtx[k];
                element->dof[k] = vtx[k];
            } 
                
            for(int k = 0; k<6; k++){
                mesh.GetElementEdges(e, edges_, cor);
                element->edge[k] = edges_[k];
                element->dof[k+4] = edges_[k] + nnode;
            }
                
            int *idx = mesh.GetElement(e)->GetVertices();
            double *coord1 = mesh.GetVertex(idx[0]);
            double *coord2 = mesh.GetVertex(idx[1]);
            double *coord3 = mesh.GetVertex(idx[2]);
            double *coord4 = mesh.GetVertex(idx[3]);
            element->coord << coord1[0], coord1[1], coord1[2],
                                    coord2[0], coord2[1], coord2[2],
                                    coord3[0], coord3[1], coord3[2],
                                    coord4[0], coord4[1], coord4[2];
            element->volume = mesh.GetElementVolume(e);


            const FiniteElement *fe = fespace.GetFE(e);
            ElementTransformation* eltrans = fespace.GetElementTransformation(e);

            const FiniteElement *fe1 = fespace1.GetFE(e);
            ElementTransformation* eltrans1 = fespace1.GetElementTransformation(e);
            for (int i = 0; i < rule.GetNPoints(); i++){
                const IntegrationPoint &ip = rule.IntPoint(i);

                eltrans1->SetIntPoint(&ip);
                fe1->CalcPhysShape(*eltrans1, shape); // shape function
                for (int j = 0; j < 4; j++){
                    element->hs(j, i) = shape[j];
                }
                    

                double x1 = coord1[0], y1 = coord1[1], z1 = coord1[2],
                        x2 = coord2[0], y2 = coord2[1], z2 = coord2[2],
                        x3 = coord3[0], y3 = coord3[1], z3 = coord3[2],
                        x4 = coord4[0], y4 = coord4[1], z4 = coord4[2];
                
            
            
                GaussPts(i_gp, 0) = x1 * element->hs(0, i) + x2 * element->hs(1, i) + x3 * element->hs(2, i) + x4 * element->hs(3, i);
                GaussPts(i_gp, 1) = y1 * element->hs(0, i) + y2 * element->hs(1, i) + y3 * element->hs(2, i) + y4 * element->hs(3, i);
                GaussPts(i_gp, 2) = z1 * element->hs(0, i) + z2 * element->hs(1, i) + z3 * element->hs(2, i) + z4 * element->hs(3, i);
                i_gp++;
                element->w[i] = ip.weight * element->volume*6.0;

                eltrans->SetIntPoint(&ip);
                fe->CalcPhysShape(*eltrans, shape); // shape function
                fe->CalcPhysDShape(*eltrans, dshape); // shape derivative functions

                for (int k = 0; k<elem_ndof;k++){
                    element->h(k, i) = shape[k];
                    element->hx(k, i) = dshape(k, 0);
                    element->hy(k, i) = dshape(k, 1);
                    element->hz(k, i) = dshape(k, 2);
                    // printf("hz = %f %f %f\n", element->hx(k, i), element->hy(k, i), element->hz(k, i));
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

        delete fec1;
        delete fec;
        return edges;
    
}

NNFEM_Mesh3::~NNFEM_Mesh3(){
    for(int i = 0; i<nelem; i++){
        delete elements[i];
    }
}

NNFEM_Element3::NNFEM_Element3(int ngauss, int ndof): ngauss(ngauss), ndof(ndof), nnode(4){
    h.resize(ndof, ngauss);
    hx.resize(ndof, ngauss);
    hy.resize(ndof, ngauss);
    hz.resize(ndof, ngauss);
    hs.resize(4, ngauss);
    w.resize(ngauss);
    coord.resize(4, 3); // each row is a coordinate for a vertex
}
