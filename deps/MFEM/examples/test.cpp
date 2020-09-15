#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
   Mesh mesh(3, 3, Element::Type::TRIANGLE);
   for (int i = 0; i < mesh.GetNE(); i++){
       Element *elem = mesh.GetElement(i);
       int nv = elem->GetNVertices();
       int *idx = elem->GetVertices();
        printf("Element %d has %d vertices: %d %d %d\n", i, nv, idx[0], idx[1], idx[2]);
        for(int k = 0; k<3; k++){
            double *coord = mesh.GetVertex(idx[k]);
            printf("Coord: (%f %f)\n", coord[0], coord[1]);
        }
        
   }

   IntegrationRules rule_;
   IntegrationRule rule = rule_.Get(Element::Type::TRIANGLE, 2);
   printf("Quadrature rule has %d points\n", rule.GetNPoints());
   auto w = rule.GetWeights();
   printf("Weights = %f %f %f\n", w[0], w[1], w[2]);
   for (int i = 0; i<3; i++){
       auto point = rule.IntPoint(i);
       printf("Point %d: (%f, %f), w = %f\n", i, point.x, point.y, point.weight);
   }

// H1 finite element (CalcPhysDShape exists)
   FiniteElementCollection *fec = new H1_FECollection(1, 2);
   FiniteElementSpace fespace(&mesh, fec);

// focus on element 3
   Vector shape;
   DenseMatrix dshape(3, 2);
   int e = 3;
   const FiniteElement *fe = fespace.GetFE(e);
   ElementTransformation* eltrans = fespace.GetElementTransformation(e);

   for (int i = 0; i < rule.GetNPoints(); i++){
       const IntegrationPoint &ip = rule.IntPoint(i);
       eltrans->SetIntPoint(&ip);
       w = ip.weight * eltrans->Weight();
       fe->CalcPhysShape(*eltrans, shape); // shape function
       fe->CalcPhysDShape(*eltrans, dshape); // Dshape function
       printf("--------------------------------");
       shape.Print();
       dshape.Print();
       w.Print();
   }


}