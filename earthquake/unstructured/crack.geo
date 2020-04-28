Point(1) = {0, 0, 0, 1.0};
Point(2) = {2, 0, 0, 1.0};
Point(3) = {0, 1, 0, 1.0};
Point(4) = {2, 1, 0, 1.0};
Point(5) = {0.5, 0, 0, 1.0};
Point(6) = {1.5, 0.5, 0, 1.0};
//+
Line(1) = {3, 1};
//+
Line(2) = {1, 5};
//+
Line(3) = {5, 6};
//+
Line(4) = {5, 2};
//+
Line(5) = {2, 4};
//+
Line(6) = {4, 3};
//+
Curve Loop(1) = {6, 1, 2, 4, 5};
//+
Plane Surface(1) = {1};
Line{3} In Surface {1};
Mesh.RecombinationAlgorithm = 2;
Recombine Surface {1};
Mesh.SubdivisionAlgorithm = 2;
RefineMesh;
//+
Physical Curve("Dirichlet") = {6, 5, 1, 2};
//+
Physical Curve("Crack") = {3};
//+
Physical Curve("Neumann") = {4};
