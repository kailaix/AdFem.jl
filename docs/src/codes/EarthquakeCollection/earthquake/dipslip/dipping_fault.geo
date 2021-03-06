Point(1) = {0, 0, 0, 1.0};
Point(2) = {8, 0, 0, 1.0};
Point(3) = {0, 4, 0, 1.0};
Point(4) = {8, 4, 0, 1.0};
Point(5) = {3.13, 0, 0, 1.0};
Point(6) = {4.0, 0.5, 0, 1.0};
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

//+
Recombine Surface {1};
//+
Characteristic Length {1, 3, 4, 2} = 1;
//+
Characteristic Length {5, 6} = 0.5;
