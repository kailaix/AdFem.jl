%% Rootfinding using the bisection method
%
%   Input:
%   Function f    	- The function handle to the function that the root
%                     should be found in
%   Scalar xL       - The left boundary
%   Scalar xR     	- The right boundary
%   Scalar maxIter  - Maximum iterations (default 1e5)
%   xTol            - Error tolerance for root
%   fTol            - Error tolerance for f(x)
%
%   Output:
%   Scalar xM    	- The value of the root
%
% Authors: Kim Torberntsson, Vidar Stiernstrom

function [ xM ] = Bisection(f, xL, xR, maxIter, xTol, fTol)

% Find the values of the function at the left and right endpoint.
fL = f(xL);
fR = f(xR);

% Check if the boundaries fulfil the error tolerance
if abs(fL) < fTol
    xM = xL;
    return
elseif abs(fR) < fTol
    xM = xR;
    return;
end

% Check that the root is bracketed
if fL*fR >= 0
    error(['Invalid bracket, f(', num2str(xL), ') = ', num2str(fL), ...
        ' must have opposite sign of f(', num2str(xR), ') = ', num2str(fL)]);
end

% Always split the interval at least one time
xM = (xL + xR)/2;
fM = f(xM);
xErr = abs(xM - xL);
fErr = abs(fM);
numIter = 1;

% Split the interval until the error tolerances are fulfilled
% (or until maximum iterations are reached)
while numIter <= maxIter && (xErr >= xTol || fErr >= fTol)
    if (fL*fM <= 0)
        xR = xM;
    else
        xL = xM;
        fL = fM;
    end
    % Calculate values of the new mid-point
    xM = (xL + xR)/2;
    fM = f(xM);
    
    xErr = abs((xM - xL));
    fErr = abs(fM);
    numIter = numIter + 1;
end

% Check if convergence was achieved. If not throw an error
if xErr > xTol || fErr > fTol
    xErr
    fErr
    error('No convergence');
end

end
