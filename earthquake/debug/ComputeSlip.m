%% Calculates the residual of the friction law for given parameters
%
%   Input:
%   Scalar N        - Number of grid points at the (1D) fault
%   Vector sigma_p	- Effective stress sigma'
%   Vector a      	- a-parameter in the friction law
%   Scalar V_0      - Steady sliding slip velocity
%   Vector psi   	- State variable psi
%   Scalar tau_0  	- Initial strain
%   Vector tau_qs 	- Quasi-static strain
%   Scalar eta    	- Radiation damping
%   Vector Range    - Defines the range of points that are on the fault
%
%   Output:
%   Scalar V    	- The slip velocity that is solved for using the
%                     bisection method.
%
% Authors: Kim Torberntsson, Vidar Stiernstrom

function [V] = ComputeSlip(N, sigma_p, a, V_0, psi, tau_0, tau_qs, eta, Range)

% Initiate slip velocity vector
V = zeros(N, 1);

% Loop through each point at the fault and find the slip velocity for
% solving the friction law using the bisection method.
for i = 1:N
    % If a fault range is specified and the point is not on the fault, set
    % the slip rate to zero.
    if nargin > 8 && (i < Range(1) || i > Range(2))
        V(i) = 0;
    % Otherwise calculate the slip on the fault
    else
        % Set the left and right bounds for the slip velocity (guesses for
        % solution)
        left = 0;
        right = (tau_0 + tau_qs(i))/eta;
        
        % Swap the bounds if right is negative
        if right < 0
            left = right;
            right = 0;
        end
        
        f = @(V) sigma_p(i)*a(i)*asinh(V/(2*V_0)*exp(psi(i)/a(i))) ...
            - (tau_0 + tau_qs(i)) + eta*V;
        V(i) = Bisection(f, left, right, 1e6, 1e-10, 1e-10);
    end
end

end