%% Calculates the rate of the state change at time t.
%
%   Input:
%   Vector a      	- a-parameter in the friction law
%   Vector b      	- b-parameter in the friction law
%   Vector V       	- Slip velocity
%   Scalar V_0      - Steady sliding slip velocity
%   Vector psi   	- State variable psi
%   Scalar L        - State evolution distance
%   Scalar f_0      - Reference friction coefficient at slip velocity V_0     
%
%   Output:
%   Scalar dPsi    	- The rate of change of the state variable
%
% Authors: Kim Torberntsson, Vidar Stiernstrom

function [dPsi] = StateRate(a, b, V, V_0, psi, L, f_0)

% Calculate the friction
f = a.*asinh((V/(2*V_0)).*exp(psi./a));

% Calculate the rate of change in state psi
dPsi = -V/L.*(f - f_0 + (b - a).*log(V/V_0));

end