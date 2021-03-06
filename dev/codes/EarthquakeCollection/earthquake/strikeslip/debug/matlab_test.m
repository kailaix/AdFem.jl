N = 22;
n = 21;
sigma_p = 50*linspace(1,2,n+1);
a = 0.01*linspace(1,2,n+1);
V_0 = 1e-6;
tau_0 = 0;
tau_qs = 30*linspace(1,2,n+1);
eta = 4.7;
f_0 = 0.6;
psi = linspace(1,2,n+1) * 1.01 * f_0;
b = 0.02*linspace(1,2,n+1);
L = 0.01;
V = ComputeSlip(N, sigma_p, a, V_0, psi, tau_0, tau_qs, eta);
% StateRate(a, b, V, V_0, psi, L, f_0)
