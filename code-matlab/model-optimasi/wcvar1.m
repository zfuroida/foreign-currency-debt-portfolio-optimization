clear; clc;

% ====== DATA INPUT ======
mu = [0.000023;
0.000022;
0.000017;
0.000067;
0.000068;
0.000334;
0.000337];

Sigma = [0.00000019	0.00000020	0.00000018	0.00000029	0.00000028	0.00000031	0.00000031;
0.00000020	0.00000030	0.00000024	0.00000033	0.00000032	0.00000039	0.00000040;
0.00000018	0.00000024	0.00000025	0.00000033	0.00000032	0.00000038	0.00000040;
0.00000029	0.00000033	0.00000033	0.00002299	0.00002296	0.00001439	0.00001436;
0.00000028	0.00000032	0.00000032	0.00002296	0.00002297	0.00001438	0.00001435;
0.00000031	0.00000039	0.00000038	0.00001439	0.00001438	0.00003703	0.00003687;
0.00000031	0.00000040	0.00000040	0.00001436	0.00001435	0.00003687	0.00003683];
n = length(mu);

% ====== BATASAN BOBOT ======
% lb = [0.425 0.100 0.130 0.130 0.015 0.050 0.030]';
% ub = [0.490 0.125 0.165 0.155 0.030 0.085 0.055]';

% lb = [0.410 0.085 0.115 0.115 0.000 0.035 0.015]';
% ub = [0.505 0.140 0.180 0.170 0.045 0.100 0.070]';

lb = [0.375 0.050 0.080 0.180 0.065 0.100 0.080]';
ub = [0.440 0.075 0.115 0.205 0.080 0.135 0.105]';

% lb = [0.439 0.094 0.128 0.124 0.004 0.047 0.024]';
% ub = [0.479 0.134 0.168 0.164 0.044 0.087 0.064]';

% lb = zeros(n,1);
% ub = ones(n,1);

Aeq = ones(1,n);
beq = 1;

% ====== TITIK AWAL FEASIBLE ======
capacity  = ub - lb;
remaining = 1 - sum(lb);
x0 = lb + capacity * (remaining / sum(capacity));
x0 = x0 / sum(x0);

% ====== PARAMETER WCVaR ======
alpha  = 0.95;                       % tingkat kepercayaan
kappa = sqrt(alpha / (1 - alpha));    % kappa = sqrt(0.95/0.05) ≈ 4.3589

% ====== NILAI Rmax ======
Rmax_list = [0.000007439 0.000025577 0.005494758 0.010354242 0.011764944 0.021980535];

% ====== OPSI SOLVER ======
options = optimoptions('fmincon','Algorithm', 'sqp','Display','off', ...
    'TolFun',    1e-10,'TolX',      1e-10);

% ====== LOOP UNTUK SETIAP Rmax ======
for k = 1:length(Rmax_list)
    Rmax = Rmax_list(k);
    objfun = @(x) mu' * x;
    nonlcon = @(x) deal(mu'*x + kappa * sqrt(x'*Sigma*x) - Rmax, []);
    [x_opt, fval, exitflag] = fmincon(objfun, x0, [], [], Aeq, beq, lb, ub, nonlcon, options);

    % ====== HASIL ======
    expCost = mu' * x_opt;
    WCVaR_val = mu'*x_opt + kappa * sqrt(x_opt'*Sigma*x_opt);
    
    fprintf('\n==== HASIL MEAN-WCVaR (Rmax = %.8f) ====\n', Rmax);
    disp(table((1:n)', x_opt, 'VariableNames', {'Aset','Bobot'}))
    fprintf('Expected Cost  : %.10f\n', expCost);
    fprintf('Risk (WCVaR)            : %.10f\n', WCVaR_val);
    fprintf('Sum Bobot       : %.1f\n',  sum(x_opt));
    fprintf('exitflag = %d', exitflag);
end