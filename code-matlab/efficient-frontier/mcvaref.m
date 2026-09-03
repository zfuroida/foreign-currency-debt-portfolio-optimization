clc; clear; close all;

%% ===== DATA =====
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

assetNames = {'USD_5Y','USD_10Y','USD_30Y','EUR_7Y','EUR_10Y','JPY_5Y','JPY_10Y'};
n = length(mu);

%% ===== GENERATE SCENARIOS =====
nScenarios = 10000;
rng(42);
Scenarios = mvnrnd(mu', Sigma, nScenarios);

%% ===== PARAMETER CVaR =====
alpha = 0.95;   % confidence level

%% ===== BATAS =====
% lb = zeros(n, 1);
% ub = ones(n, 1);
lb = [0.410 0.085 0.115 0.115 0.000 0.035 0.015]';
ub = [0.505 0.140 0.180 0.170 0.045 0.100 0.070]';

%% ===== TITIK AWAL =====
capacity  = ub - lb;
remaining = 1 - sum(lb);
x0 = lb + capacity * (remaining / sum(capacity));
x0 = x0 / sum(x0);

%% ===== OPTIONS =====
options = optimoptions('fmincon','Display','none','Algorithm','sqp', ...
    'OptimalityTolerance', 1e-10, ...
    'ConstraintTolerance', 1e-10, ...
    'MaxIterations',       1000);
Aeq = ones(1, n);
beq = 1;

cost_fun  = @(x) mu' * x;
cvar_fun  = @(w) cvar_calc(w, Scenarios, alpha);                            

%% ===== TENTUKAN RANGE CVaR OTOMATIS =====
% Titik A: minimum CVaR (risiko paling kecil, cost tinggi)
[wA, ~, efA] = fmincon(cvar_fun, x0, [], [], Aeq, beq, lb, ub, [], options);
% Titik B: maksimasi return = minimasi -mu'*x (CVaR tertinggi di frontier)
[wB, ~, efB] = fmincon(cost_fun, x0, [], [], Aeq, beq, lb, ub, [], options);

if efA <= 0 || efB <= 0
    error('Titik ekstrem tidak feasible. Periksa lb/ub.');
end

Rmax_min = cvar_fun(wA);   
Rmax_max = cvar_fun(wB);   

fprintf('=== Range M-CVaR Frontier ===\n');
fprintf('Min Risiko : %.6e\n', Rmax_min);
fprintf('Min Cost   : %.6e\n', Rmax_max);
fprintf('\n');

%% ===== LOOP EFFICIENT FRONTIER =====
nFront          = 1000;
TargetRisiko    = linspace(Rmax_min, Rmax_max, nFront);
risiko_frontier = zeros(nFront, 1);
biaya_frontier  = zeros(nFront, 1);
bobot_frontier  = zeros(n, nFront);
exitflags       = zeros(nFront, 1);

w_prev = wA;   % mulai dari titik min CVaR
for i = 1:nFront
    Rmax    = TargetRisiko(i);
    nonlcon = @(x) deal(cvar_fun(x) - Rmax, []);
    [x_opt, ~, ef] = fmincon(cost_fun, w_prev, [], [], Aeq, beq, lb, ub, nonlcon, options);
    exitflags(i) = ef;
    if ef > 0
        bobot_frontier(:, i) = x_opt;
        biaya_frontier(i)    = cost_fun(x_opt);   
        risiko_frontier(i)   = cvar_fun(x_opt);
        w_prev = x_opt;
    else
        bobot_frontier(:, i) = NaN(n,1);
        biaya_frontier(i)    = NaN;
        risiko_frontier(i)   = NaN;
    end
end

valid = exitflags > 0;
fprintf('Titik valid: %d / %d\n\n', sum(valid), nFront);

%% ===== GRAFIK EFFICIENT FRONTIER =====

figure('Name','Efficient Frontier Mean-CVaR','NumberTitle','off');
plot(risiko_frontier(valid), biaya_frontier(valid), 'b-', 'LineWidth', 2);
hold on;

rsk_v = risiko_frontier(valid);
cost_v = biaya_frontier(valid);

% Titik Min CVaR (kiri atas — CVaR kecil, return rendah)
[~, idx_minCVaR] = min(cost_v);
plot(rsk_v(idx_minCVaR), cost_v(idx_minCVaR), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

xlabel('CVaR (Loss, 95%)');
ylabel('Expected Cost');
title('Efficient Frontier — Mean-CVaR');
legend('Frontier', 'Min Cost', 'Location', 'best');
grid on;

%% ===== TABEL KOMPOSISI SEMUA TITIK =====
fprintf('=== Komposisi Bobot Seluruh Titik Frontier ===\n\n');
fprintf('%-6s', 'Titik');
for j = 1:n
    fprintf('  %-13s', assetNames{j});
end
fprintf('  %-14s  %-14s  %-13s\n', 'Variance', 'Std Dev', 'Exp.Return');
fprintf('%s\n', repmat('-', 1, 6 + n*15 + 45));
for i = 1:nFront
    if exitflags(i) > 0
        fprintf('%-6d', i);
        for j = 1:n
            fprintf('  %-13.6f', bobot_frontier(j, i));
        end
        fprintf('  %-14.6e  %-14.6e  %-13.6e\n', ...
            risiko_frontier(i), sqrt(risiko_frontier(i)), biaya_frontier(i));
    end
end

%% ===== TITIK KUNCI (ringkasan) =====
idx_valid  = find(valid);
nv         = length(idx_valid);
idx_key    = idx_valid(round(linspace(1, nv, 5)));
key_labels = {'MinVar','Q25','Q50','Q75','MaxReturn'};

fprintf('\n=== Ringkasan Titik Kunci ===\n');
fprintf('%-12s', 'Metric');
for k = 1:5
    fprintf('  %-13s', key_labels{k});
end
fprintf('\n%s\n', repmat('-', 1, 12 + 5*15));
for j = 1:n
    fprintf('%-12s', assetNames{j});
    for k = 1:5
        fprintf('  %-13.6f', bobot_frontier(j, idx_key(k)));
    end
    fprintf('\n');
end
fprintf('%-12s', 'Variance');
for k = 1:5
    fprintf('  %-13.6e', risiko_frontier(idx_key(k)));
end
fprintf('\n%-12s', 'Std Dev');
for k = 1:5
    fprintf('  %-13.6e', sqrt(risiko_frontier(idx_key(k))));
end
fprintf('\n%-12s', 'Exp.Return');
for k = 1:5
    fprintf('  %-13.6e', biaya_frontier(idx_key(k)));
end
fprintf('\n');

%% ===== FUNGSI CVaR (dari KERUGIAN / LOSS) =====
function cvar = cvar_calc(w, Scenarios, alpha)
    port_return = Scenarios * w;            % return tiap skenario
    port_loss   = -port_return;             % PERBAIKAN: loss = -return
    VaR         = quantile(port_loss, alpha); % VaR dari distribusi loss
    cvar        = mean(port_loss(port_loss >= VaR));  % rata-rata tail loss
end