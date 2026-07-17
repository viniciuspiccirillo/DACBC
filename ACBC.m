% ================================================================
% ACBC — Derivative-free arclength continuation
%  System:  m*x'' + c*x' + k*x + k3*x^3 = u(t)
%  Control: u = Kp*(x_ref - x) + Kd*(x_ref' - x')   [PD, non-invasive]
%  LMS:     adaptive filter identifies higher harmonics in x_ref
%
%  Dependency: measure_force.m 
%
% Reference: Raze et al., Nonlinear Dyn 2025
% ================================================================

clear; clc;
close all;

% =========================================================================
%  SECTION 1 — System and controller parameters
% =========================================================================

% PD gains (fixed throughout the run)
ctrl.Kd = 1.0;
ctrl.Kp = 1.0;

% Duffing oscillator:  m*x'' + c*x' + k*x + k3*x^3 = u(t)
sys.m  = 1.0;
sys.c  = 0.05;
sys.k  = 1.0;
sys.k3 = 1.0;

% Target force amplitude and number of LMS harmonics
F_target = 0.05;
N_harm   = 15;


% =========================================================================
%  SECTION 2 — ACBC core parameters
% =========================================================================

k_alpha_0     = 0.04;
ds            = 0.04;  
ds1           = 0.04;
ds_init       = 0.04;     
ds_fold       = 0.04;     
ds_min        = 0.003;    
rho_acbc      = 0.03;     
sigma_acbc    = 0.8;      
k_alpha_dt    = k_alpha_0;      
max_steps     = 500;      
max_corr      = 250;       
n_min_fold    = 60;    
n_max_no_fold = 1400;   
thr_retro     = 0.01;
jan_excl      = 5.0;

           
% =====================================================================
% SECTION 3. Initial point: scan in A at omega_start + linear interp.
% =====================================================================

omega_n = 0.1;
x1 = 0;
x2 = 0;

A_scan = linspace(0.01, 2.0, 200);
f_scan = zeros(size(A_scan));
for k = 1:length(A_scan)
    [fv, ~, ~] = measure_force(omega_n, A_scan(k), sys, ctrl, x1, x2, N_harm);
    f_scan(k) = fv;
end

cross_idx = find(diff(sign(f_scan - F_target)) ~= 0, 1, 'first');
if isempty(cross_idx)
    S = penalidade;
    return;
end

A_lo = A_scan(cross_idx);   A_hi = A_scan(cross_idx + 1);
f_lo = f_scan(cross_idx);   f_hi = f_scan(cross_idx + 1);
A_n  = A_lo + (F_target - f_lo) * (A_hi - A_lo) / (f_hi - f_lo);

[f_check, x1, x2] = measure_force(omega_n, A_n, sys, ctrl, x1, x2, N_harm);
if abs(f_check - F_target) / F_target > 0.15
    S = penalidade;
    return;
end

% ================================================================
% SECTION 4. Initialization 
% ================================================================

omega_hist = omega_n;
A_hist     = A_n;
alpha_n    = 0.0; 
n_pts      = 0;
fold_ok    = false;
A_peak     = A_n;
retro_cnt = 0;

% Diagnostic histories
X_hist     = [];
iters      = [];

fprintf('%5s  %8s  %8s  %10s  %5s  %6s\n', ...
        'n', 'omega', 'A', 'err_f', 'iters', 'ds');

% ================================================================
% SECTION 5. Main ACBC loop
% ================================================================

n = 1;
while n <= max_steps

    % ------------------------------------------------------------------
    % Correction loop: integral law  alpha_{k+1} = alpha_k - k*(f-f*)
    % ------------------------------------------------------------------

    alpha_k   = alpha_n;
    converged = false;
    f_k       = NaN;
    num_iters = max_corr;

    for i = 1:max_corr
        omega_k = omega_n + ds * cos(alpha_k);
        A_k     = A_n     + ds1 * sin(alpha_k);

        if omega_k <= 0.05 || A_k <= 0.001
            alpha_k = alpha_k + 0.1;
            continue;
        end

        [f_k, ~, ~] = measure_force(omega_k, A_k, sys, ctrl, x1, x2, N_harm);
        err_k = f_k - F_target;

        if abs(err_k) < sigma_acbc * rho_acbc * F_target
            converged = true;
            num_iters = i;
            break;
        end

        alpha_k = alpha_k - k_alpha_dt * err_k;
    end
    
    k_alpha_dt = k_alpha_0;      % ganho da lei integral: Δα = −k·(f̂−f*)
   
    % ------------------------------------------------------------------
    % Final verification and acceptance
    % ------------------------------------------------------------------
    
    if converged
        omega_new = omega_n + ds * cos(alpha_k);
        A_new     = A_n     + ds1 * sin(alpha_k);

        [f_fin, x1_k, x2_k, X_real] = measure_force(omega_new, A_new, sys, ctrl, x1, x2, N_harm);

        if abs(f_fin - F_target) / F_target <= rho_acbc

            % --- Aceita o ponto ---
            omega_n = omega_new;
            A_n     = A_new;
            x1      = x1_k;
            x2      = x2_k;

            omega_hist(end+1) = omega_n; 
            A_hist(end+1)     = A_n;     
            X_hist(end+1)     = X_real;
            iters(end+1)      = num_iters;
            n_pts             = n_pts + 1;


            if A_n > A_peak
                A_peak = A_n;
            end

            % CORRETION: alpha_n receive the angle of the sweeping 
            alpha_n = alpha_k;

            fprintf('%5d  %8.4f  %8.4f  %10.6f  %5d  %6.4f\n', ...
                    n, omega_n, A_n, f_fin - F_target, num_iters, ds);
      
               % Retrocession check
            if n_pts > jan_excl + 3
                omega_old = omega_hist(1:end-jan_excl);
                A_old     = A_hist(1:end-jan_excl);
                dist_min  = min(sqrt((omega_old-omega_n).^2 + ...
                                     (A_old-A_n).^2));
                if dist_min < thr_retro
                    retro_cnt = retro_cnt + 1;
                    fprintf('  Retrocession (dist=%.4f)\n', dist_min);
                    if retro_cnt >= 2 && ~fold_ok
                        fprintf('  ABORTED: persistent retrocession.\n');
                        break;
                    end
                    if retro_cnt == 1
                        fprintf('  -> Reversing direction (alpha + 120 deg)\n');
                        alpha_n = alpha_n + 2*pi/3;
                    end
                else
                    retro_cnt = 0;
                end
            end

            % Fold detection (geometric criterion on amplitude history)
            if n_pts >= n_min_fold && length(A_hist) >= 3
                dA_last = A_hist(end)   - A_hist(end-1);
                dA_prev = A_hist(end-1) - A_hist(end-2);
                if dA_last < 0 && dA_prev < 0 && A_n < 0.99*A_peak
                    if ~fold_ok
                        fprintf('  Fold at omega=%.4f, A=%.4f\n', omega_n, A_n);
                        fold_ok     = true;
                        ds_init_cur = ds_fold;
                    end
                end
            end

            if ~fold_ok && n_pts >= n_max_no_fold
                fprintf('  No fold after %d points — stopping.\n', n_pts);
                break;
            end

            n  = n + 1;

        else

            converged = false;
        end
    end


    if ~converged

    k_alpha_dt = 8.0; 
    end

    if omega_n > 2.0 || omega_n < 0.1 || A_n < 1e-5; break; end

end

fprintf('\nDone: %d accepted points.  Fold detected: %s\n', n_pts, string(fold_ok));


% =========================================================================
%  SECTION 6 — Visualization
% =========================================================================

openfig('FRC_MATCONT.fig') % Load the NFR curve obtained using the MATCONT package
hold on; box on;
omega_plot = omega_hist(2:end);

plot(omega_plot, X_hist, 'bo-', 'LineWidth',1.5, 'MarkerFaceColor','b','MarkerSize',4);
xlabel('\omega','FontSize', 20, 'FontWeight', 'bold'); 
ylabel('|x|','FontSize', 20, 'FontWeight', 'bold');
ax = gca;  
ax.FontSize = 20; 

% Inset
axInset = axes('Position',[0.55 0.55 0.30 0.30]);
hFig = openfig('FRC_MATCONT.fig','invisible');
hAx = gca;
copyobj(allchild(hAx),axInset)
close(hFig)
hold(axInset,'on')
box on
plot(omega_plot, X_hist, 'bo-', 'LineWidth',1.5, 'MarkerFaceColor','b','MarkerSize',4);


[omega_unique,~,idx] = unique(omega_plot);
iter_sum = accumarray(idx(:),iters(:));

cost_acum = cumsum(iters);

figure(2)
hold on
box on
plot(omega_plot,cost_acum,'-or','LineWidth',1.5)
xlabel('\omega', 'FontSize', 20, 'FontWeight', 'bold')
ylabel('Cumulative correction iterations', 'FontSize', 20, 'FontWeight', 'bold')
ax = gca;  
ax.FontSize = 20; 


figure(3)
edges = min(omega_plot):0.05:max(omega_plot);

cost_bin = zeros(length(edges)-1,1);

for k=1:length(cost_bin)

    ind = omega_plot >= edges(k) & omega_plot < edges(k+1);

    cost_bin(k) = sum(iters(ind));

end

omega_bin = edges(1:end-1)+diff(edges)/2;

bar(omega_bin,cost_bin)

xlabel('\omega', 'FontSize', 20, 'FontWeight', 'bold')
ylabel('Total correction iterations', 'FontSize', 20, 'FontWeight', 'bold')
ax = gca;  
ax.FontSize = 20; 