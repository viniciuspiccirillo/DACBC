% =========================================================================
%  DACBC.M
%  ACBC with three extensions:
%    E1 — Local Polynomial Predictor  (Chebyshev basis)
%    E2 — Variable Ellipse            (size and orientation from E1)
%    E3 — Invariant Manifolds         (Phi + S via parametrised Poincare map)
%           * mu_max  -> fold detection (Floquet multiplier)
%           * det(S)  -> fold detection (parametric sensitivity)
%           * Ws proj -> exact normal-axis orientation (Tikhonov regularised)
%
%  System:  m*x'' + c*x' + k*x + k3*x^3 = u(t)
%  Control: u = Kp*(x_ref - x) + Kd*(x_ref' - x')   [PD, non-invasive]
%  LMS:     adaptive filter identifies higher harmonics in x_ref
%
%  Dependency: measure_force.m 
% =========================================================================

clear; clc;
close all;

% =========================================================================
%  SECTION 1 — System and controller parameters
% =========================================================================
ctrl.Kp = 1.0;
ctrl.Kd = 1.0;

sys.m  = 1;
sys.c  = 0.05;
sys.k  = 1;
sys.k3 = 1;

F_target = 0.05;
N_harm   = 15;

% =========================================================================
%  SECTION 2 — DACBC core parameters
% =========================================================================

k_alpha_0     = 0.04;
ds_init       = 0.04;
ds_fold       = 0.04;
rho_acbc      = 0.03;
sigma_acbc    = 0.8;
k_alpha_dt    = k_alpha_0;
max_steps     = 500;
max_corr      = 250;
n_min_fold    = 60;
n_max_no_fold = 140;
thr_retro     = 0.01;
jan_excl      = 5;

% =========================================================================
%  SECTION 3 — Extension E1: Local Polynomial Predictor
%
%  Fits a degree-p polynomial omega(s), A(s) to the last n_fit accepted
%  points parametrised by cumulative arc length s.  Exponential weights
%  w_i = exp(-3*(s_max - s_i)/s_max) favour the most recent points.
%
%  From the polynomial the predictor extracts:
%    tau   = [d(omega)/ds, dA/ds] / norm(...)   unit tangent to NFR
%    kappa = |omega'*A'' - A'*omega''| / (omega'^2 + A'^2)^1.5  curvature
%    ds_opt = theta_tol / max(kappa, 1e-3)       curvature-adaptive step
%
%  With fewer than 3 accepted points the predictor degenerates to a
%  secant (kappa = 0, ds = ds_min).
% =========================================================================

poly_deg       = 3;     % polynomial degree (2 = quadratic, 3 = cubic)
n_fit          = 8;     % points used in the fit (>= deg+1)
theta_tol      = 0.1;  % angular tolerance [rad]: ds_opt = theta_tol/kappa
kappa_fold_thr = 6.0;   % kappa threshold for predictive fold warning
ds_min = theta_tol / kappa_fold_thr;

% =========================================================================
%  SECTION 4 — Extension E2: Variable Ellipse
%
%  The continuation ellipse is defined in the LOCAL frame aligned with tau:
%    semi-axis along tau  : Dw_cur = Dw_base * s_fac           (tangential)
%    semi-axis along tau^perp: Da_cur = Dw_cur * razao_normal  (normal)
%
%  where  s_fac = ds / ds_init  scales both axes with the curvature-
%  adapted step.  The ellipse is then rotated by theta_ell = atan2(tau_nd)
%  to align its major axis with the NFR tangent.
% =========================================================================

Dw_base      = 0.04;   % base semi-axis in omega direction [rad/s]
Da_base      = 0.04;   % base semi-axis in A direction
razao_normal = Da_base / Dw_base;   % base semi-axis in A direction

% =========================================================================
%  SECTION 5 — Extension E3: Invariant Manifolds via Poincare map
%
%  Computes two matrices per accepted point:
%
%  Phi (transition matrix) — state perturbations:
%    Phi(:,j) = [x(T; x_ss + eps_j*e_j) - x_ss] / eps_j
%    eps_j = epsilon_pert * max(|x_ss(j)|, 1e-4)   [FIX-2: per-component]
%    => Floquet multipliers mu_1, mu_2 and eigenvectors v_s, v_u
%
%  S (parametric sensitivity matrix) — parameter perturbations:
%    S(:,1) = [x(T; x_ss, omega+domega) - x_ss] / domega
%    S(:,2) = [x(T; x_ss, A*+dA)        - x_ss] / dA
%    => exact Ws projection: (domega,dA*)_Ws = (1-mu_min)*S^+(lam)*v_s
%    => fold indicator: det(S) -> 0 at fold
%
%  Regularisation: Tikhonov S^+(lambda), lambda = lambda_tikhonov * sigma_1
% =========================================================================

epsilon_pert    = 0.007;
epsilon_param   = 0.003;
mu_fold_thr     = 0.92;
mu_fold_crit    = 0.98;
det_S_thr       = 0.05;
lambda_tikhonov = 0.01;
n_warmup_var    = 8;

% =========================================================================
%  SECTION 6 — Initial point: scan in A at omega_start + linear interp.
% =========================================================================

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
    result = penalidade;
    return;
end

A_lo = A_scan(cross_idx);   A_hi = A_scan(cross_idx + 1);
f_lo = f_scan(cross_idx);   f_hi = f_scan(cross_idx + 1);
A_n  = A_lo + (F_target - f_lo) * (A_hi - A_lo) / (f_hi - f_lo);

[f_check, x1, x2] = measure_force(omega_n, A_n, sys, ctrl, x1, x2, N_harm);
if abs(f_check - F_target) / F_target > 0.15
    result = penalidade;
    return;
end

% =========================================================================
%  SECTION 7 — Initialization
% =========================================================================

omega_hist  = omega_n;
A_hist      = A_n;
alpha_n     = 0.0;
n_pts       = 0;
fold_ok     = false;
A_peak      = A_n;
retro_cnt   = 0;
ds_init_cur = ds_init;
ds          = ds_init;

% LMS state carried across E3 calls
w_lms_ext = zeros(2*N_harm, 1);

% Diagnostic histories
X_hist     = [];
kappa_hist = [];
ds_hist    = [];
iters_hist = [];
err_hist   = [];
theta_hist = [];
Dw_hist    = [];
Da_hist    = [];
mu_hist    = [];
detS_hist  = [];
condS_hist = [];
iters      = [];
align_hist = []; 

fprintf('%5s  %8s  %8s  %10s  %5s  %6s  %8s\n', ...
        'n','omega','A','err_f','iters','ds','kappa');

% =========================================================================
%  SECTION 8 — Main DACBC loop
% =========================================================================
n = 1;
while n <= max_steps

    % ── E1: polynomial predictor (Chebyshev basis) ───────────────────────

    pts_now = [omega_hist(:), A_hist(:)];
    [tau_pred, kappa, ds_poly] = local_poly_predictor( ...
        pts_now, poly_deg, n_fit, theta_tol, ds_min);

    % ── E2: variable ellipse — step and base semi-axes ───────────────────

    ds     = min(max(ds_poly, ds_min), ds_init_cur);
    s_fac  = ds / max(ds_init_cur, ds_min);
    Dw_cur = Dw_base * s_fac;

    % ── E3: invariant manifolds via parametrised Poincare map ─────────────
    
    mu_max     = 0;
    mu_min     = 0;
    det_S      = 1.0;
    cond_S_eff = 1.0;
    ws_param   = [0; 1];   % fallback: Ws points in A direction

    if n_pts >= n_warmup_var
        % pass w_lms_ext so warm-up starts from last known weights
        
        [mu_min, mu_max, v_stable_ss, S_mat, w_lms_ext] = floquet_phi_and_S( ...
            omega_n, A_n, sys, ctrl, x1, x2, N_harm, ...
            epsilon_pert, epsilon_param, w_lms_ext);

        det_S = det(S_mat);

        % ── Tikhonov-regularised pseudo-inverse of S ──────────────────
        % S^+(lambda) = V * diag(sigma_i / (sigma_i^2 + lambda^2)) * U'
        % lambda = lambda_tikhonov * sigma_1  (scale-independent)
        % Behaviour:
        %   cond(S) small  : S^+(lam) ~ S^{-1}  (exact inversion)
        %   sigma_2 -> 0   : solution rotates smoothly toward tau
        
        [U_s, Sig_s, V_s] = svd(S_mat);
        sv     = diag(Sig_s);
        lam    = lambda_tikhonov * sv(1);
        filt   = sv ./ (sv.^2 + lam^2);
        S_pinv = V_s * diag(filt) * U_s';
       

        cond_S_eff = sv(1) / max(sv(2), 1e-14);

        % ── Exact Ws projection ────────────────────────────────────────
        % Equation: S*(domega,dA*)_Ws = (1 - mu_min)*v_s
        
        rhs    = (1 - mu_min) * v_stable_ss;
        ws_raw = S_pinv * rhs;
        nm_ws  = norm(ws_raw);
        if nm_ws > 1e-10
            ws_param = ws_raw / nm_ws;
        end

        % ── Fold indicators ───────────────────────────────────────────

        if mu_max > mu_fold_crit && ~fold_ok
            ds_init_cur = min(ds_init_cur, ds_fold);
        end
    end

    % ── E2: ellipse orientation (tau from E1, dimensionless frame) ───────
    
    tau_nd = [tau_pred(1)/Dw_base, tau_pred(2)/Da_base];
    nm_nd  = norm(tau_nd);
    if nm_nd > 1e-10
        tau_nd = tau_nd / nm_nd;
    else
        tau_nd = [1, 0];
    end
    theta_ell = atan2(tau_nd(2), tau_nd(1));
    % tau_perp in the dimensionless frame (used for dot product with ws_nd)
    tau_perp_nd = [-tau_nd(2); tau_nd(1)];

    % ── E3 -> E2: modulate normal semi-axis with Ws alignment ────────────
    % ws_param is in the physical (omega, A*) frame; convert to the
    % dimensionless ellipse frame before computing the dot product, so
    % both vectors live in the same metric space.

    ws_nd = [ws_param(1)/Dw_base; ws_param(2)/Da_base];
    nm_wn = norm(ws_nd);
    if nm_wn > 1e-10
        ws_nd = ws_nd / nm_wn;
    else
        ws_nd = tau_perp_nd;
    end
    
    % Ensure ws_nd is in the same half-space as the geometric normal
    
    if dot(ws_nd, tau_perp_nd) < 0
        ws_nd = -ws_nd;
    end

    % Effective ratio: |cos(angle between Ws and geometric normal)|
    % = 1 when perfectly aligned; -> 0 when Ws aligns with tangent (fold)
    
    razao_ef = razao_normal * abs(dot(ws_nd, tau_perp_nd));
    razao_ef = max(razao_ef, 0.1);   % floor to avoid degenerate ellipse
    Da_cur   = Dw_cur * razao_ef;

    % ========== Alignment ========================
  
    align_ws = abs(dot(ws_nd, tau_perp_nd));
    
    % =============================================

    % ── Correction loop: integral law  alpha_{k+1} = alpha_k - ka*(f-f*) ─
    alpha_k   = 0.0;
    converged = false;
    num_iters = max_corr;

    R_theta = [cos(theta_ell), -sin(theta_ell);
               sin(theta_ell),  cos(theta_ell)];

    for i = 1:max_corr
        p_loc   = [Dw_cur * cos(alpha_k); Da_cur * sin(alpha_k)];
        p_can   = R_theta * p_loc;
        omega_k = omega_n + p_can(1);
        A_k     = A_n     + p_can(2);

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

    k_alpha_dt = k_alpha_0;

    % ── Final verification and point acceptance ───────────────────────────
  
    if converged
        p_loc_f   = [Dw_cur * cos(alpha_k); Da_cur * sin(alpha_k)];
        p_can_f   = R_theta * p_loc_f;
        omega_new = omega_n + p_can_f(1);
        A_new     = A_n     + p_can_f(2);

        [f_fin, x1_k, x2_k, X_real] = measure_force( ...
            omega_new, A_new, sys, ctrl, x1, x2, N_harm);

        if abs(f_fin - F_target) / F_target <= rho_acbc

            omega_n = omega_new;
            A_n     = A_new;
            x1      = x1_k;
            x2      = x2_k;

            omega_hist(end+1) = omega_n;
            A_hist(end+1)     = A_n;
            n_pts             = n_pts + 1;
            X_hist(end+1)     = X_real;
            iters(end+1)      = num_iters;
            align_hist(end+1) = align_ws;

            if A_n > A_peak; A_peak = A_n; end
            alpha_n = alpha_k;

            kappa_hist(end+1) = kappa;
            ds_hist(end+1)    = ds;
            iters_hist(end+1) = num_iters;
            err_hist(end+1)   = abs(f_fin - F_target) / F_target;
            theta_hist(end+1) = theta_ell;
            Dw_hist(end+1)    = Dw_cur;
            Da_hist(end+1)    = Da_cur;
            mu_hist(end+1)    = mu_max;
            detS_hist(end+1)  = det_S;
            condS_hist(end+1) = cond_S_eff;

            fprintf('%5d  %8.4f  %8.4f  %10.6f  %5d  %6.4f  %8.4f\n', ...
                    n, omega_n, A_n, f_fin-F_target, num_iters, ds, kappa);

            % Retrocession check
            if n_pts > jan_excl + 3
                omega_old = omega_hist(1:end-jan_excl);
                A_old     = A_hist(1:end-jan_excl);
                dist_min  = min(sqrt((omega_old - omega_n).^2 + (A_old - A_n).^2));
                if dist_min < thr_retro
                    retro_cnt = retro_cnt + 1;
                    fprintf('  Retrocession (dist=%.4f)\n', dist_min);
                    if retro_cnt >= 2 && ~fold_ok
                        fprintf('  ABORTED: persistent retrocession.\n');
                        break;
                    end
                    if retro_cnt == 1
                        alpha_n = alpha_n + 2*pi/3;
                    end
                else
                    retro_cnt = 0;
                end
            end

            % Geometric fold detection (amplitude history)
            if n_pts >= n_min_fold && length(A_hist) >= 3
                dA1 = A_hist(end)   - A_hist(end-1);
                dA2 = A_hist(end-1) - A_hist(end-2);
                if dA1 < 0 && dA2 < 0 && A_n < 0.99*A_peak
                    if ~fold_ok
                        fprintf('  Fold at omega=%.4f, A=%.4f\n', omega_n, A_n);
                        fold_ok     = true;
                        ds_init_cur = ds_fold;
                    end
                end
            end

            % Safety floor for ds
            if ds < ds_min
                ds = ds_min;
            end

            if ~fold_ok && n_pts >= n_max_no_fold
                fprintf('  No fold after %d points.\n', n_pts);
                break;
            end

            n = n + 1;
        else
            converged = false;
        end
    end

    if ~converged

        ds_init_cur = min(ds_init_cur, ds);
        k_alpha_dt = 8.0; 
   
    end

    if omega_n > 3.0 || omega_n < 0.1 || A_n < 1e-5
        break;
    end
end

fprintf('\nDone: %d accepted points.  Fold: %s\n', n_pts, string(fold_ok));

% =========================================================================
%  SECTION 9 — Visualisation
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


% =========================================================================
%  SECTION 10 — Phi + S via parametrised Poincare map  (Extension E3)
% =========================================================================
function [mu_min, mu_max, v_stable, S_mat, w_lms_out] = floquet_phi_and_S( ...
    omega, A_n, sys, ctrl, x1_orb, x2_orb, N_harm, ...
    epsilon_pert, epsilon_param, w_lms_in)
% Estimates the monodromy matrix Phi and parametric sensitivity matrix S
% of the stroboscopic Poincare map P(x0; omega, A*) = x(T; x0, omega, A*).
%
% Phase 1 — Warm-up (n_preheat periods, LMS active):
%   Starts from w_lms_in (last converged weights) instead of zeros.
%   Produces steady-state x_ss and frozen weights w_ss.
%
% Phase 2 — State perturbations (2 integrations, LMS frozen at w_ss):
%   Phi(:,j) = [P(x_ss + eps_j*e_j) - x_ss] / eps_j
%   eps_j scaled per-component: eps_j = epsilon_pert*max(|x_ss(j)|,1e-4)
%
% Phase 3 — Eigendecomposition of Phi:
%   mu_max   = max(|mu_j|)          -> Floquet fold indicator
%   idx_min  = argmin(|mu_j|)       -> index of stable multiplier
%   mu_min   = real(mu_all(idx_min)) [FIX-1]: same index as v_stable
%   v_stable = real(V(:, idx_min))  -> stable eigenvector
%
% Phase 4 — Parameter perturbations (2 integrations, LMS frozen):
%   S(:,1) = [P(x_ss, omega+domega) - x_ss] / domega
%   S(:,2) = [P(x_ss, A*+dA)        - x_ss] / dA
%   Safety clamp: |S_ij| > phi_max -> S = eye(2).
%
% Returns w_lms_out = w_ss for use as w_lms_in in the next call.

    phi_max   = 20.0;
    n_pts_per = 150;

    T_per  = 2*pi / omega;
    ts_phi = T_per / n_pts_per;
    N1     = n_pts_per;


    k_ord_p  = ceil((1:2*N_harm) / 2);
    is_sin_p = mod((1:2*N_harm), 2) == 1;

    [t_off, x_st, w_lms] = poincare(omega, A_n, sys, ctrl, x1_orb, x2_orb, N_harm, n_pts_per);

    
    x_ss      = x_st;
    w_ss      = w_lms;
    w_lms_out = w_ss;   % return for next call

    % ── Phase 2: state perturbations -> Phi ──────────────────────────────

    Phi = zeros(2, 2);

    for col = 1:2
        eps_col = epsilon_pert * max(abs(x_ss(col)), 1e-4);
        delta = zeros(2, 1);
        delta(col) = eps_col;

        % +eps
        x_plus = x_ss + delta;
        for i = 1:N1
            ti = t_off + (i-1)*ts_phi;
            [x_plus, ~] = rk4_step(x_plus, w_ss, ti, ts_phi, ...
                omega, A_n, ctrl, sys, k_ord_p, is_sin_p, 0, N_harm);
        end

        % -eps
        x_minus = x_ss - delta;
        for i = 1:N1
            ti = t_off + (i-1)*ts_phi;
            [x_minus, ~] = rk4_step(x_minus, w_ss, ti, ts_phi, ...
                omega, A_n, ctrl, sys, k_ord_p, is_sin_p, 0, N_harm);
        end

        Phi(:, col) = (x_plus - x_minus) / (2 * eps_col);
    end

    % Safety check
    if any(~isfinite(Phi(:))) || max(abs(Phi(:))) > phi_max
        mu_max   = 0;
        mu_min   = 0;
        v_stable = [0; 1];
        S_mat    = eye(2);
        return;
    end

    % % ── Phase 3: eigendecomposition ───────────────────────────────────────
    [V_phi, D_phi] = eig(Phi);
    mu_all = diag(D_phi);

    [mu_max,  ~      ] = max(abs(mu_all));
    [~,       idx_min] = min(abs(mu_all));

    mu_min   = real(mu_all(idx_min));
    v_stable = real(V_phi(:, idx_min));
    nm_vs    = norm(v_stable);
    if nm_vs > 1e-10
        v_stable = v_stable / nm_vs;
    else
        v_stable = [0; 1];
    end


    % ── Phase 4: parameter perturbations -> S (CENTRAL DIFFERENCES) ───────

    S_mat = zeros(2, 2);
    d_omega = epsilon_param * omega;
    d_A     = epsilon_param * max(abs(A_n), 1e-4);

    % Column 1: ∂P/∂ω central
    x_plus = x_ss;
    for i = 1:N1
        ti = t_off + (i-1)*ts_phi;
        [x_plus, ~] = rk4_step(x_plus, w_ss, ti, ts_phi, ...
            omega + d_omega, A_n, ctrl, sys, k_ord_p, is_sin_p, 0, N_harm);
    end
    x_minus = x_ss;
    for i = 1:N1
        ti = t_off + (i-1)*ts_phi;
        [x_minus, ~] = rk4_step(x_minus, w_ss, ti, ts_phi, ...
            omega - d_omega, A_n, ctrl, sys, k_ord_p, is_sin_p, 0, N_harm);
    end
    S_mat(:, 1) = (x_plus - x_minus) / (2 * d_omega);

    % Column 2: ∂P/∂A* central
    x_plus = x_ss;
    for i = 1:N1
        ti = t_off + (i-1)*ts_phi;
        [x_plus, ~] = rk4_step(x_plus, w_ss, ti, ts_phi, ...
            omega, A_n + d_A, ctrl, sys, k_ord_p, is_sin_p, 0, N_harm);
    end
    x_minus = x_ss;
    for i = 1:N1
        ti = t_off + (i-1)*ts_phi;
        [x_minus, ~] = rk4_step(x_minus, w_ss, ti, ts_phi, ...
            omega, A_n - d_A, ctrl, sys, k_ord_p, is_sin_p, 0, N_harm);
    end
    S_mat(:, 2) = (x_plus - x_minus) / (2 * d_A);

    if any(~isfinite(S_mat(:))) || max(abs(S_mat(:))) > phi_max
        S_mat = eye(2);
    end
end

% =========================================================================
%  SECTION 11 — RK4 step (u recomputed at every stage)
% =========================================================================

function [x_new, w_new] = rk4_step(x_st, w_lms, ti, ts, ...
    omega, A_n, ctrl, sys, k_ord_p, is_sin_p, mu_lms_p, N_harm)

% Fourth-order Runge-Kutta integrator for the closed-loop PD system.
% The control input u is recomputed at every RK4 stage (t, t+h/2, t+h)
% so that the time-varying reference is correctly captured.
% LMS update uses u and h evaluated at the START of the step (Widrow-Hoff).

    n_coef = 2*N_harm;

    function [u_val, h_val] = compute_u_and_h(x_current, t_local)
        phases = k_ord_p * (omega * t_local);
        h_val  = zeros(n_coef, 1);
        h_val( is_sin_p) = sin(phases( is_sin_p));
        h_val(~is_sin_p) = cos(phases(~is_sin_p));

        x_ref = A_n * sin(omega * t_local);
        v_ref = A_n * omega * cos(omega * t_local);
        for k = 2:N_harm
            js = 2*k-1; jc = 2*k; kw = k*omega;
            x_ref = x_ref + w_lms(js)*sin(kw*t_local) + w_lms(jc)*cos(kw*t_local);
            v_ref = v_ref + w_lms(js)*kw*cos(kw*t_local) - w_lms(jc)*kw*sin(kw*t_local);
        end

        u_val = ctrl.Kp*(x_ref - x_current(1)) + ctrl.Kd*(v_ref - x_current(2));
        u_val = max(-1e6, min(1e6, u_val));
    end

    f_dyn = @(xv, uu) [xv(2); ...
        (uu - sys.c*xv(2) - sys.k*xv(1) - sys.k3*xv(1)^3) / sys.m];

    [u0, h0] = compute_u_and_h(x_st,                   ti);
    k1 = f_dyn(x_st,                   u0);

    [u1, ~ ] = compute_u_and_h(x_st + (ts/2)*k1,       ti + ts/2);
    k2 = f_dyn(x_st + (ts/2)*k1,       u1);

    [u2, ~ ] = compute_u_and_h(x_st + (ts/2)*k2,       ti + ts/2);
    k3 = f_dyn(x_st + (ts/2)*k2,       u2);

    [u3, ~ ] = compute_u_and_h(x_st + ts*k3,            ti + ts);
    k4 = f_dyn(x_st + ts*k3,            u3);

    x_new = x_st + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);

    % LMS update at start-of-step point (standard Widrow-Hoff)
    if mu_lms_p > 0
        e_lms = u0 - h0' * w_lms;
        w_new = w_lms + mu_lms_p * ts * e_lms * h0;
    else
        w_new = w_lms;
    end
end

% =========================================================================
%  SECTION 12 — Local polynomial predictor  (Extension E1, Chebyshev basis)
% =========================================================================

function [tau, kappa, ds_opt] = local_poly_predictor( ...
    pts, deg, n_fit, theta_tol, ds_min)

% Fits degree-p Chebyshev polynomials omega(t), A(t) to the last n_fit
% accepted points, evaluates first and second derivatives at t = +1
% (current point) via Clenshaw recurrence, and returns:
%   tau    — unit tangent vector
%   kappa  — Frenet curvature
%   ds_opt — curvature-adaptive step size

    N = size(pts, 1);

    if N < 3
        if N == 2
            d   = pts(end,:) - pts(end-1,:);
            tau = d / max(norm(d), 1e-10);
        else
            tau = [1, 0];
        end
        kappa  = 0;
        ds_opt = ds_min;
        return;
    end

    idx = max(1, N - n_fit + 1) : N;
    P   = pts(idx, :);
    M   = size(P, 1);

    ds_seg = sqrt(sum(diff(P).^2, 2));
    s      = [0; cumsum(ds_seg)];
    s_max  = s(end);

    if s_max < 1e-10
        tau    = [1, 0];
        kappa  = 0;
        ds_opt = ds_min;
        return;
    end

    % Map arc length to Chebyshev interval; current point -> t = +1
    t = 2*s / s_max - 1;

    % Exponential weights: w(last) = 1, w(first) ~ 0.05
    w = exp(-3*(s_max - s) / s_max);

    deg_eff = min(deg, M-1);

    % Chebyshev Vandermonde matrix  V(i, j+1) = T_j(t_i)
    V_cheb      = zeros(M, deg_eff+1);
    V_cheb(:,1) = ones(M, 1);
    if deg_eff >= 1; V_cheb(:,2) = t; end
    for j = 2:deg_eff
        V_cheb(:,j+1) = 2*t .* V_cheb(:,j) - V_cheb(:,j-1);
    end

    % Weighted normal equations; cond(A_lhs) = O(1)
    W_mat = diag(w);
    A_lhs = V_cheb' * W_mat * V_cheb;
    c_om  = A_lhs \ (V_cheb' * W_mat * P(:,1));
    c_A   = A_lhs \ (V_cheb' * W_mat * P(:,2));

    % Derivatives at t = +1; chain rule: d/ds = (2/s_max) * d/dt
    [dom_t, d2om_t] = cheb_eval_d12(c_om, 1.0, deg_eff);
    [dA_t,  d2A_t ] = cheb_eval_d12(c_A,  1.0, deg_eff);
    fac  = 2 / s_max;
    dom  = fac   * dom_t;
    d2om = fac^2 * d2om_t;
    dA   = fac   * dA_t;
    d2A  = fac^2 * d2A_t;

    % Unit tangent
    tau_raw = [dom, dA];
    nm      = norm(tau_raw);
    if nm > 1e-10
        tau = tau_raw / nm;
    else
        d   = P(end,:) - P(end-1,:);
        tau = d / max(norm(d), 1e-10);
    end

    % Curvature kappa = |omega' A'' - A' omega''| / (omega'^2+A'^2)^(3/2)
    kappa  = abs(dom*d2A - dA*d2om) / max((dom^2 + dA^2)^1.5, 1e-10);
    ds_opt = theta_tol / max(kappa, 1e-3);
    ds_opt = max(ds_opt, ds_min);
end

% =========================================================================
%  SECTION 13 — Chebyshev first and second derivatives at a point
% =========================================================================

function [df, d2f] = cheb_eval_d12(c, t, deg)

% Evaluates the first and second derivatives of
%   f(t) = sum_{j=0}^{n} c(j+1) * T_j(t)
% at the given point t, using the backward recurrence for derivative
% coefficients followed by Clenshaw evaluation.
%
% Recurrence for first-derivative coefficients d_j:
%   d_j = 2*(j+1)*c_{j+1} + d_{j+2},  j = n-1, ..., 1
%   d_0 = c_1 + 0.5*d_2               <- special: factor 1/2 from T_0 norm
%
% The factor 0.5 in d_0 is NOT optional: omitting it doubles d_0, causing
% O(1) errors in omega' and A', and consequently in tau and kappa.
% The same recurrence (with d replacing c) gives the second derivative.

    n = length(c) - 1;

    % ── First derivative coefficients ────────────────────────────────────
    d = zeros(n+2, 1);
    for j = n-1:-1:1
        d(j+1) = 2*(j+1)*c(j+2) + d(j+3);
    end
    d(1) = c(2) + 0.5*d(3);        % [FIX-6]: factor 0.5 required

    df = clenshaw_eval(d(1:max(n,1)), t);

    % ── Second derivative coefficients ───────────────────────────────────
    m = max(n-1, 0);
    e = zeros(m+2, 1);
    for j = m-1:-1:1
        e(j+1) = 2*(j+1)*d(j+2) + e(j+3);
    end
    if m >= 1
        e(1) = d(2) + 0.5*e(3);    % [FIX-6]: same factor 0.5
    end

    d2f = clenshaw_eval(e(1:max(m,1)), t);
end

% =========================================================================
%  SECTION 14 — Clenshaw evaluation of a Chebyshev series
% =========================================================================

function y = clenshaw_eval(c, t)

% Evaluates f(t) = sum_{j=0}^{n} c(j+1)*T_j(t) via backward Clenshaw
% recurrence.  O(n), numerically stable, avoids explicit T_j(t).
%
% Recurrence:
%   b_{n+1} = 0,  b_n = c_n
%   b_j = c_j + 2t*b_{j+1} - b_{j+2},   j = n-1, ..., 1
%   f(t) = c_0 + t*b_1 - b_2
% The terminal formula f = c_0 + t*b_1 - b_2 (NOT 2t*b_1 - b_2 + c_0)
% is the standard Clenshaw termination and is correct as written.

    n = length(c) - 1;
    if n < 0; y = 0;    return; end
    if n == 0; y = c(1); return; end

    b2 = 0;
    b1 = c(n+1);
    for j = n-1:-1:1
        b0 = c(j+1) + 2*t*b1 - b2;
        b2 = b1;
        b1 = b0;
    end
    y = c(1) + t*b1 - b2;
end


function [t_end, x_st, w_lms] = poincare(omega, A_target, sys, ctrl, xi, yi, N_harm, n)


    T  = 2*pi / omega;
    % n = 5;
    ts = T / n;          
    t_end = n * T;       
    N  = round(t_end / ts);
    mu_lms = 0.5 / (2*pi*sqrt(N_harm));

    % ================== SIMULAÇÃO ==================
    n_coef = 2 * N_harm;
    w_lms  = zeros(n_coef, 1);
    x_st   = [xi; yi];

    
    f_dyn = @(xv, uu) [xv(2); (uu - sys.c*xv(2) - sys.k*xv(1) - sys.k3*xv(1)^3 )/sys.m];

    x_history = zeros(N,1);   
    v_history = zeros(N,1);   


    for i = 1:N
        ti = (i-1)*ts;
        phases = ceil((1:n_coef)/2) * (omega*ti);
        h = zeros(n_coef,1);
        h(1:2:end) = sin(phases(1:2:end));
        h(2:2:end) = cos(phases(2:2:end));

        % Referência
        x_ref = A_target * sin(omega*ti);
        v_ref = A_target * omega * cos(omega*ti);
        for k = 2:N_harm
            j_s = 2*k-1; j_c = 2*k;
            kw = k*omega;
            x_ref = x_ref + w_lms(j_s)*sin(kw*ti) + w_lms(j_c)*cos(kw*ti);
            v_ref = v_ref + w_lms(j_s)*kw*cos(kw*ti) - w_lms(j_c)*kw*sin(kw*ti);
        end

        u = ctrl.Kp*(x_ref - x_st(1)) + ctrl.Kd*(v_ref - x_st(2));

        % LMS
        e_lms = u - h'*w_lms;
        w_lms = w_lms + mu_lms*ts*e_lms*h;

        % Integration
        k1 = f_dyn(x_st, u);
        k2 = f_dyn(x_st + ts/2*k1, u);
        k3 = f_dyn(x_st + ts/2*k2, u);
        k4 = f_dyn(x_st + ts*k3, u);
        x_st = x_st + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);

        x_history(i) = x_st(1);
        v_history(i) = x_st(2); 

        
    end


        Ns = round(length(x_history)*0.7);
        Np = length(x_history);
        

        X = x_history(Ns:n:Np);
        Y = v_history(Ns:n:Np);

        x_st = [X(end); Y(end)];

end 
