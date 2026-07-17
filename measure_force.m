
function [f, x1, x2, X_real, X, Y] = measure_force(omega, A_target, sys, ctrl, xi, yi, N_harm)

    T  = 2*pi / omega;
    n = 150;
    ts = T / n;          
    t_end = n * T;       
    N  = round(t_end / ts);
    mu_lms = 0.5 / (2*pi*sqrt(N_harm));

    n_coef = 2 * N_harm;
    w_lms  = zeros(n_coef, 1);
    x_st   = [xi; yi];

    
    f_dyn = @(xv, uu) [xv(2); (uu - sys.c*xv(2) - sys.k*xv(1) - sys.k3*xv(1)^3 )/sys.m];  % Duffing oscillator

    x_history = zeros(N,1);   
    v_history = zeros(N,1);   

    for i = 1:N
        ti = (i-1)*ts;
        phases = ceil((1:n_coef)/2) * (omega*ti);
        h = zeros(n_coef,1);
        h(1:2:end) = sin(phases(1:2:end));
        h(2:2:end) = cos(phases(2:2:end));

        % Reference
        x_ref = A_target * sin(omega*ti);
        v_ref = A_target * omega * cos(omega*ti);
        for k = 2:N_harm
            j_s = 2*k-1; j_c = 2*k;
            kw = k*omega;
            x_ref = x_ref + w_lms(j_s)*sin(kw*ti) + w_lms(j_c)*cos(kw*ti);
            v_ref = v_ref + w_lms(j_s)*kw*cos(kw*ti) - w_lms(j_c)*kw*sin(kw*ti);
        end

        u = ctrl.Kp*(x_ref - x_st(1)) + ctrl.Kd*(v_ref - x_st(2));

        % LMS of the force 
        e_lms = u - h'*w_lms;
        w_lms = w_lms + mu_lms*ts*e_lms*h;

        % Integration RK-4th
        k1 = f_dyn(x_st, u);
        k2 = f_dyn(x_st + ts/2*k1, u);
        k3 = f_dyn(x_st + ts/2*k2, u);
        k4 = f_dyn(x_st + ts*k3, u);
        x_st = x_st + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);

        x_history(i) = x_st(1);  % displacement
        v_history(i) = x_st(2);  % velocity
    end

    f = norm(w_lms(1:2));   % Harmonic force

    % ================== REAL AMPLITUDE OF x(t) ==================
        %  Keep only the last period (the most stable one)
        idx_last = (N - round(N/n) + 1):N;   % last perioc
        x_last = x_history(idx_last);

        X_real = max(x_last);

        Ns = round(length(x_history)*0.7);
        Np = length(x_history);

        X = x_history(Ns:N:Np,1);
        Y = v_history(Ns:N:Np,1);
 
        x1 = x_st(1);
        x2 = x_st(2);
end  


