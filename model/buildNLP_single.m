function [solver,par] = buildNLP_single(F,par)

% build NLP to solve a FHOCP using single shooting
% Written by Chenchen Zhou, 7 2023

import casadi.*

% global lbx ubx dx0 lbu ubu u0 nx nd nu


% %%
% % Fixed step Runge-Kutta 4 integrator
%    M = 4; % RK4 steps per interval
%    DT = par.tf/M;
% %    f = Function('f', {x, u}, {xdot, L});
%    X0 = MX.sym('X0', nx);
%    U = MX.sym('U',nu);
%    D = MX.sym('U',nd);
%    X = X0;
%    Q = 0;
%    for j=1:M
%        [k1, k1_q] = f(X, U,D);
%        [k2, k2_q] = f(X + DT/2 * k1, U,D);
%        [k3, k3_q] = f(X + DT/2 * k2, U,D);
%        [k4, k4_q] = f(X + DT * k3, U,D);
%        X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
%        Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
%     end
%     F = Function('F', {X0, U, D}, {X, Q}, {'x0','u','d'}, {'xf', 'qf'});

%% Build NLP solver

% empty nlp
w   = {};
w0  = [];
lbw = [];
ubw = [];
J   = 0;

g   = {};
lbg = [];
ubg = [];

% initial conditions for each scenario
% X0  = MX.sym('X0',nx);
% U0  = MX.sym('U0',nu);
x_init  = MX.sym('x_init',par.nx);
Dk    = MX.sym('Dk',par.nd);

X0  = x_init;
% w   = {w{:}, X0};
% lbw = [lbw;lbx];
% ubw = [ubw;ubx];
% w0  = [w0; dx0];

% Formulate NLP
Xk  = X0;

% Uk_prev = U0;

for k = 0:par.N-1
    
    Uk  = MX.sym(['U_' num2str(k)],par.nu);
    w   = {w{:},Uk};
    lbw = [lbw;par.lbu];
    ubw = [ubw;par.ubu];
    w0  = [w0;par.u0];
    
    % Integrate till the end of the interval
    Fk = F('x0',Xk,'p',vertcat(Uk,Dk));
%     Fk =  F('x0',Xk,'u',Uk,'d',Dk);
    Xk = Fk.xf;
    J=J+Fk.qf;
    % Add inequality constraint
    g = {g{:}, Xk};
    lbg = [lbg; par.lbx];
    ubg = [ubg; par.ubx];
     
end

% ---------- create and solve NLP solver ---------------
% Definition of the options for casadi
opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',1) ...
    );

nlp     = struct('x',vertcat(w{:}),'p',vertcat(x_init,Dk),'f',J,'g',vertcat(g{:}));
solver  = nlpsol('solver','ipopt',nlp,opts);

par.w0 = w0;
par.lbw = lbw;
par.ubw = ubw;
par.lbg = lbg;
par.ubg = ubg;
par.nlp = nlp;

