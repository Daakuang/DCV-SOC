function [sol,flag] = myDCVsim(sys,par,d)

% N number of decision steps
%

import casadi.*

[x_start,u_start]=myOCPinitial(sys.F,par,par.N,d);

tf = par.N*par.ts;
con_path=sys.con_path;

nx = numel(sys.x);
nu = numel(sys.u);
nd = numel(sys.d);



% dae = struct('x',sys.x,'z',sys.u,'p',sys.d,'ode',sys.diff,'alg',sys.c);
% opts = struct('tf',par.ts);
% 
% F = integrator('F', 'idas', dae, opts);
% 
% r = F('x0',par.x0,'z0',par.u0,'p',d);


d_s=MX.sym('d',nd); %  distrbance
% Uk = MX.sym('U',nu);
% Integrate till the end of the interval
M = 4; % RK4 steps per interval
DT = tf/par.N/M;
f = Function('f', {sys.x, sys.d, sys.u}, {sys.diff,sys.nlcon,sys.L_path,sys.L_terminal,sys.c});
X0 = MX.sym('X0', nx);
U = MX.sym('U', nu);
X = X0;
Q = 0;
G = 0;
for j=1:M
    [k1, k1_g] = f(X, d_s, U);
    [k2, k2_g] = f(X + DT/2 * k1, d_s, U);
    [k3, k3_g] = f(X + DT/2 * k2, d_s, U);
    [k4, k4_g] = f(X + DT * k3, d_s, U);
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4);
    %         Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
    %         G = G + DT/6*(k1_g + 2*k2_g + 2*k3_g + k4_g);
    [~,G,L_p,L_t,cf] = f(X, d_s, U);
end

F = Function('F', {X0,vertcat(d_s,U)}, {X, G,L_p,L_t,cf}, {'x0','p'}, {'xf', 'gf','L_p','L_t','cf'});


% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
discrete = [];
cost = 0;
g={};
lbg = [];
ubg = [];

% "Lift" initial conditions
% Formulate the NLP
Xk = par.x0;
for k=1:par.N
    % New NLP variable for the control

    Uk = MX.sym(['U_' num2str(k)],nu);
    w   = {w{:} Uk};
    lbw = [lbw;par.lbu];
    ubw = [ubw;par.ubu];
    w0  = [w0;u_start(:,k)];
    discrete = [discrete;1];

    Fk = F('x0',Xk,'p',[d;Uk]);

    if k==par.N
        cost=cost+Fk.L_p+Fk.L_t;
    elseif k>1
        cost=cost+Fk.L_p;
    end
    % New NLP variable for state at end of interval
    Xk = MX.sym(['X_' num2str(k+1)], nx);
    w   = {w{:} Xk};
    lbw = [lbw;par.lbx];
    ubw = [ubw;par.ubx];
    w0  = [w0;x_start(:,k+1)];
    discrete = [discrete;0;0];

    Xk_end = Fk.xf;
    % Add equality constraint
    g   = {g{:};Xk_end-Xk};
    lbg = [lbg;zeros(nx,1)];
    ubg = [ubg;zeros(nx,1)];
    %     Xk = Fk.xf;

    % Add inequality nonlinear constraint
    % if k<par.N
    %     g   = vertcat(g{:},Fk.gf(con_path));
    %     lbg = [lbg;sys.lb(con_path)];
    %     ubg = [ubg;sys.ub(con_path)];
    % else
    %     g   = vertcat(g{:},Fk.gf);
    %     lbg = [lbg;sys.lb];
    %     ubg = [ubg;sys.ub];
    % end
    % Add equality nonlinear constraint DCV
    g   = vertcat(g{:},Fk.cf);
    lbg = [lbg;zeros(nu,1)];
    ubg = [ubg;zeros(nu,1)];
end




% Concatenate decision variables and constraint terms
w = vertcat(w{:});
g = vertcat(g{:});
% Create an NLP solver
nlp_prob = struct('f', 1, 'x', w, 'g', g);





% Create an NLP solver
% nlp_prob = struct('f', cost, 'x', w, 'g', g);
% nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, struct('discrete', discrete));
%nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});
% opts = struct('allow_free',true);
opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',2)...
    );
nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob,opts);


% Solve the NLP
sol = nlp_solver('x0',w0, 'lbx',lbw, 'ubx',ubw, 'lbg',lbg, 'ubg',ubg);
flag = nlp_solver.stats();
% exitflag =  flag.return_status;
% assert(flag.success == 1,'Error: OCP solver failed !')

end

function [x_start,u_start]=myOCPinitial(F,par,N,d)

% Initial guess for u
u_start = par.u0*ones(1,N);

% Get a feasible trajectory as an initial guess
xk = par.x0;
if ~isnumeric(xk)
    xk=(par.lbx+par.ubx)/2;
    if sum(isinf(xk))>0 || sum(isnan(xk))>0
        xk(isinf(xk))=0;
        xk(isnan(xk))=0;
    end
end
x_start = xk;
for k=1:N
    ret = F('x0',xk, 'p',[d;u_start(:,k)]);
    xk = ret.xf;
    x_start = [x_start xk];
end
end