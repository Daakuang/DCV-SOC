function [sol,flag] = myOCP(sys,par,N,d)

% N number of decision steps
%

import casadi.*

[x_start,u_start]=myOCPinitial(sys.F,par,N,d);

isTFfree = sys.isTFfree;
con_path=sys.con_path;
if isTFfree >0
    tf = MX.sym('tf',1); %  Terminal time
else
    tf = N*par.ts;
end

nx = numel(sys.x);
nu = numel(sys.u);
nd = numel(sys.d);

d_s=MX.sym('d',nd); %  distrbance
% Uk = MX.sym('U',nu);
% Integrate till the end of the interval
M = 4; % RK4 steps per interval
DT = tf/N/M;
f = Function('f', {sys.x, sys.d, sys.u}, {sys.diff,sys.nlcon,sys.L_path,sys.L_terminal});
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
    [~,G,L_p,L_t] = f(X, d_s, U);
end
if isTFfree >0
    F = Function('F', {X0,vertcat(d_s,U),tf}, {X, G,L_p,L_t}, {'x0','p','tf'}, {'xf', 'gf','L_p','L_t'});
%     Fk = F('x0',X0,'p',[d;U],'tf',tf);
else
    F = Function('F', {X0,vertcat(d_s,U)}, {X, G,L_p,L_t}, {'x0','p'}, {'xf', 'gf','L_p','L_t'});
%     Fk = F('x0',X0,'p',[d;U]);
end

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
for k=1:N
    % New NLP variable for the control
    % if k >1
    %     Ukp1 = Uk;
    % end
    Uk = MX.sym(['U_' num2str(k)],nu);
    w   = {w{:} Uk};
    lbw = [lbw;par.lbu];
    ubw = [ubw;par.ubu];
    w0  = [w0;u_start(:,k)];
    discrete = [discrete;1];

if isTFfree >0
    Fk = F('x0',Xk,'p',[d;Uk],'tf',tf);

else
    Fk = F('x0',Xk,'p',[d;Uk]);
end

    if k==N
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
    if k<N
        g   = vertcat(g{:},Fk.gf(con_path));
        lbg = [lbg;sys.lb(con_path)];
        ubg = [ubg;sys.ub(con_path)];
    else
        g   = vertcat(g{:},Fk.gf);
        lbg = [lbg;sys.lb];
        ubg = [ubg;sys.ub];
    end
end



if isTFfree >0
    % Concatenate decision variables and constraint terms
    w = vertcat(tf,w{:});
    g = vertcat(g{:});
    w0=[eps;w0];
    lbw=[0;lbw];
    ubw=[inf;ubw];
    % Create an NLP solver
    nlp_prob = struct('f', cost, 'x', w, 'g', g);
    if isTFfree == 2 % minimum time problem
        nlp_prob = struct('f', tf, 'x', w, 'g', g);
    elseif isTFfree == 3
        nlp_prob = struct('f', tf+cost, 'x', w, 'g', g);
    end
else
    % Concatenate decision variables and constraint terms
    w = vertcat(w{:});
    g = vertcat(g{:});
    % Create an NLP solver
    nlp_prob = struct('f', cost, 'x', w, 'g', g);
end




% Create an NLP solver
% nlp_prob = struct('f', cost, 'x', w, 'g', g);
% nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, struct('discrete', discrete));
%nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});
% opts = struct('allow_free',true);
opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',0)...
    );
nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob,opts); % Solve relaxed problem


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