function [x_v,u_v,J_v,Ju_v,Juu_v,Gy_v,Gyd_v,Jud_v,Jd_v]=semibacthReactor(dd)
import casadi.*

T = 250; % Time horizon
N = 20; % number of control intervals
k1=dd.k1;k2=dd.k2;Cb_in=dd.Cb_in;Ca_0=dd.Ca_0;Cb_0=dd.Cb_0;V_0=dd.V_0;tf=dd.tf;Cc_0=dd.Cc_0;Cd_0=dd.Cd_0;
% k1=0.053;k2=0.128;Cb_in=5;Ca_0=0.72;Cb_0=0.0614;V_0=1;tf=250;Cc_0=0;Cd_0=0;
% k1=0.053;k2=0.128;Cb_in=5;Ca_0=0.72;Cb_0=0.05;V_0=1;tf=250;Cc_0=0;Cd_0=0;
% Declare model variables
Cb_in = MX.sym('Cb_in',1);
k1 = MX.sym('k1',1);
k2 = MX.sym('k2',1);
Ca = MX.sym('Ca',1);
Cb = MX.sym('Cb',1);
Cc = MX.sym('Cc',1);
Cd = MX.sym('Cd',1);
V = MX.sym('V',1);
x=[Ca;Cb;V;Cc;Cd];
u = MX.sym('u');
% Model equations
xdot = [-k1*Ca*Cb-Ca*u/V;...
       -k1*Ca*Cb-2*k2*Cb^2-(Cb-Cb_in)*u/V;...
       u;...
       k1*Ca*Cb-Cc*u/V;...
       k2*Cb^2-Cd*u/V];
% rhs_alg = [Cc-(Ca_0*V_0-Ca*V)/V;
%            Cd-((Ca+Cb_in-Cb*V-(Ca_0+Cb_in-Cb_0)/2/V))];
% Objective term
L = (-Cc+Cd)*V;

% Continuous time dynamics
f = casadi.Function('f', {x, u}, {xdot});

% Formulate discrete time dynamics
% Fixed step Runge-Kutta 4 integrator
M = 8; % RK4 steps per interval
DT = T/N/M;
% f = Function('f', {x, u}, {xdot, L});
X0 = MX.sym('X0', length(x));
U = MX.sym('U');
X = X0;
Q = 0;
for j=1:M
    kk1 = f(X, U);
    kk2 = f(X + DT/2 * kk1, U);
    kk3 = f(X + DT/2 * kk2, U);
    kk4 = f(X + DT * kk3, U);
    X=X+DT/6*(kk1 +2*kk2 +2*kk3 +kk4);
%     Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
%     Q = k1_q;
end
F = Function('F', {X0, U}, {X}, {'x0','p'}, {'xf'});
% Q = (-F(4)+F(5))*F(3);

% Start with an empty NLP
w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];

% "Lift" initial conditions
% X0 = MX.sym('X0', length(x));
% % X0 = [Ca_0;Cb_0;V_0;Cc_0;Cd_0];
% w = {w{:}, X0};
% lbw = [lbw; 0; 0; 0; 0; 0];
% ubw = [ubw; 100; 100; 100; 100; 100];
w0 = [w0; 0; 1; 1; 1; 1];
X = {};
y = {};
% Formulate the NLP
Xk = X0;
for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)]);
    w = {w{:}, Uk};
    lbw = [lbw; 0];
    ubw = [ubw;  0.001];
    w0 = [w0;  0];
    
    % Integrate till the end of the interval
    Fk = F('x0', Xk, 'p', Uk);
    Xk_end = Fk.xf;
%     J=J+Fk.qf;
    
    % New NLP variable for state at end of interval
%     Xk = MX.sym(['X_' num2str(k+1)], length(x));
    X = {X{:},Xk};
    yk = Xk(1:3);
    y = {y{:}, yk, Uk};
    Xk = Xk_end;
%     w = {w{:}, Xk};
%     lbw = [lbw; -inf; -inf; -inf; -inf; -inf];
%     ubw = [ubw;  inf;  inf;  inf;  inf;  inf];
%     w0 = [w0; 0; 0; 0; 0; 0];
    
    % Add equality constraint
%     if k ==N-1
%     g = {g{:}, Xk_end([2 5])};
%     lbg = [lbg; 0; 0];
%     ubg = [ubg; 0.025; 0.15];
%     end
end
% X0 = [Ca_0;Cb_0;V_0;Cc_0;Cd_0];
% J = (-Xk_end(4)+Xk_end(5))*Xk_end(3);
J = (-Xk_end(4)+Xk_end(5))*Xk_end(3);
Ju = jacobian(J,vertcat(w{:}));
Jud = jacobian(Ju,[X0;k1;k2;Cb_in]);
Gyd = jacobian(vertcat(y{:}),[X0;k1;k2;Cb_in]);
Jd = jacobian(J,[X0;k1;k2;Cb_in]);

Jud_F = Function('f',{[X0;k1;k2;Cb_in];vertcat(w{:})},{Jud});
Gyd_F = Function('f',{[X0;k1;k2;Cb_in];vertcat(w{:})},{Gyd});
Jd_F = Function('f',{[X0;k1;k2;Cb_in];vertcat(w{:})},{Jd});

k1=dd.k1;k2=dd.k2;Cb_in=dd.Cb_in;
xdot = [-k1*Ca*Cb-Ca*u/V;...
       -k1*Ca*Cb-2*k2*Cb^2-(Cb-Cb_in)*u/V;...
       u;...
       k1*Ca*Cb-Cc*u/V;...
       k2*Cb^2-Cd*u/V];
% rhs_alg = [Cc-(Ca_0*V_0-Ca*V)/V;
%            Cd-((Ca+Cb_in-Cb*V-(Ca_0+Cb_in-Cb_0)/2/V))];
% Objective term
L = (-Cc+Cd)*V;

% Continuous time dynamics
f = casadi.Function('f', {x, u}, {xdot});

% Formulate discrete time dynamics
% Fixed step Runge-Kutta 4 integrator
M = 8; % RK4 steps per interval
DT = T/N/M;
% f = Function('f', {x, u}, {xdot, L});
X0 = MX.sym('X0', length(x));
U = MX.sym('U');
X = X0;
Q = 0;
for j=1:M
    kk1 = f(X, U);
    kk2 = f(X + DT/2 * kk1, U);
    kk3 = f(X + DT/2 * kk2, U);
    kk4 = f(X + DT * kk3, U);
    X=X+DT/6*(kk1 +2*kk2 +2*kk3 +kk4);
%     Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q);
%     Q = k1_q;
end
F = Function('F', {X0, U}, {X}, {'x0','p'}, {'xf'});

X0 = [Ca_0;Cb_0;V_0;Cc_0;Cd_0];

w={};
w0 = [];
lbw = [];
ubw = [];
J = 0;
g={};
lbg = [];
ubg = [];
w0 = [w0; 0; 1; 1; 1; 1];
X = {};
y = {};
% Formulate the NLP
Xk = X0;
for k=0:N-1
    % New NLP variable for the control
    Uk = MX.sym(['U_' num2str(k)]);
    w = {w{:}, Uk};
    lbw = [lbw; 0];
    ubw = [ubw;  0.001];
    w0 = [w0;  0];
    
    % Integrate till the end of the interval
    Fk = F('x0', Xk, 'p', Uk);
    Xk_end = Fk.xf;
%     J=J+Fk.qf;
    
    % New NLP variable for state at end of interval
%     Xk = MX.sym(['X_' num2str(k+1)], length(x));
    X = {X{:},Xk};
    yk = Xk(1:3);
    y = {y{:}, yk, Uk};
    Xk = Xk_end;
%     w = {w{:}, Xk};
%     lbw = [lbw; -inf; -inf; -inf; -inf; -inf];
%     ubw = [ubw;  inf;  inf;  inf;  inf;  inf];
%     w0 = [w0; 0; 0; 0; 0; 0];
    
    % Add equality constraint
%     if k ==N-1
%     g = {g{:}, Xk_end([2 5])};
%     lbg = [lbg; 0; 0];
%     ubg = [ubg; 0.025; 0.15];
%     end
end
% X0 = [Ca_0;Cb_0;V_0;Cc_0;Cd_0];
J = (-Xk_end(4)+Xk_end(5))*Xk_end(3);
Ju = jacobian(J,vertcat(w{:}));
Juu = jacobian(Ju,vertcat(w{:}));
Gy = jacobian(vertcat(y{:}),vertcat(w{:}));

% Gd = jacobian([k1,k2,Ca_0,Cb_0,V_0],vertcat(w{:}));
Ju_F = Function('f',{vertcat(w{:})},{Ju});
Juu_F = Function('f',{vertcat(w{:})},{Juu});
Gy_F =  Function('f',{vertcat(w{:})},{Gy});
% Gd_F =  Function('f',{vertcat(w{:})},{Gd});
% J = (-Xk_end(4))*Xk_end(3);
% Create an NLP solver
% J = J('X0',{[Ca_0;Cb_0;V_0;Cc_0;Cd_0]},'U_0',w{1},'U_1',w{2},'U_2',w{3},'U_3',w{4},'U_4',w{5},'U_5',w{6},'U_6',w{7},'U_7',w{8},'U_8',w{9},'U_9',w{10}...
% ,'U_1','U_10',w{11},'U_11',w{12},'U_12',w{13},'U_13',w{14},'U_14',w{15},'U_15',w{16},'U_16',w{17},'U_17',w{18},'U_18',w{19},'U_19',w{20});
prob = struct('f', J, 'x', vertcat(w{:}), 'g', vertcat(g{:}));
options = struct('ipopt',struct('print_level',0),'print_time',false);
solver = nlpsol('solver', 'ipopt', prob, options);

uu = solver('x0',ones(N,1)*6.5e-4,'lbg',lbg,'ubg',ubg,'lbx',lbw,'ubx',ubw);
J_v = full(uu.f);
Ju_v = full(Ju_F(full(uu.x)));
Juu_v  = full(Juu_F(full(uu.x)));
Gy_v = full(Gy_F(full(uu.x)));
Gyd_v = full(Gyd_F([X0;k1;k2;Cb_in],full(uu.x)));
% Gyd_v = Gyd_v(:,[1:3 6:7]);
Jud_v = full(Jud_F([X0;k1;k2;Cb_in],full(uu.x)));
% Jud_v = Jud_v(:,[1:3 6:7]);
Jd_v = full(Jd_F([X0;k1;k2;Cb_in],full(uu.x)));
% Gd_v = Gd_F([k1,k2,Ca_0,Cb_0,V_0]);
%%
% output y 
Xk = X0;
xx=X0;
for k=1:N
    Fk = F('x0', Xk, 'p', uu.x(k));
    Xk_end = Fk.xf;
    Xk = Xk_end;
    xx = [xx,Xk];
end
x_v = full(xx);
u_v = full(uu.x);
end