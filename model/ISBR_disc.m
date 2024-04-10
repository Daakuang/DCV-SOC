function [sys,par] = ISBR_disc(par)
%
% Isothermal semi-batch reactor with a safety constraint
% Structural mismatch with WilliamOtto(Ts) function.
%
% brich, O., Srinivasan, B., Stoessel, F., & Bonvin, D. (1999). 
% Optimization of a semi-batch reaction system under safety constraints. 
% In European control conference (pp. F306.1 /F306.6). Karlsruhe, Germany
%
% Written by: Chenchen Zhou, Apr. 2023 ZJU

import casadi.*

Ts = par.ts;
% Ts=1;
% t=MX.sym('t',1);
% States struct (optimization variables):
c_A= MX.sym('c_A',1);
c_B= MX.sym('c_B',1);
V= MX.sym('V',1);

% Input struct (optimization variables):
u = MX.sym('u',1);

%  Uncertain parameters:
k = MX.sym('k',1);%0.0482
% detH = MX.sym('detH',1);%-60000;
% T = MX.sym('T',1);%-60000;
%Certain parameters
% k=0.0482;
T=70;
detH=-60000;
pho=900;
c_p=4.2;
c_Bin=2;
u_min=0;
u_max=0.1;
T_max=80;
V_max=1;
n_Cdes=0.6;
c_A0=2;
c_B0=0.63;
c_C0=0;
V_0=0.7;


dc_A = -k*c_A*c_B-u/V*c_A;
dc_B = -k*c_A*c_B+u/V*(c_Bin-c_B);
dV = u;

c_C = (c_A0*V_0+c_C0*V_0-c_A*V)/V;
T_cf = T+min(c_A,c_B)*-detH/pho/c_p;

% Objective term


sys.diff = vertcat(dc_A,dc_B,dV);
sys.x = vertcat(c_A,c_B,V);
sys.d = vertcat(k);
sys.u = vertcat(u);

sys.L_path = (n_Cdes-c_C*V).^2;%tf;%+0.1*m_dot_f^2+0.02*T_in_M^2+0.01*T_in_EK^2;
sys.L_terminal=0;
sys.isTFfree = 0;  % 1:state cost 2: minimum time 3:state cost + minimum time

sys.nlcon = vertcat(u,T_cf,V);
sys.lb = vertcat(u_min,-Inf,-Inf);
sys.ub = vertcat(u_max,T_max,V_max);  
sys.con_path=[1 2 3];
sys.y = vertcat(sys.x,sys.nlcon,sys.L_path,sys.L_terminal);

ode = struct('x',sys.x,'p',vertcat(sys.d,sys.u),'ode',sys.diff,'quad',vertcat(sys.L_path,sys.L_terminal));
opts = struct('tf',Ts);

% create IDAS integrator
sys.F = integrator('F','cvodes',ode,opts);

% create cost function
sys.J = Function('J',{sys.x,sys.u,sys.d},{vertcat(sys.L_path,sys.L_terminal)});

par.lbx = [0;0;0];
par.ubx = [inf;inf;inf];
par.lbu = [0];
par.ubu = [inf];
% par.dx0 = [10000;853;26.5;90;90;90;35;35;0;104.897];
par.x0 = [c_A0;c_B0;V_0];
par.u0 = [0.1];


par.nx = numel(sys.x);
par.nu = numel(sys.u);
par.nd = numel(sys.d);


%% DCV

sys.c = min((T_max-T_cf),V_max-V);