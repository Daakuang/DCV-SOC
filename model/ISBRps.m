function [sys,par] = ISBRps(par)
% B. Srinivasan, C.J Primus, D. Bonvin, N.L. Ricker,
% Run-to-run optimization via control of generalized constraints,
% Control Engineering Practice,
% Volume 9, Issue 8,
% 2001,
%
% Written by: Chenchen Zhou, Dec. 2023 ZJU

import casadi.*

Ts = par.ts;

% States struct (optimization variables):
Ca = MX.sym('Ca',1);
Cb = MX.sym('Cb',1);
V = MX.sym('V',1);
Cc = MX.sym('Cc',1);
Cd = MX.sym('Cd',1);
x=[Ca;Cb;V;Cc;Cd];

% Input struct (optimization variables):
u = MX.sym('u');

%  Uncertain parameters:
Cb_in = MX.sym('Cb_in',1);
k1 = MX.sym('k1',1);
k2 = MX.sym('k2',1);

%Certain parameters
% k1=0.053;
% k2=0.128;
% Cb_in=5;
k3=0;0.028;
k4=0;0.001;
tf=250;
Ca_0=0.72;
Cb_0=0.076;%0.05
V_0=1;
Cc_0=0.08;
Cd_0=0.01;


% Model equations
dCa = -k1*Ca*Cb-Ca*u/V;
dCb = -k1*Ca*Cb-2*k2*Cb.^2-(Cb-Cb_in)*u/V-k3*Cb-k4*Cb*Cc;
dV = u;
dCc = k1*Ca*Cb-Cc*u/V-k4*Cb*Cc;
dCd = k2*Cb^2-Cd*u/V;

% Objective term
sys.L_path = 0;
sys.L_terminal= -Cc*V;
sys.isTFfree = 0;  % 1:state cost 2: minimum time 3:state cost + minimum time 0:free

sys.diff = vertcat(dCa,dCb,dV,dCc,dCd);
sys.x = vertcat(Ca,Cb,V,Cc,Cd);
sys.d = vertcat(k1,k2,Cb_in);
sys.u = vertcat(u);



sys.nlcon = [Cb;Cd];
sys.lb = [0;0];
sys.ub = [0.025;0.15];  
sys.con_path=[]; % which is path 
sys.y = vertcat(sys.x,sys.nlcon,sys.L_path,sys.L_terminal);

ode = struct('x',sys.x,'p',vertcat(sys.d,sys.u),'ode',sys.diff,'quad',vertcat(sys.L_path,sys.L_terminal));
opts = struct('tf',Ts);

% create IDAS integrator
sys.F = integrator('F','cvodes',ode,opts);

% create cost function
sys.J = Function('J',{sys.x,sys.u,sys.d},{vertcat(sys.L_path,sys.L_terminal)});

par.lbx = [0;0;0;0;0];
par.ubx = [inf;inf;inf;inf;inf];
par.lbu = [0];
par.ubu = [0.002];
% par.dx0 = [10000;853;26.5;90;90;90;35;35;0;104.897];
par.x0 = [Ca_0;Cb_0;V_0;Cc_0;Cd_0];
par.u0 = [0.002];


par.nx = numel(sys.x);
par.nu = numel(sys.u);
par.nd = numel(sys.d);


%% DCV

% sys.c = min((T_max-T_cf),V_max-V);