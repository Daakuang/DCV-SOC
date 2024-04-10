function [sys,par] = bioreactor(par)

% Fed-batch bioreactor with inhibition and a biomass constraint 
% E. Visser, B. Srinivasan, S. Palanki, D. Bonvin,
% A feedback-based implementation scheme for batch process optimization,
% Journal of Process Control,
% Volume 10, Issue 5,
% 2000,
% Pages 399-410,
% ISSN 0959-1524,
%
% Written by: Chenchen Zhou, Apr. 2023 ZJU

import casadi.*

% States struct (optimization variables):
X= MX.sym('X',1);
S= MX.sym('S',1);
P= MX.sym('P',1);
V= MX.sym('V',1);


% Input struct (optimization variables):
u = MX.sym('u ',1);


%  Uncertain parameters:
mu_m = MX.sym( 'mu_m',1); %l/h kinetic parameters
v = MX.sym( 'v',1);%l/h kinetic parameters
Sin = MX.sym( 'Sin',1);%

%Certain parameters
% % case 1 
% % mu_m= 0.53; %l/h kinetic parameters
% Km= 1.2; %g/l kinetic parameters
% Ki= 22; %g/l kinetic parameters
% Yx= 0.4; %
% Yp= 1; %
% % v= 0.5; %l/h kinetic parameters
% Sin= 20; %g/l
% umin= 0; %l/h
% umax= 1; %l/h
% Xmax= 3; %g/l
% tf= 8; %h

% case 2 
% mu_m= 0.02; %l/h
Km= 0.05; %g/l
Ki= 5; %g/l
Yx= 0.5; %g[X]/g[S]
Yp= 1.2; %g[P]/g[S]
% v= 0.004; %l/h
% Sin= 200; %g/l
umin= 0; %l/h
umax= 1; %l/h
Xmax= 3.7; %g/l
tf= 150; %h








% algebraic equations
mu = mu_m*S/(Km+S+S*S/Ki);

% Differential equations
dX = mu*X-u/V*X;
dS = -mu*X/Yx-v*X/Yp+u/V*(Sin-S);
dP = v*X-u/V*P;
dV = u;


diff = vertcat(dX,dS,dP,dV);
x_var = vertcat(X,S,P,V);
d_var = vertcat(mu_m,v,Sin);
p_var = vertcat(u);


L =0*(-dP+dS);
Ltf = -P;
sys.f = Function('f',{x_var,p_var,d_var},{diff,L,Ltf},{'x','p','d'},{'xdot','qj','qtf'});

ode = struct('x',x_var,'p',vertcat(p_var,d_var),'ode',diff,'quad',L); 

% create CVODES integrator
sys.F = integrator('F','cvodes',ode,struct('tf',par.tf));


sys.x = x_var;
% sys.y = x_var([1 4:10]);
sys.u = p_var;
sys.d = d_var;
sys.dx = diff;
sys.L = L;
sys.Ltf = Ltf;
sys.nlcon = [];


par.lbx = [0;0;0;0];
par.ubx = [Xmax;inf;inf;inf];
% par.ubx = [inf;inf;inf;inf;100;100;100;100;30000;109];
par.lbu = [umin];
par.ubu = [umax];
X0= 1; %g/l
S0= 0.5; %g/l
P0= 0; %g/l
V0= 150; %l
par.x0 = [X0;S0;P0;V0];
% par.x0(end) = par.x0(2)*950/(par.x0(1)+par.x0(2)+par.x0(3)*c_pR)+par.x0(4)+273.15;
%x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']
%         (m_W,m_A,m_P,T_R,T_S,Tout_M,T_EK,Tout_AWT,accum_monom,T_adiab);
% par.x0 = x0;
par.u0 = [0.05];
par.d0 = [0.02;0.004;200];%3600;360;5*20e4*3.6]*1.0;

par.nx = numel(sys.x);
par.nu = numel(sys.u);
par.nd = numel(sys.d);

par.istf = 1;

% for MHE problem

% par.P_x=eye(par.nx);
% par.P_d=eye(par.nd);
% par.P_y=diag(1./[1,0.1,0.1,0.1,0.1,0.1,1,1]);
% par.P_w=diag(1./[10,2,0.5,0.1,0.1,0.1,0.1,0.1,2,0.1]);
