function [sys,par] = BenchmarkCSTR(par)

% Benhcmark CSTR from Hicks and Ray, 1971, modified by Kameswaran and
% Biegler, 2006. 
% Written by Chenchen Zhou

import casadi.*

% states x
x1 = MX.sym('x1'); % Concentration
x2 = MX.sym('x2'); % Temperature

% input u
u = MX.sym('u'); % Energy flux from heating system

x_sp = MX.sym('x_sp',2); % Temperature
u_sp = MX.sym('u_sp',1); %0.7853

tau = 20;
M = 5;
xf = 0.3947;
xc = 0.3816;
a = 0.117;
k = 300;
ue=u_sp;%0.7853;

dx1 = (1/tau)*(1-x1) - k*x1*exp(-M/x2);
dx2 = (1/tau)*(xf-x2) + k*x1*exp(-M/x2) - a*u*(x2-xc);

diff = vertcat(dx1,dx2);
x_var = vertcat(x1,x2);
d_var = vertcat(x_sp,u_sp);
p_var = vertcat(u);

L = sum(([x1;x2]-x_sp).^2) + 1e-4*(u-ue).^2; % maintain desired temperature + min Uh heating costs? 

sys.f = Function('f',{x_var,p_var,d_var},{diff,L},{'x','p','d'},{'xdot','qj'});

ode = struct('x',x_var,'p',vertcat(p_var,d_var),'ode',diff,'quad',L); 

% create CVODES integrator
sys.F = integrator('F','cvodes',ode,struct('tf',par.tf));

sys.x = x_var;
sys.u = p_var;
sys.d = d_var;
sys.dx = diff;
sys.L = L;
sys.nlcon = [];

par.d0=[0.263193758468718;0.651883659521251;0.758340018600918];
d_init = [0.2632;0.6519];

par.lbx = [0.0632;0.4632];
par.ubx = [0.4519;0.8519];
par.lbu = [0];
par.ubu = [2];
par.x0 = d_init;
par.u0 = [0];

par.nx = numel(sys.x);
par.nu = numel(sys.u);
par.nd = numel(sys.d);

