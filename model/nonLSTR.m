function [sys,par] = nonLSTR(par)
%
% A multimodal optimal control problem has been used by Luus (R. Luus, Iterative Dynamic Programming)
% Also, this problem is a member of the list of benchmark problems 
% proposed in the Handbook of Test Problems in Local and Global Optimization
%
%
% Written by: Chenchen Zhou, Jan. 2024 ZJU

import casadi.*


% States struct (optimization variables):
x1= MX.sym('x1',1);
x2= MX.sym('x2',1);

% Input struct (optimization variables):
u = MX.sym('u',1);

%  Uncertain parameters:
a = MX.sym('a',1);% 25
b = MX.sym('b',1);%0.5
% detH = MX.sym('detH',1);%-60000;
% T = MX.sym('T',1);%-60000;
%Certain parameters
% k=0.0482;


dx1 = (x2+b)*exp(a*x1/(x1+2))-(2+u)*(x1+0.25);
dx2 = 0.5-x2-(x2+b)*exp(a*x1/(x1+2));


% Objective term
diff = vertcat(dx1,dx2);
x_var = vertcat(x1,x2);
d_var = vertcat(a,b);
p_var = vertcat(u);


L = x1.^2+x2.^2+0.1*u.^2;

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

par.lbx = [-inf;-inf];
par.ubx = [inf;inf];
par.lbu = [-0.5];
par.ubu = [5];
par.d0 = [25;0.5];
par.x0 = [0.09;0.09];
par.u0 = [0];

par.nx = numel(sys.x);
par.nu = numel(sys.u);
par.nd = numel(sys.d);

