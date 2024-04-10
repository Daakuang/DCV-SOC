function [sys,par] = ddRobot(Ts,x0,H)
%
% Differential drive robots
% x= x y theta
% Written by: Chenchen Zhou, Apr. 2023 ZJU

import casadi.*
% Ts=1;
% t=MX.sym('t',1);

% States struct (optimization variables):

v_max = 0.6; v_min = -v_max;
omega_max = pi/4*1; omega_min = -omega_max;

x = MX.sym('x'); y = MX.sym('y'); theta = MX.sym('theta');
states = [x;y;theta];  n_states = length(states);

% Input struct (optimization variables):
if nargin <3
v = MX.sym('v'); omega = MX.sym('omega');
u = [v;omega]; %n_controls = length(controls);
elseif nargin>2
    nx = length(states);
    u=(H(1,:)-states'*H(2:nx+1,:))/H(nx+2:end,:);
    u=u';
    v=u(1);omega=u(2);
end

%  Uncertain parameters:
k =   MX.sym('k',3);
% x0 = MX.sym('x0', n_states);
P = MX.sym('P',n_states);
%Certain parameters

% disturbance

% system r.h.s
rhs = [v*cos(theta)+k(1);v*sin(theta)+k(2);omega+k(3)];


sys.diff = vertcat(rhs);
sys.x = vertcat(states);
sys.d = vertcat(k,P);
sys.u = vertcat(u);

Q = zeros(3,3); Q(1,1) = 1;Q(2,2) = 5;Q(3,3) = 0.1; % weighing matrices (states)
Q = Q;
R = zeros(2,2); R(1,1) = 0.5; R(2,2) = 0.05; % weighing matrices (controls)


sys.L_path = (states-P)'*Q*(states-P) + u'*R*u;
sys.L_terminal=0;
sys.isTFfree = 0;  %0:Not free 1:state cost 2: minimum time 3:state cost + minimum time

sys.nlcon = vertcat(x,y,theta);
sys.lb = vertcat(-2,-2,-inf);
sys.ub = vertcat(2,2,inf);  

sys.con_path=[1 2 3];
sys.y = vertcat(sys.x,sys.nlcon,sys.L_path,sys.L_terminal);
if nargin<3
    ode = struct('x',sys.x,'p',vertcat(sys.d,sys.u),'ode',sys.diff,'quad',vertcat(sys.L_path,sys.L_terminal));
else
    ode = struct('x',sys.x,'p',vertcat(sys.d),'ode',sys.diff,'quad',vertcat(sys.L_path,sys.L_terminal));
end
opts = struct('tf',Ts);
% integrator. print_options()

% create IDAS integrator
sys.F = integrator('F','cvodes',ode,opts);

% create cost function

if nargin<3
    sys.J = Function('J',{sys.x,sys.u,sys.d},{vertcat(sys.L_path,sys.L_terminal)});
else
    sys.J = Function('J',{sys.x,sys.d},{vertcat(sys.L_path,sys.L_terminal)});
end

par.lbx = [-2;-2;-inf];
par.ubx = [2;2;inf];
par.lbu = [v_min;omega_min];
par.ubu = [v_max; omega_max];
% par.dx0 = [10000;853;26.5;90;90;90;35;35;0;104.897];
par.x0 = x0;
par.u0 = [0;0];
par.ts = Ts;