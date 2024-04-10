function [sys,par] = industrial_poly(par)

% industrial batch polymerization reactor
% Structural mismatch with WilliamOtto(Ts) function.
%
% Model taken from ZLucia S, Andersson J A E, Brandt H, et al. 
% Handling uncertainty in economic nonlinear model predictive control: A comparative case study
% [J]. Journal of Process Control, 2014, 24(8): 1247-1259.
%
% Written by: Chenchen Zhou, Apr. 2023 ZJU

import casadi.*

% States struct (optimization variables):
m_W= MX.sym('m_W',1);
m_A= MX.sym('m_A',1);
m_P= MX.sym('m_P',1);
accum_monom= MX.sym('accum_monom',1);
T_R= MX.sym('T_R',1);
T_S= MX.sym('T_S',1);
Tout_M= MX.sym('Tout_M',1);
T_EK= MX.sym('T_EK',1);
Tout_AWT= MX.sym('Tout_AWT',1);
T_adiab= MX.sym('T_adiab',1);

% Input struct (optimization variables):
m_dot_f = MX.sym('m_dot_f ',1);
T_in_M = MX.sym('T_in_M ',1);
T_in_EK= MX.sym('T_in_EK',1);

%  Uncertain parameters:
delH_R = MX.sym( 'delH_R',1);
k_0 =    MX.sym('k_0',1);

%Certain parameters
R=8.314;      %gas constant
T_F=25;      %feed temperature
E_a=8500;      %activation energy
% delH_R=950.0*1.00;      %sp reaction enthalpy
A_tank=65;      %area heat exchanger surface jacket 65
% k_0=7.0*1.00;      %sp reaction rate
k_U2=32;      %reaction parameter 1
k_U1=4;      %reaction parameter 2
w_WF=0.333;      %mass fraction water in feed
w_AF=0.667;      %mass fraction of A in feed
m_M_KW=5000;      %mass of coolant in jacket
fm_M_KW=300000;      %coolant flow in jacket 300000;
m_AWT_KW=1000;      %mass of coolant in EHE
fm_AWT_KW=100000;      %coolant flow in EHE
m_AWT=200;      %mass of product in EHE
fm_AWT=20000;      %product flow in EHE
m_S=39000;      %mass of reactor steel
c_pW=4.2;      %sp heat cap coolant
c_pS=0.47;      %sp heat cap steel
c_pF=3;      %sp heat cap feed
c_pR=5;      %sp heat cap reactor contents
k_WS=17280;      %heat transfer coeff water-steel
k_AS=3600;      %heat transfer coeff monomer-steel
k_PS=360;      %heat transfer coeff product-steel
alfa=5*20e4*3.6;      %
p_1=1;      %


Tset = 90;

% algebraic equations
U_m    = m_P / (m_A + m_P);
m_ges  = m_W + m_A + m_P;
k_R1   = k_0 * exp(- E_a/(R*(T_R+273.15))) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m));
k_R2   = k_0 * exp(- E_a/(R*(T_EK+273.15)))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m));
k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS);

% Differential equations
dot_m_W = m_dot_f * w_WF;
dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT);
dot_m_P = (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT);
dot_T_R = 1.0/(c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (delH_R * k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))));

dm_W= dot_m_W;
dm_A= dot_m_A;
dm_P= dot_m_P;
dT_R= dot_T_R;
dT_S= 1./(c_pS * m_S)*((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M)));
dTout_M= 1./(c_pW * m_M_KW)*((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M)));
dT_EK= 1./(c_pR * m_AWT)   * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * delH_R));
dTout_AWT= 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK)));
daccum_monom= m_dot_f;
dT_adiab= delH_R/(m_ges*c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*delH_R/(m_ges*m_ges*c_pR))+dot_T_R;


diff = vertcat(dm_W,dm_A,dm_P,dT_R,dT_S,dTout_M,dT_EK,dTout_AWT,daccum_monom,dT_adiab);
x_var = vertcat(m_W,m_A,m_P,T_R,T_S,Tout_M,T_EK,Tout_AWT,accum_monom,T_adiab);
d_var = vertcat(delH_R,k_0);
p_var = vertcat(m_dot_f,T_in_M,T_in_EK);


L =-m_P+1e4*(Tset-T_R)^2;%+5*1*m_dot_f^2+10*T_in_M^2+10*T_in_EK^2;% -1e4*m_P;%+1e4*max(Tset+2-T_R,0);%+1e4*min(Tset-2-T_R,0);%0.1*m_dot_f^2+0.02*T_in_M^2+0.01*T_in_EK^2; 

sys.f = Function('f',{x_var,p_var,d_var},{diff,L},{'x','p','d'},{'xdot','qj'});

ode = struct('x',x_var,'p',vertcat(p_var,d_var),'ode',diff,'quad',L); 

% create CVODES integrator
sys.F = integrator('F','cvodes',ode,struct('tf',par.tf));


sys.x = x_var;
sys.y = x_var([1 4:10]);
sys.u = p_var;
sys.d = d_var;
sys.dx = diff;
sys.L = L;
sys.nlcon = [];


par.lbx = [0;0;0;Tset-2.0;0;0;0;0;0;0];
par.ubx = [inf;inf;inf;Tset+2.0;100;100;100;100;30000;109];
% par.ubx = [inf;inf;inf;inf;100;100;100;100;30000;109];
par.lbu = [0;60;60];
par.ubu = [30000;100;100];
par.x0 = [10000;853;26.5;90;90;90;35;35;0;104.897];
% par.x0(end) = par.x0(2)*950/(par.x0(1)+par.x0(2)+par.x0(3)*c_pR)+par.x0(4)+273.15;
%x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']
%         (m_W,m_A,m_P,T_R,T_S,Tout_M,T_EK,Tout_AWT,accum_monom,T_adiab);
% par.x0 = x0;
par.u0 = [0;60;60];
par.d0 = [950.0;7.0];%3600;360;5*20e4*3.6]*1.0;

par.nx = numel(sys.x);
par.nu = numel(sys.u);
par.nd = numel(sys.d);

% for MHE problem

% par.P_x=eye(par.nx);
% par.P_d=eye(par.nd);
% par.P_y=diag(1./[1,0.1,0.1,0.1,0.1,0.1,1,1]);
% par.P_w=diag(1./[10,2,0.5,0.1,0.1,0.1,0.1,0.1,2,0.1]);
