function [solver,par] = buildNLP(f,par,F,isg)

% build NLP to solve a FHOCP using collocation
% Written by CC zhou y, Oct 2023

import casadi.*

lbx=par.lbx;
ubx=par.ubx;
lbu=par.lbu;
ubu=par.ubu;
u0=par.u0;
nx=par.nx;
nd=par.nd;
nu=par.nu;
dx0=par.x0;

if nargin<4
    isg=0;
elseif nargin==4
    lbnlcon=par.lbnlcon;
    ubnlcon=par.ubnlcon;
    nnlcon = par.nnlcon;
end
%% Direct Collocation

% Degree of interpolating polynomial
d = 3;par.degree=d;

% Get collocation points
tau_root = [0, collocation_points(d, 'radau')];

% Coefficients of the collocation equation
C = zeros(d+1,d+1);

% Coefficients of the continuity equation
D = zeros(d+1, 1);

% Coefficients of the quadrature function
B = zeros(d+1, 1);

% Construct polynomial basis
for j=1:d+1
    % Construct Lagrange polynomials to get the polynomial basis at the collocation point
    coeff = 1;
    for r=1:d+1
        if r ~= j
            coeff = conv(coeff, [1, -tau_root(r)]);
            coeff = coeff / (tau_root(j)-tau_root(r));
        end
    end
    % Evaluate the polynomial at the final time to get the coefficients of the continuity equation
    D(j) = polyval(coeff, 1.0);
    
    % Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
    pder = polyder(coeff);
    for r=1:d+1
        C(j,r) = polyval(pder, tau_root(r));
    end
    
    % Evaluate the integral of the polynomial to get the coefficients of the quadrature function
    pint = polyint(coeff);
    B(j) = polyval(pint, 1.0);
end


%% Build NLP solver

% empty nlp
w   = {};
w0  = [];
lbw = [];
ubw = [];
J   = 0;

g   = {};
lbg = [];
ubg = [];

% initial conditions for each scenario
X0  = MX.sym('X0',nx);
U0  = MX.sym('U0',nu);
x_init  = MX.sym('x_init',nx);
Dk    = MX.sym('Dk',nd);

w   = {w{:}, X0};
lbw = [lbw;lbx];
ubw = [ubw;ubx];
w0  = [w0; dx0];

% Formulate NLP
Xk  = X0;

% Initial condition constraint
g   = {g{:},X0 - x_init};
lbg = [lbg;zeros(nx,1)];
ubg = [ubg;zeros(nx,1)];

Uk_prev = U0;

for k = 0:par.N-1
    
    Uk  = MX.sym(['U_' num2str(k)],nu);
    w   = {w{:},Uk};
    lbw = [lbw;lbu];
    ubw = [ubw;ubu];
    w0  = [w0;u0];
    
    Fk=F('x0',dx0,'p',vertcat(u0,par.d0));
    dx0 =  full(Fk.xf) ;

    Xkj = {};
    
    for j = 1:d
        Xkj{j} = MX.sym(['X_' num2str(k) '_' num2str(j)],nx);
        w   = {w{:},Xkj{j}};
        lbw = [lbw;lbx];
        ubw = [ubw;ubx];
        w0  = [w0; dx0];
    end
    
    % Loop over collocation points
    Xk_end  = D(1)*Xk;
    
    for j = 1:d
        % Expression for the state derivative at the collocation point
        xp  = C(1,j+1)*Xk;  % helper state
        for r = 1:d
            xp = xp + C(r+1,j+1)*Xkj{r};
        end
        if ~isg
            [fj,qj] =  f(Xkj{j},Uk,Dk);
            g   = {g{:},par.tf*fj-xp};  % dynamics and algebraic constraints
            lbg = [lbg;zeros(nx,1)];
            ubg = [ubg;zeros(nx,1)];
        else
            [fj,qj,gj] =  f(Xkj{j},Uk,Dk);
            nlcon  = gj; % constrains
            g   = {g{:},par.tf*fj-xp,nlcon};  % dynamics and algebraic constraints
            lbg = [lbg;zeros(nx,1);lbnlcon];
            ubg = [ubg;zeros(nx,1);ubnlcon];
        end
        
        % Add contribution to the end states
        Xk_end  = Xk_end + D(j+1)*Xkj{j};
        
        J   = J + (B(j+1)*qj*par.tf); % economic cost
    end
    
    J = J + par.ROC*((Uk_prev - Uk).^2);
    
%     Uk_prev = MX.sym(['Uprev_' num2str(k+1)],nu);
    Uk_prev = Uk;
    
    % New NLP variable for state at end of interval
    Xk      = MX.sym(['X_' num2str(k+1) ], nx);
    w       = {w{:},Xk};
    lbw     = [lbw;lbx];
    ubw     = [ubw;ubx];
    w0      = [w0; dx0];
    
    % Shooting Gap constraint
    g   = {g{:},Xk_end-Xk};
    lbg = [lbg;zeros(nx,1)];
    ubg = [ubg;zeros(nx,1)];
    
end

% 
g   = {g{:},Uk,Xk};
lbg = [lbg;par.d0(3)-0e-5*ones(nu,1);par.d0(1:2)-0e-5*ones(nx,1)];
ubg = [ubg;par.d0(3)+0e-5*ones(nu,1);par.d0(1:2)+0e-5*ones(nx,1)];
% case 3
% g   = {g{:},Uk,Xk};
% lbg = [lbg;-0e-5*ones(nu,1);-0e-5*ones(nx,1)];
% ubg = [ubg;+0e-5*ones(nu,1);+0e-5*ones(nx,1)];

% ---------- create and solve NLP solver ---------------
% Definition of the options for casadi   ,'max_iter',3000
opts = struct('warn_initial_bounds',false, ...
    'print_time',false, ...
    'ipopt',struct('print_level',0)...
    ,'calc_f',1 ...
    );

nlp     = struct('x',vertcat(w{:}),'p',vertcat(x_init,U0,Dk),'f',J,'g',vertcat(g{:}));
solver  = nlpsol('solver','ipopt',nlp,opts);

par.w0 = w0;
par.lbw = lbw;
par.ubw = ubw;
par.lbg = lbg;
par.ubg = ubg;
par.nlp = nlp;

