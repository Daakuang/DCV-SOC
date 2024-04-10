import casadi.*

%%
par.N = 1000;
par.ts = 250/par.N;
[sys,par] = ISBRps(par);
d = [0.053;0.128;5;];
[sol,flag]  = myOCP(sys,par,par.N,d);
flag.success == 1
% optimal solution
Primal = full(sol.x);
% tf = Primal(1);
% Primal = Primal(2:end);

% n_w_i = par.nx + par.N*(4*par.nx+par.nu);
% coor_X=(par.nu+4*par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';
% coor_U=(par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nu,1)+(0:par.nu-1)';

n_w_i = par.N*(par.nx+par.nu);

coor_U=(1:(par.nx+par.nu):n_w_i).*ones(par.nu,1)+(0:par.nu-1)';
coor_X=(par.nu+1:par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';

u_opt = Primal(coor_U);
x_opt = Primal(coor_X);
figure(1)
stackedplot([x_opt;u_opt';]');
%%
figure(1)
stackedplot([diff([par.x0,x_opt]')';u_opt';]');

%% dcv 
par.ts = tf/100;
[sys,par] = ISBRps(par);
[sol,flag] = myDCVsim(sys,par,d);
flag.success == 1

Primal = full(sol.x);

% n_w_i = par.nx + par.N*(4*par.nx+par.nu);
% coor_X=(par.nu+4*par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';
% coor_U=(par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nu,1)+(0:par.nu-1)';

n_w_i = par.N*(par.nx+par.nu);

coor_U=(1:(par.nx+par.nu):n_w_i).*ones(par.nu,1)+(0:par.nu-1)';
coor_X=(par.nu+1:par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';

u = Primal(coor_U);
x = Primal(coor_X);

figure(2)
stackedplot([x;u']');