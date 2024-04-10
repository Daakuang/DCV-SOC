clc
clear
% www.casadi.org
% Writte by: Chenchen
import casadi.*

FileName = mfilename('fullpath');
[directory,~,~] = fileparts(FileName);
[parent,~,~] = fileparts(directory);
addpath([directory '/model'])
addpath([parent '/functions'])
%%
par.tf = 0.5;

[sys,par] = bioreactor(par);
bx = par.lbx;
ubx = par.ubx;

par.N = 150/par.tf;

nData = 1;

par.ROC = 0;

n_w_i = par.nx + par.N*(4*par.nx+par.nu);
[Solver,par] = buildNLP(sys.f,par,sys.F);
w0=par.w0;

kk=1;
x_i = par.x0;
d_i = [0.03;0.005;200];%par.d0;[0.02;0.004;200];
u_i = par.u0;
% ------ SOLVE OFFLINE OCP -------
tic;
sol = Solver('x0',w0,'p',vertcat(x_i,u_i,d_i),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
elapsednlp = toc;

flag = Solver.stats();
exitflag =  flag.return_status;
disp(['Data point nr. ' num2str(kk)  ' - ' exitflag '. CPU time: ' num2str(elapsednlp) 's'])

Primal = full(sol.x);
coor_X=(par.nu+4*par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';
coor_U=(par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nu,1)+(0:par.nu-1)';

u_opt = Primal(coor_U);
x_opt = Primal(coor_X);
stackedplot([x_opt;u_opt']');
%%
par.tf = 0.5/3;
[sys,par] = bioreactor(par);

rng default
d3=200+200*0.2*(rand(1,2*3*150)-0.5);

X = [];
U = [];

Yx= 0.5; %g[X]/g[S]
Yp= 1.2; %g[P]/g[S]
Km= 0.05; %g/l
Ki= 5; %g/l

xk = par.x0;uk=0;dk=par.d0;
for i = 1:2*150
    % dk(3)=d3(i);
    for j =1:3
        mu = dk(1)*xk(2)/(Km+xk(2)+xk(2)*xk(2)/Ki);
        if full(xk(1))<3.46
            uk = xk(4)/(dk(3)-xk(2))*(mu*xk(1)/Yx+dk(2)*xk(1)/Yp);
        elseif full(xk(2))<0.0001 %&& full(xk(1))>=3.7
             % uk = mu*xk(4);
             uk = xk(4)/(dk(3)-xk(2))*(mu*xk(1)/Yx+dk(2)*xk(1)/Yp);
        else
            uk = 0;
        end
        uk = max(min(uk,1),0);
        fk=sys.f(xk,uk,dk);
        xk = xk+par.tf*fk;
        xk = max(xk,0);
    end
    X = [X,full(xk)];
    U = [U,full(uk)];
end
stackedplot([X;U]');