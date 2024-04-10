par.ts = 1;
[sys,par] = ISBR(par);
par.N = 20;
d0 = 0.0483;
rng(1)
d = d0.*(1+0.4*(rand(1,100)-0.5));
data(length(d))=struct('tf',1,"y_opt",1,'u_opt',1);
for i = 1:1%length(d)
    % optimal solution
    [sol] = myOCP(sys,par,par.N,d(:,i));

    Primal = full(sol.x);
    data(i).tf = Primal(1);
    Primal = Primal(2:end);

    n_w_i = par.N*(par.nx+par.nu);

    coor_U=(1:(par.nx+par.nu):n_w_i).*ones(par.nu,1)+(0:par.nu-1)';
    coor_X=(par.nu+1:par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';

    data(i).u_opt = Primal(coor_U);
    x_opt = Primal(coor_X);
    c_C = (c_A0*V_0+c_C0*V_0-x_opt(1,:).*x_opt(3,:))./x_opt(3,:);
    T_cf = T+min(x_opt(1,:),x_opt(2,:))*-detH/pho/c_p;
    
    data(i).y_opt = [x_opt;c_C;T_cf];
    
    % plot optimal trajectory
    % figure(1)
    % subplot(4,1,1)
    % plot(linspace(0,data(i).tf,par.N),c_C,'b')
    % hold on
    % subplot(4,1,2)
    % plot(linspace(0,data(i).tf,par.N),T_cf,'b')
    % hold on
    % subplot(4,1,3)
    % plot(linspace(0,data(i).tf,par.N),x_opt(3,:),'b')
    % hold on
    % subplot(4,1,4)
    % plot(linspace(0,data(i).tf,par.N),data(i).u_opt,'b')
    % hold on
end

%% 
stackedplot([data(1).y_opt;data(1).u_opt']');

%% dcv 
par.ts = tf/100;
[sys,par] = ISBR(par);
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