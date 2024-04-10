clc
clear

% uses CasADi v3.5.1
% www.casadi.org
% Writte by: Chenchen

import casadi.*

FileName = mfilename('fullpath');
[directory,~,~] = fileparts(FileName);
[parent,~,~] = fileparts(directory);
addpath([directory '/data'])
addpath([directory '/model'])
addpath([parent '/functions'])

%%
% global nx nu nd
% global lbx ubx dx0 lbu ubu u0
par.tf = 50/3600;

[sys,par] = industrial_poly(par);



% dx0 = par.x0;
% lbu = par.lbu;
% ubu = par.ubu;
% u0  = par.u0;

u_init = par.u0;
d_init = [950.0*1.00;7.0*1.00]*1.3;

lbx = par.lbx;
ubx = par.ubx;

par.N = 200;

nData = 1;

par.ROC = [2e-4 4e-3 2e-3];

par.scaling = [];
% par.scaling.x = [10 10 10 1 1 1 1 1 10 1]';
% par.scaling.u = [100 1 1]';
[mpcSolver,par] = buildNLP(sys.f,par,sys.F);
n_w_i = par.nx + par.N*(4*par.nx+par.nu);
sensitivity = 1;


rng default
d_var = d_init*(1+0.6*(rand(1,100)-0.5));

w0=par.w0;
for kk=1:length(d_var)
    
    x_i = par.x0;
    d_i = d_var(:,kk);
    u_i = u_init;

    % ------ SOLVE OFFLINE OCP -------
    tic;
    sol = mpcSolver('x0',w0,'p',vertcat(x_i,u_i,d_i),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
    elapsednlp = toc;

    flag = mpcSolver.stats();
    exitflag =  flag.return_status;
    disp(['Data point nr. ' num2str(kk)  ' - ' exitflag '. CPU time: ' num2str(elapsednlp) 's'])
%     assert(flag.success==1,'Error! solveOCP unsuccessful')

    Primal = full(sol.x);
    Dual.lam_g = full(sol.lam_g);
    Dual.lam_x = full(sol.lam_x);
    Dual.lam_p = full(sol.lam_p);

    indAS = find(round(Dual.lam_x,6) >0);

    
    coor_X=(par.nu+4*par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nx,1)+(0:par.nx-1)';
    coor_U=(par.nx+1:4*par.nx+par.nu:n_w_i).*ones(par.nu,1)+(0:par.nu-1)';

    u_opt = Primal(coor_U);
    x_opt = Primal(coor_X);

    ind_X=sort(coor_X(:));
    ind_U = sort(coor_U(:));
    ind_notU = setdiff(1:length(Primal),ind_U(:));

    
    if flag.success && kk==1 %&& x_opt(3,end)>=20680 && sensitivity
        Lar = par.nlp.f+Dual.lam_x'*par.nlp.x+Dual.lam_g'*par.nlp.g+Dual.lam_p'*par.nlp.p;
        Lw = jacobian(Lar,par.nlp.x);
        Lx = Lw(ind_notU);
        Lu = Lw(ind_U);
%         Lww = hessian(Lar,par.nlp.x);
        dgdw = jacobian(vertcat(par.nlp.g),par.nlp.x);
        dxdu = -dgdw(:,ind_notU)'\dgdw(:,ind_U);
        Lu = Lu + Lx*dxdu;
        Lu_f = Function('Lu',{vertcat(par.nlp.p,par.nlp.x)},{Lu});  
%         dLudw = jacobian(Lu,par.nlp.x);
%         Luu = dLudw(:,ind_U)+dLudw(:,ind_notU)*dxdu;
%         Luu_f = Function('Luu',{vertcat(par.nlp.p,par.nlp.x)},{Luu});        
    end
    if flag.success %&& x_opt(3,end)>=20680
        GenData.u{kk} = u_opt;
        GenData.x{kk} = x_opt;
        GenData.d{kk} = d_i;
        GenData.sol_t(kk) = elapsednlp;
        tic;
%         Luu_v  = full(Luu_f(vertcat(x_i,u_i,d_i,Primal)));
        %first order
%         for i = 1:length(ind_U)
%             for j = 1:length(ind_U)
%                 epsilon = zeros(size(Primal));
%                 epsilon(ind_U(i)) = 1e-5;
%                 Ai= full(Lu_f(vertcat(x_i,u_i,d_i,Primal+epsilon)));
%                 A = [A; (Ai-full(sol.f))/1e-5];
%             end
%         end


        A0 = full(Lu_f(vertcat(x_i,u_i,d_i,Primal)));
        A=[];
        for i = 1:length(ind_U)
            if mod(i,par.nu)==1
                p = 1;
            else
                p=1;
            end
            epsilon = zeros(size(Primal));
            epsilon(ind_U(i)) = p*1e-8;
            Ai= full(Lu_f(vertcat(x_i,u_i,d_i,Primal+epsilon)));
            A = [A; (Ai-A0)/p/1e-8];
        end
        Luu_v = (A+A')/2;
        GenData.Luu{kk} = Luu_v;
        elapsedSen = toc;
        GenData.sen_t(kk) = elapsedSen;
        disp(['Data point nr. ' num2str(kk)  ' - Luu . CPU time: ' num2str(elapsedSen) 's'])
    end

end


save('data/GenData','GenData')

%%
kk=par.N;
    figure(110)
    for i = 1:par.nu
    subplot(par.nu+4,1,i+4)
    stairs((0:kk-1)*par.tf,GenData.u{1}(i,:))
    end
    subplot(par.nu+4,1,1)
    plot((0:kk-1)*par.tf,GenData.x{1}(10,:))
    hold on 
%     plot((0:sim_k-1)*par.tf,ones(1,sim_k)*par.lbx(10),'r--')
    plot((0:kk-1)*par.tf,ones(1,kk)*par.ubx(10),'r--')
    hold off
    subplot(par.nu+4,1,2)
    plot((0:kk-1)*par.tf,GenData.x{1}(9,:))
    subplot(par.nu+4,1,3)
    plot((0:kk-1)*par.tf,GenData.x{1}(4,:))
    hold on 
    plot((0:kk-1)*par.tf,ones(1,kk)*par.lbx(4),'r--')
    plot((0:kk-1)*par.tf,ones(1,kk)*par.ubx(4),'r--')
    hold off
    subplot(par.nu+4,1,4)
    plot((0:kk-1)*par.tf,GenData.x{1}(3,:))