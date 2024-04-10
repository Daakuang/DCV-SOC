%%
addpath('model/')
clear;
%%
% A optimal control problem

par.tf = 1;
[sys,par] = BenchmarkCSTR(par);
par.N = 140;
% par.d0 = [0.9;0.9];

u_init = par.u0;
d_init = par.d0;
lbx = par.lbx;
ubx = par.ubx;
par.ROC = [0];
par.u0 = 0.7853;
[mpcSolver,par] = buildNLP(sys.f,par,sys.F);
%
n_w_i = par.nx + par.N*((par.degree+1)*par.nx+par.nu);
coor_U=(par.nx+1:(par.degree+1)*par.nx+par.nu:n_w_i).*ones(par.nu,1)+(0:par.nu-1)';
% ind_X=sort(coor_X(:));
ind_U = sort(coor_U(:));
ind_notU = setdiff(1:length(par.nlp.x),ind_U(:));
coor_X = reshape(ind_notU,par.nx,[]);
coor_X = reshape(setdiff(coor_X,coor_X(:,(1+par.degree+1):par.degree+1:end)),par.nx,[]);
%%
import casadi.*
dual_g= MX.sym('dual_g',length(par.nlp.g));
dual_x= MX.sym('dual_x',length(par.nlp.x));
% dual_p= MX.sym('dual_p',length(par.nlp.p));
% 
tic
Lw_f =mpcSolver.get_function('nlp_grad');% Function calculating f, g and the gradient of the Lagrangian w.r.t. x and p
gw_f =mpcSolver.get_function('nlp_jac_g');% Function Jacobian of the constraints
Lww_f = mpcSolver.get_function('nlp_hess_l');
Lww = Lww_f(par.nlp.x,par.nlp.p,1,dual_g);

[~,~,Lw] = Lw_f(par.nlp.x,par.nlp.p,1,dual_g); % 1 is lam_f, is fixed here stand for min problem;get the gradient of the Lagrangian w.r.t. x and p
Lw=Lw+dual_x; % plus Lagrange multipliers for decision variables
[~,dgdw] = gw_f(par.nlp.x,par.nlp.p); % get Function Jacobian of the constraints
Lx = Lw(ind_notU);
Lu = Lw(ind_U);
% dgdw = jacobian(vertcat(par.nlp.g),par.nlp.x);
dxdu = -(dgdw(:,ind_notU)'*dgdw(:,ind_notU))\dgdw(:,ind_notU)'*dgdw(:,ind_U);

Luu = Lww(ind_U,ind_U)+Lww(ind_U,ind_notU)*dxdu+dxdu'*Lww(ind_notU,ind_U)+2*dxdu'*Lww(ind_notU,ind_notU)*dxdu;
Luu = (Luu'+Luu)/2;

% Lu = Lu + dxdu'*Lx;
% % Lu_f = Function('Lu',{vertcat(par.nlp.x,par.nlp.p,dual_g,dual_x)},{Lu});
dxdu_f = Function('Lu',{vertcat(par.nlp.x,par.nlp.p,dual_g,dual_x)},{dxdu});
% % Hessian matrix
% dLudw = jacobian(Lu,par.nlp.x);
% Luu = dLudw(:,ind_U)+dLudw(:,ind_notU)*dxdu;
Luu_f = Function('Luu',{vertcat(par.nlp.x,par.nlp.p,dual_g,dual_x)},{Luu});
toc
%% mode1
[X10,X20] = meshgrid(linspace(par.lbx(1),par.ubx(1),20),linspace(par.lbx(2),par.ubx(2),20));
X0 = mat2cell([X10(:),X20(:)],ones(1,numel(X10)));
% [mpcSolver,par] = buildNLP(sys.f,par,sys.F);
% U_opt=cell(1,numel(X10));X_opt=cell(1,numel(X10));J=zeros(1,numel(X10));

% sol = mpcSolver('x0',par.w0,'p',vertcat(x_0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
tic
[sol,flag,f,MPCTime]=cellfun(@(x)mySolver(mpcSolver,par.w0,x',par.u0,par.d0,par),X0,'UniformOutput',false);
toc
%%
rng default
X0Test = rand(2,50).*(par.ubx-par.lbx)+par.lbx;
X0Test = mat2cell(X0Test',ones(50,1),2);
[solTest,flagTest,fTest,MPCTime]=cellfun(@(x)mySolver(mpcSolver,par.w0,x',par.u0,par.d0,par),X0Test,'UniformOutput',false);
%
tic
Luu_v=cellfun(@(x,sol)full(Luu_f(vertcat(sol.x,x',par.u0,par.d0,sol.lam_x,sol.lam_g))),X0,sol,'UniformOutput',false);
LuuTime = toc;
Luu=cellfun(@(x)x(1:par.nu,1:par.nu),Luu_v);
%
f=cell2mat(f);flag=cell2mat(flag);sol(~flag)=[];X0(~flag)=[];f(~flag)=[];Luu(~flag)=[];
fTest=cell2mat(fTest);flagTest=cell2mat(flagTest);solTest(~flagTest)=[];X0Test(~flagTest)=[];fTest(~flagTest)=[];
%%
[X_opt,U_opt] = cellfun(@(sol)getOpt(sol,coor_U,coor_X),sol','UniformOutput',false);
[X_optTest,U_optTest] = cellfun(@(sol)getOpt(sol,coor_U,coor_X),solTest','UniformOutput',false);
save Case1ForWR U_opt  X_opt  f flag Luu sol U_optTest  X_optTest  fTest flagTest solTest
%%
% figure(1)
% plot((0:length(U_opt{1})-1)*par.tf,U_opt{1})
% hold on
% figure(2)
% plot(X_opt{1}(1,:),X_opt{1}(2,:),'b-')
% hold on
% for i =2:numel(X10)
%     try
%     figure(1)
%     plot((0:length(U_opt{i})-1)*par.tf,U_opt{i})
%     figure(2)
%     plot(X_opt{i}(1,:),X_opt{i}(2,:),'b-')
%     catch
%         continue
%     end
% end
% figure(2)
% hold off
% figure(1)
% hold off
%%
X = cell2mat(cellfun(@(x)x(:,1),X_opt,'UniformOutput',false));
U = cell2mat(cellfun(@(x)x(:,1),U_opt,'UniformOutput',false));

x = X;
t = U;

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(49121838)
% Create a Fitting Network
hiddenLayerSize = [5 5 5 5];
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
net.trainParam.min_grad = 0;
% net.trainParam.min_grad = 0;
EW = Luu/max(Luu);
% Train the Network
tic
[net,tr] = train(net,x,t,[],[],EW');
T1=toc;
%
[J1,U_n1,X_n1] = cellfun(@(x0)mysim3(x0,sys,par,net),X0Test,'UniformOutput',false);
mean(cell2mat(J1))
%%

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(49121838) 
% Create a Fitting Network
hiddenLayerSize = [5 5 5 5];
net = fitnet(hiddenLayerSize,trainFcn);
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
net.trainParam.min_grad = 0;
% Train the Network
tic
[net,tr] = train(net,x,t);
T2=toc;
%
[J2,U_n2,X_n2] = cellfun(@(x0,uopt)mysim3(x0,sys,par,net),X0Test,'UniformOutput',false);
mean(cell2mat(J2))
%%
figure(11)
semilogy(1:length(X0Test),abs(cell2mat(J1)-(fTest)),1:length(X0Test),abs(cell2mat(J2)-(fTest)),'LineWidth',2)
figure(12)
% plot(X_opt{2}(1,:),X_opt{2}(2,:),'b-')
% plot(X_n{1}(1,:),X_n{1}(2,:),'g--')
ind=[1,3,91,100];
ind=[3,23,27,34];
hold on
for k= 1:4
    i=ind(k);
    plot(X_optTest{i}(1,:),X_optTest{i}(2,:),'b-','LineWidth',2)
    % plot(X_n0{i}(1,:),X_n0{i}(2,:),'g--','LineWidth',2)
    plot(X_n1{i}(1,:),X_n1{i}(2,:),'r-.','LineWidth',2)
    plot(X_n2{i}(1,:),X_n2{i}(2,:),'c:','LineWidth',2)
end
plot(0.263193758468718,0.651883659521251,'o')
hold off
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
legend('$\pi_{mpc}(x)$','$\pi_{approx}(x,\theta_1)$','$\pi_{approx}(x,\theta_2)$',...
    'Interpreter','latex')
%%
 neurons_per_layer = [par.nx 5 5 5 5 par.nu];
setdemorandstream(49121838) 
net_my = createNN2(neurons_per_layer,par);
miniBatchSize = 112;
numEpochs = 2500;

% X = cell2mat(X_opt);
% U = cell2mat(U_opt);
X = cell2mat(cellfun(@(x)x(:,1),X_opt,'UniformOutput',false));
U = cell2mat(cellfun(@(x)x(:,1),U_opt,'UniformOutput',false));
par.nu =1; %par.H = 0;


% index = randperm(length(sol),400);
% indTr = index(1:300);indTest = index(301:end);
% index = randperm(length(sol),400);
% indTr = [1:10:1113 3:10:1113];indTest = 1:10:1113;
% indTr = setdiff(1:1113,indTest);
%
import casadi.*
Lar = par.nlp.f+dual_x'*par.nlp.x+dual_g'*par.nlp.g;
Lar_F = Function('Lar',{par.nlp.x,par.nlp.p,dual_g,dual_x},{Lar});
%
tic
[theta2,Losses2,Grades2]=trainNN_mse(net_my,Lar_F,sol,par,[],miniBatchSize,numEpochs);
T2 = toc;
%%
tic
[theta1,Losses1,Grades1]=trainNN_Wmse(net_my,sol,mat2cell(Luu./max(Luu),ones(1,length(sol))),[],miniBatchSize,numEpochs);
T1 = toc;
mean( cellfun(@(sol)full((sol.x(3)-net_my.nn(sol.x(1:2),theta1)).^2),sol));
%%
tic
[theta0,Losses0,Grades0]=trainNN_Lar(net_my,Lar_F,sol,par,mpcSolver.get_function('nlp_g'),theta1,miniBatchSize/2,numEpochs);
T0 = toc;
%%
% tic
% [theta2,Losses2,Grades2]=trainNN_mse(net_my,Lar_F,sol,par,theta2,miniBatchSize,numEpochs);
% T2 = toc;
%%
theta2=full(theta2); theta1=full(theta1); theta0=full(theta0);
save WRnet theta2 theta1 theta0 net_my T2 T1 T0
%%
[J2,U_n2,X_n2,Time2] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta2),X0Test,'UniformOutput',false);
%
[J1,U_n1,X_n1,Time1] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta1),X0Test,'UniformOutput',false);
%
[J0,U_n0,X_n0,Time0] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta0),X0Test,'UniformOutput',false);
%%
[J,U_n,X_n,Time]=cellfun(@(x0)mysim2(x0,sys,par,mpcSolver),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
%%
figure(11)
semilogy(1:length(solTest),cell2mat(J0)-(fTest),1:length(solTest),cell2mat(J2)-(fTest),1:length(solTest),cell2mat(J1)-(fTest),'LineWidth',2)
legend('$\pi_{approx}(x,\theta_0)$','$\pi_{approx}(x,\theta_1)$','$\pi_{approx}(x,\theta_2)$',...
     'Location','best','Interpreter','latex')
ylabel('Closed-loop Loss','Interpreter','latex')
xlabel('No','Interpreter','latex')

%%
meanJ(1)=mean(cell2mat(J0)-(fTest));
meanJ(2)=mean(cell2mat(J1)-(fTest));
meanJ(3)=mean(cell2mat(J2)-(fTest));
% meanJ(4)=mean(cell2mat(J)-(fTest)');
maxJ(1)=max(cell2mat(J0)-(fTest));
maxJ(2)=max(cell2mat(J1)-(fTest));
maxJ(3)=max(cell2mat(J2)-(fTest));
% maxJ(4)=max(cell2mat(J)-(fTest)');
meanTime(1)=mean(cell2mat(Time0),'all');
meanTime(2)=mean(cell2mat(Time1),'all');
meanTime(3)=mean(cell2mat(Time2),'all');
% meanTime(4)=mean(cell2mat(Time));
maxTime(1)=max(cell2mat(Time0),[],'all');
maxTime(2)=max(cell2mat(Time1),[],'all');
maxTime(3)=max(cell2mat(Time2),[],'all');
% maxTime(4)=max(cell2mat(Time));
%%
figure(12)
% plot(X_opt{2}(1,:),X_opt{2}(2,:),'b-')
% plot(X_n{1}(1,:),X_n{1}(2,:),'g--')
ind=[1,3,91,100];
ind=[1,23,34,11];
% ind=1:34;
hold on
for k= 1:4
    i=ind(k);
    % plot(X_opt{indTest(i)}(1,:),X_opt{indTest(i)}(2,:),'b-','LineWidth',2)
    plot(X_optTest{i}(1,:),X_optTest{i}(2,:),'b-','LineWidth',2)
    plot(X_n0{i}(1,:),X_n0{i}(2,:),'g--','LineWidth',2)
    plot(X_n2{i}(1,:),X_n2{i}(2,:),'r-.','LineWidth',2)
    plot(X_n1{i}(1,:),X_n1{i}(2,:),'c:','LineWidth',2)
    % pause(1)
end
plot(0.263193758468718,0.651883659521251,'o')
hold off
xlabel('$x_1$','Interpreter','latex')
ylabel('$x_2$','Interpreter','latex')
legend('$\pi_{mpc}(x)$','$\pi_{approx}(x,\theta_0)$','$\pi_{approx}(x,\theta_1)$','$\pi_{approx}(x,\theta_2)$',...
    'Interpreter','latex','FontSize',14)
% %%  Test sample index
% numEpochs = 5000;
% rng default
% indTest = randperm(length(sol),113);
% index = setdiff(1:1113,indTest);
% %% 10%
% indTr = index(1:10:end);numTr = length(indTr);
% tic
% [theta2,Losses2,Grades2]=trainNN_mse(net_my,Lar_F,sol(indTr),par,[],miniBatchSize,numEpochs);
% T2(1) = toc;
% %
% tic
% [theta1,Losses1,Grades1]=trainNN_Wmse(net_my,sol(indTr),mat2cell(Luu(indTr),ones(1,numTr)),[],miniBatchSize,numEpochs);
% T1(1) = toc;
% %
% tic
% [theta0,Losses0,Grades0]=trainNN_Lar(net_my,Lar_F,sol(indTr),par,theta1,miniBatchSize,numEpochs/2);
% T0(1) = toc;
% %
% [J2_10,U_n2_10,X_n2_10,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta2),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J1_10,U_n1_10,X_n1_10,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta1),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J0_10,U_n0_10,X_n0_10,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta0),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %% 20%
% indTr = index(1:5:end);numTr = length(indTr);
% tic
% [theta2,Losses2,Grades2]=trainNN_mse(net_my,Lar_F,sol(indTr),par,[],miniBatchSize,numEpochs);
% T2(2) = toc;
% %
% tic
% [theta1,Losses1,Grades1]=trainNN_Wmse(net_my,sol(indTr),mat2cell(Luu(indTr),ones(1,numTr)),[],miniBatchSize,numEpochs);
% T1(2) = toc;
% %
% tic
% [theta0,Losses0,Grades0]=trainNN_Lar(net_my,Lar_F,sol(indTr),par,theta1,miniBatchSize,numEpochs);
% T0(2) = toc;
% %
% [J2_20,U_n2_20,X_n2_20,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta2),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J1_20,U_n1_20,X_n1_20,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta1),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J0_20,U_n0_20,X_n0_20,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta0),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %% 50%
% indTr = index(1:2:end);numTr = length(indTr);
% tic
% [theta2,Losses2,Grades2]=trainNN_mse(net_my,Lar_F,sol(indTr),par,[],miniBatchSize,numEpochs);
% T2(3) = toc;
% %
% tic
% [theta1,Losses1,Grades1]=trainNN_Wmse(net_my,sol(indTr),mat2cell(Luu(indTr),ones(1,numTr)),[],miniBatchSize,numEpochs);
% T1(3) = toc;
% %
% tic
% [theta0,Losses0,Grades0]=trainNN_Lar(net_my,Lar_F,sol(indTr),par,theta1,miniBatchSize,numEpochs);
% T0(3) = toc;
% %
% [J2_50,U_n2_50,X_n2_50,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta2),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J1_50,U_n1_50,X_n1_50,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta1),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J0_50,U_n0_50,X_n0_50,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta0),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %% 100%
% indTr = index(1:end);numTr = length(indTr);
% tic
% [theta2,Losses2,Grades2]=trainNN_mse(net_my,Lar_F,sol(indTr),par,[],miniBatchSize,numEpochs);
% T2(4) = toc;
% %
% tic
% [theta1,Losses1,Grades1]=trainNN_Wmse(net_my,sol(indTr),mat2cell(Luu(indTr),ones(1,numTr)),[],miniBatchSize,numEpochs);
% T1(4) = toc;
% %
% tic
% [theta0,Losses0,Grades0]=trainNN_Lar(net_my,Lar_F,sol(indTr),par,theta1,miniBatchSize,numEpochs);
% T0(4) = toc;
% %
% [J2_100,U_n2_100,X_n2_100,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta2),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J1_100,U_n1_100,X_n1_100,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta1),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %
% [J0_100,U_n0_100,X_n0_100,~] = cellfun(@(x0)mysim(x0,sys,par,net_my,theta0),cellfun(@(x)x(:,1),X_opt(indTest),'UniformOutput',false),'UniformOutput',false);
% %%
% meanJ(1,1)=mean(cell2mat(J0_10)-(f(indTest))');
% meanJ(1,2)=mean(cell2mat(J1_10)-(f(indTest))');
% meanJ(1,3)=mean(cell2mat(J2_10)-(f(indTest))');
% maxJ(1,1)=max(cell2mat(J0_10)-(f(indTest))');
% maxJ(1,2)=max(cell2mat(J1_10)-(f(indTest))');
% maxJ(1,3)=max(cell2mat(J2_10)-(f(indTest))');
% %
% meanJ(2,1)=mean(cell2mat(J0_20)-(f(indTest))');
% meanJ(2,2)=mean(cell2mat(J1_20)-(f(indTest))');
% meanJ(2,3)=mean(cell2mat(J2_20)-(f(indTest))');
% maxJ(2,1)=max(cell2mat(J0_20)-(f(indTest))');
% maxJ(2,2)=max(cell2mat(J1_20)-(f(indTest))');
% maxJ(2,3)=max(cell2mat(J2_20)-(f(indTest))');
% %
% meanJ(3,1)=mean(cell2mat(J0_50)-(f(indTest))');
% meanJ(3,2)=mean(cell2mat(J1_50)-(f(indTest))');
% meanJ(3,3)=mean(cell2mat(J2_50)-(f(indTest))');
% maxJ(3,1)=max(cell2mat(J0_50)-(f(indTest))');
% maxJ(3,2)=max(cell2mat(J1_50)-(f(indTest))');
% maxJ(3,3)=max(cell2mat(J2_50)-(f(indTest))');
% %
% meanJ(4,1)=mean(cell2mat(J0_100)-(f(indTest))');
% meanJ(4,2)=mean(cell2mat(J1_100)-(f(indTest))');
% meanJ(4,3)=mean(cell2mat(J2_100)-(f(indTest))');
% maxJ(4,1)=max(cell2mat(J0_100)-(f(indTest))');
% maxJ(4,2)=max(cell2mat(J1_100)-(f(indTest))');
% maxJ(4,3)=max(cell2mat(J2_100)-(f(indTest))');
%%
function [sol,flag,f,T]=mySolver(Solver,w0,x0,u_i,d_v,par)
tic
sol = Solver('x0',w0,'p',vertcat(x0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
flag = Solver.stats().success;
f= full(sol.f);
T=toc;
end
function  [x_opt,u_opt]=getOpt(sol,coor_U,coor_X)
Primal = full(sol.x);
u_opt = Primal(coor_U)';
x_opt = Primal(coor_X);
x_opt = x_opt(:,1:3:end-1);
x_opt(:,abs(u_opt-0.758340018600918)<1e-6)=[];
u_opt(:,abs(u_opt-0.758340018600918)<1e-6)=[];
end

function [J,U_n,X_n,Time]=mysim(X,sys,par,net,theta)
import casadi.*

    i=1;
    J=0;
    xk = X(1:2);uk = 0;    
    while    i<=100 %&& abs(uk -0.758340018600918)>1e-5
        X_n(:,i)=xk;
        tic
        uk = net.nn(xk,theta);
        uk = full(uk);
        uk = min(max(full(uk),par.lbu),par.ubu);
        Time(i) = toc;
        U_n(:,i) = uk;
        Fk = sys.F('x0',xk,'p',vertcat(uk,par.d0));
        xk =  full(Fk.xf) ;
        
        J = J+full(Fk.qf);
        i=i+1;
    end

end

function [J,U_n,X_n,Time]=mysim2(X,sys,par,mpcSolver)
import casadi.*

    i=1;
    J=0;
    xk = X(1:2);uk = 0;    
    while    i<=100 %&& abs(uk -0.758340018600918)>1e-5
        X_n(:,i)=xk;
        [sol,~,~,Time(i)]=mySolver(mpcSolver,par.w0,xk,par.u0,par.d0,par);
        
        uk = full(sol.x(par.nx+1:par.nx+par.nu));
        uk = min(max(full(uk),par.lbu),par.ubu);

        U_n(:,i) = uk;
        Fk = sys.F('x0',xk,'p',vertcat(uk,par.d0));
        xk =  full(Fk.xf) ;
        
        J = J+full(Fk.qf);
        i=i+1;
    end

end

function [J,U_n,X_n,Time]=mysim3(X,sys,par,net)
import casadi.*

    i=1;
    J=0;
    xk = X(1:2)';uk = 0;    
    while    i<=100 %&& abs(uk -0.758340018600918)>1e-5
        X_n(:,i)=xk;
        tic
        uk = net(xk);
        % uk = full(uk);
        uk = min(max(full(uk),par.lbu),par.ubu);
        Time(i) = toc;
        U_n(:,i) = uk;
        Fk = sys.F('x0',xk,'p',vertcat(uk,par.d0));
        xk =  full(Fk.xf) ;
        
        J = J+full(Fk.qf);
        i=i+1;
    end

end