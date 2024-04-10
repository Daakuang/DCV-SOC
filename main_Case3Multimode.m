% A multimodal optimal control problem
clear;clc
addpath model\
%%
par.tf = 0.05;
[sys,par] = nonLSTR(par);
par.N = 150;
% par.d0 = [0.9;0.9];

u_init = par.u0;
d_init = par.d0;
lbx = par.lbx;
ubx = par.ubx;
par.ROC = [0];
par.u0 = 1;
[~,par] = buildNLP(sys.f,par,sys.F);
%%
n_w_i = par.nx + par.N*((par.degree+1)*par.nx+par.nu);
coor_U=(par.nx+1:(par.degree+1)*par.nx+par.nu:n_w_i).*ones(par.nu,1)+(0:par.nu-1)';
% ind_X=sort(coor_X(:));
ind_U = sort(coor_U(:));
ind_notU = setdiff(1:length(par.nlp.x),ind_U(:));
coor_X = reshape(ind_notU,par.nx,[]);
coor_X = reshape(setdiff(coor_X,coor_X(:,(1+par.degree+1):par.degree+1:end)),par.nx,[]);
%%
par.u0 = 3; par.ubu=5;par.lbu=-1;%par.x0=[X10(i);X20(i)];
[mpcSolver1,par] = buildNLP(sys.f,par,sys.F);

% U_opt=cell(1,numel(X10));X_opt=cell(1,numel(X10));isSuccess=false(1,numel(X10));J=zeros(1,numel(X10));

x_0 = [0.09;0.09;];u_i=1;d_v=[25;0.5];

sol = mpcSolver1('x0',par.w0,'p',vertcat(x_0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
% flag = mpcSolver.stats();
exitflag =  mpcSolver1.stats().return_status
% isSuccess = flag.success
sol.f
Primal = full(sol.x);
u_opt = Primal(coor_U);
x_opt = Primal(coor_X);
% end

figure(1)
plot((0:par.N-1)*par.tf,u_opt,'LineWidth',2)
hold on
figure(2)
comet(x_opt(1,:),x_opt(2,:))
hold on
%%
par.u0 = 1; par.ubu=5;par.lbu=-1;
[mpcSolver,par] = buildNLP(sys.f,par,sys.F);
x_0 = [0.09;0.09;];u_i=1;d_v=[25;0.5];
sol = mpcSolver('x0',par.w0,'p',vertcat(x_0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
% flag = mpcSolver.stats();
exitflag =  mpcSolver.stats().return_status

Primal = full(sol.x);
u_opt = Primal(coor_U);
x_opt = Primal(coor_X);

sol.f
figure(1)
plot((0:par.N-1)*par.tf,u_opt,'LineWidth',2)
legend("mode 1","mode 2",'latex','FontSize',13)
ylabel('$u(t)$','Interpreter','latex','FontSize',13)
xlabel('Time','Interpreter','latex','FontSize',13)
hold off
figure(2)
plot(x_opt(1,:),x_opt(2,:),'.')
hold off
%% mode1
[X10,X20] = meshgrid(linspace(0.09,-0.09,40),linspace(0.09,-0.09,40));

% [mpcSolver,par] = buildNLP(sys.f,par,sys.F);
U_opt=cell(1,numel(X10));X_opt=cell(1,numel(X10));J=zeros(1,numel(X10));
U_subopt=cell(1,numel(X10));X_subopt=cell(1,numel(X10));

x_0 = [0.9;0.9;];u_i=1;d_v=[25;0.5];
clear w0
i=1;j=0; %w0 = par.w0;
for i =-3:0.5:5
    j=j+1;
    par.u0 =i;
    [mpcSolver,par] = buildNLP(sys.f,par,sys.F);
    w0(j) = {par.w0};
end

minf=1e9*ones(1,numel(X10));subminf=1e9*ones(1,numel(X10));
for i=1:numel(X10)
    % x_0 = [X10(i);X20(i)];
    [sol,flag,f]=cellfun(@(x)mySolver(mpcSolver,x,[X10(i);X20(i)],u_i,d_v,par),w0,'UniformOutput',false);
    f = cell2mat(f);
    flag = cell2mat(flag);
    if sum(flag)>0
        %golbal optimal solution
        [minf(i),indOpt]=min(f);

        Primal = full(sol{indOpt}.x);
        u_opt = Primal(coor_U)';
        x_opt = Primal(coor_X);
        x_opt(:,abs(u_opt)<1e-3)=[];
        u_opt(:,abs(u_opt)<1e-3)=[];
        U_opt{i} = u_opt;
        X_opt{i} = x_opt;
    
        %local optimal solution suboptimal
        if sum((f>minf(i)+1e-3)&flag)>0
            subminf(i)=min(f((f>minf(i)+1e-3)&flag));
            [~,indSubopt] = max(subminf(i)==f);
            Primal = full(sol{indSubopt}.x);
            u_opt = Primal(coor_U)';
            x_opt = Primal(coor_X);
            x_opt(:,abs(u_opt)<1e-3)=[];
            u_opt(:,abs(u_opt)<1e-3)=[];
            U_subopt{i} = u_opt;
            X_subopt{i} = x_opt;
        end
    end
    i
end
save case3 U_opt U_subopt X_opt X_subopt minf subminf
% X_opt(shibai)=[];
% U_opt(shibai)=[];
% J(shibai)=[];
%%
figure(1)
plot((0:length(U_opt{1})-1)*par.tf,U_opt{1})
hold on
figure(2)
plot(X_opt{1}(1,:),X_opt{1}(2,:),'.')
hold on
for i =2:numel(X10)
    try
    figure(1)
    plot((0:length(U_opt{i})-1)*par.tf,U_opt{i})
    figure(2)
    plot(X_opt{i}(1,:),X_opt{i}(2,:),'.')
    catch
        continue
    end
end
figure(2)
hold off
figure(1)
hold off
%%
a = cellfun(@isempty, X_subopt);sum(a)
subminf(subminf==1e9)=nan;
plot(1:numel(X10),subminf,1:numel(X10),minf)

X_opt= cellfun(@(x)x(:,1:end-1), X_opt,'UniformOutput',false);
X_subopt= cellfun(@(x)x(:,1:end-1), X_subopt,'UniformOutput',false);
%%
miniBatchSize = 5000;
numEpochs = 500;
%
% X = cell2mat([X_opt,X_subopt(~a)]);
% U = cell2mat([U_opt,U_subopt(~a)]);
X = cell2mat(X_opt);
U = cell2mat(U_opt);
% X = [X10(:)';X20(:)'];
par.nu =1; %par.H = 0;
% X = cellfun(@(x)reshape(x(:,1:par.H+1),[],1),X_opt,'UniformOutput',false);
% U = cellfun(@(x)reshape(x(:,1:par.H),[],1),U_opt,'UniformOutput',false);
% net = createNN([par.nu*par.H+par.nx*(par.H+1) 10 10 par.nu*par.H],par); 
% X = cellfun(@(x)reshape(x(:,1),[],1),X_opt,'UniformOutput',false);
% U = cellfun(@(x)reshape(x(:,1),[],1),U_opt,'UniformOutput',false);
% U =cell2mat(U);X =cell2mat(X);

net = createNN([par.nu+par.nx 10 10 10 10 10 par.nu],par); 
% net = createNN_maxout([7 10 10 par.nu],2,par);
tic
theta1=trainNNCV_minibatch(net,sys,par,U,X,[],miniBatchSize,numEpochs);
% theta1=trainNNCV_minibatch(net,sys,par,U,X,theta1,miniBatchSize,numEpochs);
toc
%%
%%
X = cell2mat(cellfun(@(x)x(:,1:5),X_opt(1:10:end),'UniformOutput',false));
U = cell2mat(cellfun(@(x)x(:,1:5),U_opt(1:10:end),'UniformOutput',false));

x = X;
t = U;

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(491218382)
% Create a Fitting Network
hiddenLayerSize = [10 10];
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Train the Network
[net,tr] = train(net,x,t);
%
[J1,U_n1,X_n1] = cellfun(@(x0,uopt)mysim2(x0,sys,par,net),cellfun(@(x)x(:,1),X_opt(2:10:end),'UniformOutput',false),U_opt(2:10:end),'UniformOutput',false);
%%
Nc =10;
x = repmat(X,1,Nc+1);
u = [U,kron(linspace(0,2,Nc),ones(1,length(U)))];
x = [x;u];
t = u - repmat(U,1,Nc+1);

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(491218382)
% Create a Fitting Network
hiddenLayerSize = [10 10];
net = fitnet(hiddenLayerSize,trainFcn);
net.trainParam.epochs=10000;
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Train the Network
[net,tr] = train(net,x,t);
%
[J2,U_n2,X_n2] = cellfun(@(x0,uopt)mysim3(x0,sys,par,net),cellfun(@(x)x(:,1),X_opt(2:10:end),'UniformOutput',false),U_opt(2:10:end),'UniformOutput',false);
%%
semilogy(1:160,cell2mat(J2)-minf(2:10:end),1:160,cell2mat(J1)-minf(2:10:end))
legend("DCV Implicit policy","Explicit policy",'latex','FontSize',13)
ylabel('$\Delta \mathcal{J}$','Interpreter','latex','FontSize',13)
xlabel('No','Interpreter','latex','FontSize',13)
%%
plot(U_opt{1, 2})
hold on
plot(U_n2{1})
hold off
%%
a = cellfun(@isempty, X_subopt);
b = 1:length(X_opt);
b(a) = [];
rng(45613)
indSubopt=b(randperm(length(b),500));
indOpt= setdiff(1:length(X_opt),indSubopt);
%%
X = [cell2mat(cellfun(@(x)x(:,1:5),X_opt(indOpt),'UniformOutput',false)),cell2mat(cellfun(@(x)x(:,1:5),X_subopt(indSubopt),'UniformOutput',false))];
U = [cell2mat(cellfun(@(x)x(:,1:5),U_opt(indOpt),'UniformOutput',false)),cell2mat(cellfun(@(x)x(:,1:5),U_subopt(indSubopt),'UniformOutput',false))];

x = X;
t = U;

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(491218382)
% Create a Fitting Network
hiddenLayerSize = [10 10];
net = fitnet(hiddenLayerSize,trainFcn);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;
net.trainParam.epochs=10000;
% Train the Network
[net,tr] = train(net,x,t);
%
[J1,U_n1,X_n1] = cellfun(@(x0,uopt)mysim2(x0,sys,par,net),cellfun(@(x)x(:,1),X_opt(2:10:end),'UniformOutput',false),U_opt(2:10:end),'UniformOutput',false);
%%
Nc =2;
x = repmat(X,1,Nc+1);
u = [U,U+0.01,U-0.01];
x = [x;u];
t = u - repmat(U,1,Nc+1);

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(491218382)
% Create a Fitting Network
hiddenLayerSize = [10 10];
net = fitnet(hiddenLayerSize,trainFcn);
net.trainParam.epochs=10000;
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Train the Network
[net,tr] = train(net,x,t);
%
[J2,U_n2,X_n2] = cellfun(@(x0,uopt)mysim3(x0,sys,par,net,uopt),cellfun(@(x)x(:,1),X_opt(2:10:end),'UniformOutput',false),U_opt(2:10:end),'UniformOutput',false);
%%
%%
function [sol,flag,f]=mySolver(Solver,w0,x0,u_i,d_v,par)
sol = Solver('x0',w0,'p',vertcat(x0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
flag = Solver.stats().success;
f= full(sol.f);
end

function [J,U_n,X_n]=mysim(X10,X20,sys,par,net,theta,U_opt)
import casadi.*
try
    uk_s = MX.sym('uk',par.nu*par.H);
catch
    uk_s = MX.sym('uk',par.nu);
    par.H=0;
end
nlp = struct;nlp.x = uk_s; nlp.g = []; opts = struct('print_time',false, 'ipopt',struct('print_level',0,'max_iter',10));

J = zeros(1,numel(X10));U_n = cell(1,numel(X10));X_n = cell(1,numel(X10));
Fk = sys.F;
for j=501:501%numel(X10)

    i=1;

    xk = [X10(j);X20(j)];
    while i<=par.N
        X=[xk];
        fk = xk;
        for k =1:par.H
            fk=Fk(fk,[par.d0;uk_s((k-1)*par.nu+1:k*par.nu)],0,0,0,0);
            X = [X;fk];
        end
        CV = net.nn(uk_s,X,theta);
        % nlp.f = CV.^2;
        % F = nlpsol('F','ipopt',nlp,opts);
        % try
        %     sol = F('x0',U_opt{j}(:,i),'ubx',par.ubu,'lbx',par.lbu);
        % catch
        %     sol = F('x0',0,'ubx',par.ubu,'lbx',par.lbu);
        % end
        % uk = sol.x;
        
        opt = struct('abstol',1e-3);
        CVf = Function('cv',{uk_s},{CV});
        CV_uk = rootfinder('CV_uk','newton',CVf,opt);%kinsol
        try
            uk = CV_uk(U_opt{j}(:,i));
        catch
            uk = CV_uk(0);
        end
        uk = full(uk);

        uk = uk(1:par.nu);
  
        uk = min(max(full(uk),par.lbu),par.ubu);
        % uk = min(max(result.Network(xk),par.lbu),par.ubu);
        
        
        % ci(j,i) = full(CVf(uk));
        % uk-U_opt{j}(:,i)
        % if i ==1
        %     uk=U_opt{j}(:,i);
        % end

        U_n{j}(:,i) = uk;
        [fk,lk]=Fk(xk,[uk;par.d0;],0,0,0,0);
        xk = full(fk);
        X_n{j}(:,i)=xk;
        
        i=i+1;
    end
    J(j) = full(lk);
end
end

function [J,U_n,X_n]=mysim2(X,sys,par,net)
import casadi.*
try
    uk_s = MX.sym('uk',par.nu*par.H);
catch
    uk_s = MX.sym('uk',par.nu);
    par.H=0;
end
nlp = struct;nlp.x = uk_s; nlp.g = []; opts = struct('print_time',false, 'ipopt',struct('print_level',0,'max_iter',1000));

% J = zeros(1,numel(X10));U_n = cell(1,numel(X10));X_n = cell(1,numel(X10));
Fk = sys.F;


    i=1;
    J=0;
    xk = X(1:2);uk = 0;    
    while    i<=par.N*10 %&& abs(uk -0.758340018600918)>1e-5
        X=[xk];

        
        uk = net(X);
        uk = min(max(full(uk),par.lbu),par.ubu);

        U_n(:,i) = uk;
        [fk,lk]=Fk(xk,[uk;par.d0;],0,0,0,0);
        xk = full(fk);
        X_n(:,i)=xk;
        J = J+full(lk);
        i=i+1;
    end

end

function [J,U_n,X_n]=mysim3(X,sys,par,net,U)
import casadi.*
try
    uk_s = MX.sym('uk',par.nu*par.H);
catch
    uk_s = MX.sym('uk',par.nu);
    par.H=0;
end
nlp = struct;nlp.x = uk_s; nlp.g = []; opts = struct('print_time',false, 'ipopt',struct('print_level',0,'max_iter',1000));

% J = zeros(1,numel(X10));U_n = cell(1,numel(X10));X_n = cell(1,numel(X10));
Fk = sys.F;


    i=1;
    J=0;
    xk = X(1:2);uk = 0;    opt = optimoptions('fsolve','Display','off');
    while    i<=par.N %&& abs(uk -0.758340018600918)>1e-5
        X=[xk];
        CV = @(uk_s)net([X;uk_s]);
        try
            uk = fsolve(CV,U(i),opt);
        catch
            uk = fsolve(CV,0,opt);
        end
        % uk = full(uk);
        uk = min(max(full(uk),par.lbu),par.ubu);

        U_n(:,i) = uk;
        [fk,lk]=Fk(xk,[uk;par.d0;],0,0,0,0);
        xk = full(fk);
        X_n(:,i)=xk;
        J = J+full(lk);
        i=i+1;
    end

end
