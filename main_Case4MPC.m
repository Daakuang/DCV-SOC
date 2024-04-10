%%
addpath('model/')
%%
% A optimal control problem
clear;
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
%
% par.u0 = 3; par.ubu=5;par.lbu=-1;%par.x0=[X10(i);X20(i)];
% [mpcSolver1,par] = buildNLP(sys.f,par,sys.F);

% U_opt=cell(1,numel(X10));X_opt=cell(1,numel(X10));isSuccess=false(1,numel(X10));J=zeros(1,numel(X10));

x_0 = [0.263193758468718;0.651883659521251]+[0.1;0.1];u_i=par.u0;d_v=par.d0;

sol = mpcSolver('x0',par.w0,'p',vertcat(x_0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
% flag = mpcSolver.stats();
exitflag =  mpcSolver.stats().return_status
% isSuccess = flag.success
sol.f
Primal = full(sol.x);
u_opt = Primal(coor_U);
x_opt = Primal(coor_X);
% end

figure(1)
plot((0:par.N-1)*par.tf,u_opt,'LineWidth',2)
figure(2)
plot((0:par.N)*par.tf,x_opt(2,1:3:end),'LineWidth',2)
% comet(x_opt(1,:),x_opt(2,:))

%% mode1
[X10,X20] = meshgrid(linspace(par.lbx(1),par.ubx(1),40),linspace(par.lbx(2),par.ubx(2),40));
X0 = mat2cell([X10(:),X20(:)],ones(1,numel(X10)));
% [mpcSolver,par] = buildNLP(sys.f,par,sys.F);
% U_opt=cell(1,numel(X10));X_opt=cell(1,numel(X10));J=zeros(1,numel(X10));

% sol = mpcSolver('x0',par.w0,'p',vertcat(x_0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
[sol,flag,f]=cellfun(@(x)mySolver(mpcSolver,par.w0,x',u_i,d_v,par),X0,'UniformOutput',false);

%%
f=cell2mat(f);flag=cell2mat(flag);sol(~flag)=[];X0(~flag)=[];f(~flag)=[];%flag(~flag)=[];
%%
[X_opt,U_opt] = cellfun(@(sol)getOpt(sol,coor_U,coor_X),sol','UniformOutput',false);
save case4 U_opt  X_opt  f flag
% X_opt(shibai)=[];
% U_opt(shibai)=[];
% J(shibai)=[];
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
miniBatchSize = 111*5;
numEpochs = 2000;

% X = cell2mat(X_opt);
% U = cell2mat(U_opt);
X = cell2mat(cellfun(@(x)x(:,1:5),X_opt(1:10:end),'UniformOutput',false));
U = cell2mat(cellfun(@(x)x(:,1:5),U_opt(1:10:end),'UniformOutput',false));
par.nu =1; %par.H = 0;

net = createNN([par.nu+par.nx 10 10 par.nu],par); 
% net = createNN_maxout([7 10 10 par.nu],2,par);
tic
theta1=trainNNCV_minibatch(net,sys,par,U,X,[],miniBatchSize,numEpochs);
% theta1=trainNNCV_minibatch(net,sys,par,U,X,theta1,miniBatchSize,numEpochs);
toc
save Case4net01 theta1 net
%%
 % abs(full(net.nn(U,X,theta1)));
[J,U_n,X_n] = cellfun(@(x0,uopt)mysim(x0,sys,par,net,theta1,uopt),cellfun(@(x)x(:,1),X_opt(1:10:end),'UniformOutput',false),U_opt(1:10:end),'UniformOutput',false);
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
figure(13)
plot(cell2mat(cellfun(@(x)x(:,end),X_n,'UniformOutput',false))')
figure(14)
plot(cell2mat(J)-f')
mean(cell2mat(J))
mean(f)
max(cell2mat(J))
% figure(11)
% plot((0:length(U_opt{1})-1)*par.tf,U_opt{1})
% plot(U_n{1}(end),'b.')
% hold on
% figure(12)
% % plot(X_n{1}(1,:),X_n{1}(2,:),'b.')
% hold on
% plot(0.263193758468718,0.651883659521251,'o')
% % hold off
% for i = 1:numel(X10)
%         % if abs(U_n{i}(1,end)-0.758340018600918)<1e-2
%             % figure(11)
%             % plot((0:length(U_n{i})-1)*par.tf,U_n{i})
%             % plot(U_n{i}(end),'b.')
%             figure(12)
%             % plot(X_n{i}(1,:),X_n{i}(2,:),'b.')
%             plot(X_n{i}(1,end),X_n{i}(2,end),'ro')
%             % hold on
%             % plot(0.263193758468718,0.651883659521251,'o')
%             % hold off
%             % i
%         % end
% end
% figure(12)
% hold off
% figure(11)
% hold off

%%
function [sol,flag,f]=mySolver(Solver,w0,x0,u_i,d_v,par)
sol = Solver('x0',w0,'p',vertcat(x0,u_i,d_v),'lbx',par.lbw,'ubx',par.ubw,'lbg',par.lbg,'ubg',par.ubg);
flag = Solver.stats().success;
f= full(sol.f);
end
function  [x_opt,u_opt]=getOpt(sol,coor_U,coor_X)
Primal = full(sol.x);
u_opt = Primal(coor_U)';
x_opt = Primal(coor_X);
x_opt = x_opt(:,1:3:end-1);
x_opt(:,abs(u_opt-0.758340018600918)<1e-6)=[];
u_opt(:,abs(u_opt-0.758340018600918)<1e-6)=[];
end

function [J,U_n,X_n]=mysim(X,sys,par,net,theta,U)
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
    while    i<=100 %&& abs(uk -0.758340018600918)>1e-5
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
        %     sol = F('x0',U(:,i),'ubx',par.ubu,'lbx',par.lbu);
        % catch
        %     sol = F('x0',0.758340018600918,'ubx',par.ubu,'lbx',par.lbu);
        % end
        % uk = sol.x;
        
        % opt = struct('abstol',1e-3);
        CVf = Function('cv',{uk_s},{CV});
        CV_uk = rootfinder('CV_uk','newton',CVf);%kinsol newton
        % uk=cellfun(@(x)full(CV_uk(x)),num2cell(0:0.2:2));
        % try
        %     [~,ind]=min(abs(uk-U(i)));
        % catch
        %     [~,ind]=min(abs(uk-0.758340018600918));
        % end
        % uk = uk(ind);
        try
            uk = CV_uk(U(i));
        catch
            uk = CV_uk(0.758340018600918);
        end
        uk = full(uk);

        % uk = uk(1:par.nu);
        % ci(i) = full(sol.f);
        uk = min(max(full(uk),par.lbu),par.ubu);

        % uk = min(max(result.Network(xk),par.lbu),par.ubu);
        
        % a=  U(i)-uk;
        U_n(:,i) = uk;
        [fk,lk]=Fk(xk,[uk;par.d0;],0,0,0,0);
        xk = full(fk);
        X_n(:,i)=xk;
        J = J+full(lk);
        i=i+1;
    end
    
    % plot(U_n)
    
% end
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
    while    i<=100 %&& abs(uk -0.758340018600918)>1e-5
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

function [J,U_n,X_n]=mysim3(X,sys,par,net)
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
    while    i<=100 %&& abs(uk -0.758340018600918)>1e-5
        X=[xk];
        CV = @(uk_s)net([X;uk_s]);
        try
            uk = fsolve(CV,U(i),opt);
        catch
            uk = fsolve(CV,0.75834001860091,opt);
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