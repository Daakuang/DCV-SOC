import casadi.*

addpath model\
%%
T=70;
c_A0=2;
c_B0=0.63;
c_C0=0;
V_0=0.7;
pho=900;
c_p=4.2;
detH=-60000;
n_Cdes=0.6;

par.ts = 1;
[sys,par] = ISBR(par);
par.N = 20;
par.d0 = 0.0483;
%%
d= par.d0.*linspace(0.8,1.2,10);
data_opt(length(d))=struct('tf',1,"y",1,'u',1,'d',1);
for i = 1:length(d)
[sol] = myOCP(sys,par,par.N,d(:,i));
topt(i)=full(sol.x(1));
end
%%

par.ts = 0.1;
[sys,par] = ISBR(par);
% d = d0.*(1+0.4*(rand(1,100)-0.5));
d= par.d0.*linspace(0.8,1.2,10);
data_opt(length(d))=struct('tf',1,"y",1,'u',1,'d',1);

Fk = sys.F;
for j=1:length(d)
    i=1;n_c=0;xk = par.x0;Y=[];U=[];dk = d(:,j);
    while n_c <= n_Cdes
        if xk(3) >= 1 %Vmax = 1
            uk = 0;
        else
            uk = d(1,j)*xk(1)*xk(2)*xk(3)/(2-xk(2));
        end
        U = [U,uk];
        

        c_C = (c_A0*V_0+c_C0*V_0-xk(1,:).*xk(3,:))./xk(3,:);
        T_cf = T+min(xk(1,:),xk(2,:))*-detH/pho/c_p;
        n_c = xk(3)*c_C;
        fk=Fk(xk,[par.d0;uk],0,0,0,0);
        xk_hat = full(fk);
        c_C_hat = (c_A0*V_0+c_C0*V_0-xk_hat(1,:).*xk_hat(3,:))./xk_hat(3,:);
        T_cf_hat = T+min(xk_hat(1,:),xk_hat(2,:))*-detH/pho/c_p;
        n_c_hat = xk_hat(3)*c_C_hat;
        % Y = [Y,[xk;c_C;T_cf;n_c;xk_hat;c_C_hat;T_cf_hat;n_c_hat]];
        Y = [Y,[xk;xk_hat]];

        fk=Fk(xk,[dk;uk],0,0,0,0);
        xk = full(fk);
        i=i+1;
    end
    data_opt(j).tf=i*par.ts;
    data_opt(j).y = Y;
    data_opt(j).u = U;
    data_opt(j).d = dk;
end
Y=[];U=[];
for j=1:length(d)
    Y = [Y,data_opt(j).y];
    U = [U,data_opt(j).u];
end

%%
nu =1;
net0 = createNN([7 10 10 10 par.nu],par);
% net = createNN_maxout([7 10 10 par.nu],2,par);
tic
% theta1=trainNNCV(net,sys,par,U,Y,[],2000);
miniBatchSize=300;numIterationsPerEpoch=300;
theta0=trainNNCV_minibatch(net,sys,par,U,Y,[],miniBatchSize,numIterationsPerEpoch);
%%
theta0=trainNNCV(net,sys,par,data_opt,theta0,200);
% theta1=trainNNCV(net,sys,par,data_opt,theta1,10000);
toc
%%
theta = theta0(1:end);L =50;theta0(1); 
figure(1)
plot(full(net.nn(data_opt(1).u,data_opt(1).y,theta)));
log(1/data_opt(1).tf*(full(net.nn(data_opt(1).u,data_opt(1).y,theta)))*(full(net.nn(data_opt(1).u,data_opt(1).y,theta)))')

%%
import casadi.*
uk_s = MX.sym('uk',par.nu);uki=[];k=10;
for i=1: data_opt(k).tf/par.ts-1
    c(i,:) = (full(net.nn([0:0.001:0.1],repmat(data_opt(1).y(:,i),1,101),theta))-[0:0.001:0.1]);
    CV = (net.nn(uk_s,data_opt(k).y(:,i),theta));
    CVf = Function('cv',{uk_s},{CV-0});
    CV_uk = rootfinder('CV_uk','newton',CVf);
    uk = CV_uk();
    uki(i) = full(uk.o0);
    % uki(i) = min(max(uki(i),0),0.1); 
    
end 
figure(2)
plot(1:length(uki),uki,1:length(uki),data_opt(k).u)
%%
import casadi.*
uk_s = MX.sym('uk',par.nu);
nlp = struct;nlp.x = uk_s; nlp.g = []; opts = struct('print_time',false, 'ipopt',struct('print_level',0,'max_iter',10));

Y_n=[];U_n=[];C=[];U_opt=[];
xk = par.x0;
Fk = sys.F;
for j=1:1%length(d)
    i=1;n_c=0;xk = par.x0;dk = d(:,j);uk=U(1);Y_n=[];U_n=[];C=[];U_opt=[];
    while n_c <= n_Cdes && i<=400

        c_C = (c_A0*V_0+c_C0*V_0-xk(1,:).*xk(3,:))./xk(3,:);
        T_cf = T+min(xk(1,:),xk(2,:))*-detH/pho/c_p;
        n_c = xk(3)*c_C;
        xk_hat=Fk(xk,[par.d0;uk_s],0,0,0,0);
        yk =  [xk;xk_hat];[xk;c_C;T_cf;n_c];
        Xi = yk;
        CV = net0.nn(uk_s,Xi,theta0);
        nlp.f = CV.^2;
        F = nlpsol('F','ipopt',nlp,opts);
        try
            sol = F('x0',data_opt(j).u(:,i),'ubx',par.ubu,'lbx',par.lbu);
        catch
            sol = F('x0',0,'ubx',par.ubu,'lbx',par.lbu);
        end
        uk = sol.x;
        
        
        CVf = Function('cv',{uk_s},{CV});
        % CV_uk = rootfinder('CV_uk','kinsol',CVf);%newton
        % try
        %     uk = CV_uk(data_opt(j).u(:,i));
        % catch
        %     uk = CV_uk(0);
        % end
        % uk = full(uk.o0);

        uk = min(max(full(uk),0),0.1);
        % fun = @(uk)cvFunction(yk,uk,xk,d0,b,Fk);
        % fun = @(uk)cvFunction(yk,uk,xk,d0,results_2,Fk);
        % ci = 0:0.001:0.1;
        % for m=1:101
        %     ci(m) = fun(ci(m));
        % end
        % C = [C;ci];

        % options = optimoptions('fsolve','OptimalityTolerance',1e-8);
        % uk = fsolve(fun,0.01,options);
        
        
        ci(i,j) = full(CVf(uk));

        U_n = [U_n,uk];
        if xk(3) >= 1 %Vmax = 1
            uu = 0;
        else
            uu = dk(1)*xk(1)*xk(2)*xk(3)/(2-xk(2));
        end
        U_opt = [U_opt,uu];

        fk=Fk(xk,[par.d0;uk],0,0,0,0);
        xk_hat = full(fk);
        c_C_hat = (c_A0*V_0+c_C0*V_0-xk_hat(1,:).*xk_hat(3,:))./xk_hat(3,:);
        T_cf_hat = T+min(xk_hat(1,:),xk_hat(2,:))*-detH/pho/c_p;
        n_c_hat = xk_hat(3)*c_C_hat;
        Y_n = [Y_n,[xk;c_C;T_cf;n_c]];%;xk_hat;c_C_hat;T_cf_hat;n_c_hat

        fk=Fk(xk,[dk;uk],0,0,0,0);
        xk = full(fk);
        i=i+1;
    end
    tf(j)=i;
    Tv(j) = min(min(80-Y_n(5,:),0));
    Vv(j) = min(min(1-Y_n(3,:),0));
end
MM(1,1)=mean(tf(tf<401));MM(1,3)=-mean(Tv);MM(1,5)=-mean(Vv);
MM(1,2)=mean(tf==401);MM(1,4)=-min(Tv);MM(1,6)=-min(Vv);
%%
figure(4)
subplot(4,1,1)
plot((1:tf(1)-1)*0.1,Y_n(4,:),'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),(c_A0*V_0+c_C0*V_0-data_opt(1).y(1,:).*data_opt(1).y(3,:))./data_opt(1).y(3,:),'--','LineWidth',1.5)
% hold off
subplot(4,1,2)
plot((1:tf(1)-1)*0.1,Y_n(5,:),'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),T+min(data_opt(1).y(1,:),data_opt(1).y(2,:))*-detH/pho/c_p,'--','LineWidth',1.5)

subplot(4,1,3)
plot((1:tf(1)-1)*0.1,Y_n(3,:),'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),data_opt(1).y(3,:),'--','LineWidth',1.5)
subplot(4,1,4)
plot((1:tf(1)-1)*0.1,U_n,'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),data_opt(1).u,'--','LineWidth',1.5)

% tt = [27;25;24;24;22;22;21;20;19;19]'-tf;
%% No pre
data_opt1=data_opt;
for i=1:10
    data_opt1(i).y = data_opt(i).y(1:3,:);
end
nu =1;
net = createNN([4 10 10 10 par.nu],par);
% net = createNN_maxout([7 10 10 par.nu],2,par);
tic
% theta1=trainNNCV(net,sys,par,U,Y,[],2000);
% miniBatchSize=300;numIterationsPerEpoch=50;
% theta1=trainNNCV_minibatch(net,sys,par,U,Y,[],miniBatchSize,numIterationsPerEpoch);
theta1=trainNNCV_7(net,sys,par,data_opt1,[],2000);
% theta1=trainNNCV(net,sys,par,data_opt,theta1,10000);
toc
%%
import casadi.*
uk_s = MX.sym('uk',par.nu);
nlp = struct;nlp.x = uk_s; nlp.g = []; opts = struct('print_time',false, 'ipopt',struct('print_level',0,'max_iter',10));

Y_n=[];U_n=[];C=[];U_opt=[];
xk = par.x0;
Fk = sys.F;
for j=1:1%length(d)
    i=1;n_c=0;xk = par.x0;dk = d(:,j);uk=U(1);Y_n=[];U_n=[];C=[];U_opt=[];
    while n_c <= n_Cdes && i<=400

        c_C = (c_A0*V_0+c_C0*V_0-xk(1,:).*xk(3,:))./xk(3,:);
        T_cf = T+min(xk(1,:),xk(2,:))*-detH/pho/c_p;
        n_c = xk(3)*c_C;
        xk_hat=Fk(xk,[par.d0;uk_s],0,0,0,0);
        yk =  [xk];[xk;c_C;T_cf;n_c];
        Xi = yk;
        CV = net.nn(uk_s,Xi,theta1);
        nlp.f = CV.^2;
        F = nlpsol('F','ipopt',nlp,opts);
        try
            sol = F('x0',data_opt(j).u(:,i),'ubx',par.ubu,'lbx',par.lbu);
        catch
            sol = F('x0',0,'ubx',par.ubu,'lbx',par.lbu);
        end
        uk = sol.x;
        
        
        CVf = Function('cv',{uk_s},{CV});

        uk = min(max(full(uk),0),0.1);
        
        ci(i,j) = full(CVf(uk));

        U_n = [U_n,uk];
        if xk(3) >= 1 %Vmax = 1
            uu = 0;
        else
            uu = dk(1)*xk(1)*xk(2)*xk(3)/(2-xk(2));
        end
        U_opt = [U_opt,uu];

        fk=Fk(xk,[par.d0;uk],0,0,0,0);
        xk_hat = full(fk);
        c_C_hat = (c_A0*V_0+c_C0*V_0-xk_hat(1,:).*xk_hat(3,:))./xk_hat(3,:);
        T_cf_hat = T+min(xk_hat(1,:),xk_hat(2,:))*-detH/pho/c_p;
        n_c_hat = xk_hat(3)*c_C_hat;
        Y_n = [Y_n,[xk;c_C;T_cf;n_c]];%;xk_hat;c_C_hat;T_cf_hat;n_c_hat

        fk=Fk(xk,[dk;uk],0,0,0,0);
        xk = full(fk);
        i=i+1;
    end
    tf(j)=i;
    Tv(j) = min(min(80-Y_n(5,:),0));
    Vv(j) = min(min(1-Y_n(3,:),0));
end
MM(2,1)=mean(tf(tf<401));MM(2,3)=-mean(Tv);MM(2,5)=-mean(Vv);
MM(2,2)=mean(tf==401);MM(2,4)=-min(Tv);MM(2,6)=-min(Vv);
%%
figure(4)
subplot(4,1,1)
plot((1:tf(1)-1)*0.1,Y_n(4,:),'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),(c_A0*V_0+c_C0*V_0-data_opt(1).y(1,:).*data_opt(1).y(3,:))./data_opt(1).y(3,:),'--','LineWidth',1.5)
% hold off
subplot(4,1,2)
plot((1:tf(1)-1)*0.1,Y_n(5,:),'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),T+min(data_opt(1).y(1,:),data_opt(1).y(2,:))*-detH/pho/c_p,'--','LineWidth',1.5)

subplot(4,1,3)
plot((1:tf(1)-1)*0.1,Y_n(3,:),'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),data_opt(1).y(3,:),'--','LineWidth',1.5)
subplot(4,1,4)
plot((1:tf(1)-1)*0.1,U_n,'LineWidth',2)
hold on 
% plot(0.1:0.1:(data_opt(1).tf-0.1),data_opt(1).u,'--','LineWidth',1.5)

%%
%% No pre exp
x = Y(1:3,:);
t = U;

% Choose a Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
setdemorandstream(4912183)
% Create a Fitting Network
hiddenLayerSize = [10 10 10];
net1 = fitnet(hiddenLayerSize,trainFcn);
for i=1:3
net1.layers{i}.transferFcn='poslin';
end
% Setup Division of Data for Training, Validation, Testing
net1.divideParam.trainRatio = 100/100;
net1.divideParam.valRatio = 0/100;
net1.divideParam.testRatio = 0/100;
net1.trainParam.epochs=10000;
% Train the Network
[net1,tr] = train(net1,x,t);
%%
import casadi.*
uk_s = MX.sym('uk',par.nu);
nlp = struct;nlp.x = uk_s; nlp.g = []; opts = struct('print_time',false, 'ipopt',struct('print_level',0,'max_iter',10));

Y_n=[];U_n=[];C=[];U_opt=[];
xk = par.x0;
Fk = sys.F;
for j=1:1%%length(d)
    i=1;n_c=0;xk = par.x0;dk = d(:,j);uk=U(1);Y_n=[];U_n=[];C=[];U_opt=[];
    while n_c <= n_Cdes && i<=400

        c_C = (c_A0*V_0+c_C0*V_0-xk(1,:).*xk(3,:))./xk(3,:);
        T_cf = T+min(xk(1,:),xk(2,:))*-detH/pho/c_p;
        n_c = xk(3)*c_C;
        xk_hat=Fk(xk,[par.d0;uk_s],0,0,0,0);
        yk =  [xk];[xk;c_C;T_cf;n_c];
        Xi = yk;
        uk =net1(full(Xi));

        uk = min(max(full(uk),0),0.1);
        

        U_n = [U_n,uk];
        if xk(3) >= 1 %Vmax = 1
            uu = 0;
        else
            uu = dk(1)*xk(1)*xk(2)*xk(3)/(2-xk(2));
        end
        U_opt = [U_opt,uu];

        fk=Fk(xk,[par.d0;uk],0,0,0,0);
        xk_hat = full(fk);
        c_C_hat = (c_A0*V_0+c_C0*V_0-xk_hat(1,:).*xk_hat(3,:))./xk_hat(3,:);
        T_cf_hat = T+min(xk_hat(1,:),xk_hat(2,:))*-detH/pho/c_p;
        n_c_hat = xk_hat(3)*c_C_hat;
        Y_n = [Y_n,[xk;c_C;T_cf;n_c]];%;xk_hat;c_C_hat;T_cf_hat;n_c_hat

        fk=Fk(xk,[dk;uk],0,0,0,0);
        xk = full(fk);
        i=i+1;
    end
    tf(j)=i;
    Tv(j) = min(min(80-Y_n(5,:),0));
    Vv(j) = min(min(1-Y_n(3,:),0));
end
MM(3,1)=mean(tf(tf<401));MM(3,3)=-mean(Tv);MM(3,5)=-mean(Vv);
MM(3,2)=mean(tf==401);MM(3,4)=-min(Tv);MM(3,6)=-min(Vv);
%%
figure(4)
subplot(4,1,1)
plot((1:tf(1)-1)*0.1,Y_n(4,:),'LineWidth',2)
hold on 
plot(0.1:0.1:(data_opt(1).tf-0.1),(c_A0*V_0+c_C0*V_0-data_opt(1).y(1,:).*data_opt(1).y(3,:))./data_opt(1).y(3,:),'--','LineWidth',1.5)
hold off
ylabel('$c_{C}(t)$','Interpreter','latex','FontSize',13)
legend('$Dlicit FCV Exporecasting$','$DCV Implicit Forecasting$','$Explicit policy$',"Optimal",'Interpreter','latex')
% xlabel('time h','Interpreter','latex')
subplot(4,1,2)
plot((1:tf(1)-1)*0.1,Y_n(5,:),'LineWidth',2)
hold on 
plot(0.1:0.1:(data_opt(1).tf-0.1),T+min(data_opt(1).y(1,:),data_opt(1).y(2,:))*-detH/pho/c_p,'--','LineWidth',1.5)
hold off
ylabel('$T_{cf}(t)$','Interpreter','latex','FontSize',13)
legend('$DCV Explicit Forecasting$','$DCV Implicit Forecasting$','$Explicit policy$',"Optimal",'Interpreter','latex')
% xlabel('time h','Interpreter','latex')
subplot(4,1,3)
plot((1:tf(1)-1)*0.1,Y_n(3,:),'LineWidth',2)
hold on 
plot(0.1:0.1:(data_opt(1).tf-0.1),data_opt(1).y(3,:),'--','LineWidth',1.5)
hold off
ylabel('$V(t)$','Interpreter','latex','FontSize',13)
legend('$DCV Explicit Forecasting$','$DCV Implicit Forecasting$','$Explicit policy$',"Optimal",'Interpreter','latex')
% xlabel('time h','Interpreter','latex')
subplot(4,1,4)
plot((1:tf(1)-1)*0.1,U_n,'LineWidth',2)
hold on 
plot(0.1:0.1:(data_opt(1).tf-0.1),data_opt(1).u,'--','LineWidth',1.5)
hold off
ylabel('$u(t)$','Interpreter','latex','FontSize',13)
xlabel('time h','Interpreter','latex','FontSize',13)
legend('$DCV Explicit Forecasting$','$DCV Implicit Forecasting$','$Explicit policy$',"Optimal",'Interpreter','latex')

save case2 theta0 theta1 net net1