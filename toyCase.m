% x^2-y^2=1;
x = [1:0.02:2,2.1:0.1:10];
x = [x,-x]; 
y = sqrt(x.^2-1);
y = [y,-y];%,(-10:0.1:-1)*0,(-10:0.1:-1)*0]; 
x = [x,x];%,-10:0.1:-1,1:0.1:10]; 
%%
par.nu =1; 
net = createNN([2 10 10 10 10 1],par);
% net = createNN_maxout([7 10 10 par.nu],2,par);
tic
theta1=trainNNCV_1(net,x,y,[],10000);
% theta1=trainNNCV_1(net,x,y,theta1,5000);
toc
save toyCase1 theta1 net
%%
x = [1:0.02:2,2.1:0.1:9.5];
x = [x,-x]; 
y = sqrt(x.^2-1);
y = [y,-y];%,(-10:0.1:-1)*0,(-10:0.1:-1)*0]; 
x = [x,x];%,-10:0.1:-1,1:0.1:10]; 

import casadi.*
% x_s = MX.sym('x',1);
y_s = MX.sym('y',1);yhat=[];
for i=1:length(x)
CV = net.nn(x(i),y_s,theta1);
CVf = Function('cv',{y_s},{CV-0});
CV_uk = rootfinder('CV_uk','newton',CVf);
uk = CV_uk(y(i));
yhat(i) = full(uk);
end

[X,Y] = meshgrid(-10:.01:10,linspace(min(y)-1,max(y)+1,length(-10:.01:10)));
Z = net.nn(reshape(X,1,[]),reshape(Y,1,[]),theta1);
Z = full(reshape(Z,2001,2001));
%%
figure(1)
contour(X,Y,Z,'LineWidth',2)
hold on
mesh(X,Y,Z)
colorbar
plot(x,y,'bx','LineWidth',2)
plot(x,yhat,'+','LineWidth',1)
hold off
legend('','','Training samples','Predictons','Location','northwest')
%%
x = [-10.001:0.01:10];
y = 10*sign(x)+3*x;y=[y,10,-10];x=[x,0,-0];
%%
par.nu =1; 
net1 = createNN([2 10 10 10 10 1],par);
% net = createNN_maxout([7 10 10 par.nu],2,par);
tic
theta2=trainNNCV_1(net1,x,y,[],10000);
% theta1=trainNNCV_1(net,x,y,theta1,5000);
toc
save toyCase2 theta2 net1
%%
clear yhat
x = [-10.0001:0.001:10];
y = 10*sign(x)+3*x;y=[y,10,-10];x=[x,0,-0];
figure(4)
plot(full(net1.nn(x,y,theta2))); 
import casadi.*
% x_s = MX.sym('x',1);
y_s = MX.sym('y',1);
for i=1:length(x)
CV = net1.nn(x(i),y_s,theta2);
CVf = Function('cv',{y_s},{CV-0});
CV_uk = rootfinder('CV_uk','newton',CVf);
uk = CV_uk(y(i));
yhat(i) = full(uk);
end


[X,Y] = meshgrid(-10:.01:10,linspace(min(y),max(y),length(-10:.01:10)));
Z = net1.nn(reshape(X,1,[]),reshape(Y,1,[]),theta2);
Z = full(reshape(Z,2001,2001));
%%
figure(6)
contour(X,Y,Z,'LineWidth',2)
hold on
mesh(X,Y,Z)
colorbar
plot(x,y,'bx','LineWidth',2)
plot(x,yhat,'+','LineWidth',1)
legend('','','Training samples','Predictons','Location','northwest')
hold off
%%  controller

% x^2-y^2=1;
x = [1:0.02:2,2.1:0.1:10];
x = [x,-x]; 
y = sqrt(x.^2-1);
y = [y,-y];%,(-10:0.1:-1)*0,(-10:0.1:-1)*0]; 
x = [x,x];%,-10:0.1:-1,1:0.1:10]; 
%%
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% Create a Fitting Network
hiddenLayerSize = [10 10 10 10];
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % 均方误差
% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
% Train the Network
[net,tr] = train(net,x,y);
% Test the Network
yhat = net(x);
%%
figure(2)
plot(x,y,'bx','LineWidth',1)
hold on
plot(x,yhat,'+','LineWidth',1)
hold off
legend('Training samples','Predictons','Location','northwest')
%%
x = [-10.001:0.01:10];
y = 10*sign(x)+3*x;
%%
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% Create a Fitting Network
hiddenLayerSize = [10 10 10 10];
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % 均方误差
% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};
% Train the Network
[net,tr] = train(net,x,y);
% Test the Network

%%
x = [-10.0001:0.001:10];
y = 10*sign(x)+3*x;
yhat = net(x);
figure(7)
plot(x,y,'bx','LineWidth',1)
hold on
plot(x,yhat,'+','LineWidth',1)
hold off
legend('Training samples','Predictons','Location','northwest')


