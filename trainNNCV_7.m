function theta=trainNNCV_7(net,sys,par,data,w0,numIterationsPerEpoch)
% a batch process nn CV training process

import casadi.*

nn = net.nn;
corrP = net.corrP;
% theta = net.theta;
num_u = size(data(1).u,1);
num_Xi = size(data(1).y,1);

num_theta = nn.numel_in - num_u - num_Xi;
w = MX.sym('theta',num_theta);
uk_s = MX.sym('uk',num_u);
Xi_s = MX.sym('Xi',par.nx);

% L = 1000; MX.sym('L',num_u);
% L2 = MX.sym('L2',num_u)
b_s = MX.sym('b',par.nu);
c_s = (nn(uk_s,[Xi_s],w));
dcdu = jacobian(c_s,uk_s);
Loss_s = norm(dcdu\(b_s-c_s));%+10*norm(dcdu - 1);
 
% Loss_s = norm(dcdu\(c_s));
% Xi_f_func = Function('loss',{uk_s,Xi_s},{Xi_f});
LossFunc = Function('loss',{b_s,uk_s,Xi_s,w},{Loss_s});
% dcduFunc = Function('dcdu',{uk_s,Xi_s,Xi_f_s,w},{norm(dcdu+1)});
% cFunc = Function('dcdu',{uk_s,Xi_s,Xi_f_s,w},{c_s});

Loss = 0;
g=[];{};
lbg = [];
ubg = [];

N = 0;
for i=1:length(data)
    N = N + size(data(i).u,2);
end

for i=1:length(data)
    for j = 1:size(data(i).u,2)
        u = data(i).u(:,j);
        Loss = Loss + 1/N*(LossFunc(zeros(par.nu,1),u,data(i).y(1:par.nx,j),w));
        % Loss = Loss + 1/N*sum(LossFunc((-1e-3+1e-5:1e-4:1e-3)*L,u+(-1e-3+1e-5:1e-4:1e-3),repmat(data(i).y(1:par.nx,j),1,20),Xi_f_func(u+(-1e-3+1e-5:1e-4:1e-3),repmat(data(i).y(1:par.nx,j),1,20)),w));

    end
end

% g=norm(w(corrP{end}(1:end-num_u)));
% lbg = [ones(num_u,1)];
% ubg = [ones(num_u,1)];
% g=[w(end)];
% lbg = [0];
% ubg = [0];


rng(10086)
if nargin<5 || isempty(w0) 
    w0 = [net.w0];
end

%% NLP
opts = struct('ipopt',struct('max_iter',5000));
% nlp_prob = struct('f', Loss, 'x', [L;w], 'g', g);
% nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob,opts); % Solve relaxed problem
% Solve the NLP
% sol = nlp_solver('x0',w0, 'lbg',lbg, 'ubg',ubg);%, 'lbx',lbw, 'ubx',ubw);
% flag = nlp_solver.stats();
% flag.success
% theta = full(sol.x);

% grad = nlp_solver.get_function('nlp_grad_f');
% f = nlp_solver.get_function('nlp_f');
%% 梯度下降
vel=[];

learnRate = 0.001;
gradDecay = 0.9;
sqGradDecay = 0.999;
averageGrad = 0;
averageSqGrad = 0;
grad = Function('grad_f',{[w]},{jacobian(Loss,[w])});
f = Function('loss_f',{[w]},{Loss});
theta = w0; 

numEpochs = 1;
% numIterationsPerEpoch=10000;%floor(numObservations./miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor(Metrics=["Loss","NormOfGradient"],Info="Epoch",XLabel="Iteration");

iteration=0;
epoch = 0;
epoch = epoch + 1;

i=0;
while i < numIterationsPerEpoch && ~monitor.Stop
    i = i + 1;
    iteration=iteration+1;
    grad_f=grad(theta)';
    % [theta,vel] = sgdmupdate(theta,grad_f,vel);
    % Update the network parameters using the Adam optimizer.
    [theta,averageGrad,averageSqGrad] = adamupdate(theta,grad_f,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay);
    % Update the network parameters using the SGDM optimizer.
    % [theta,vel] = sgdmupdate(theta,grad_f,vel);
    % Update the network parameters using the Adam optimizer.
    % [theta,averageGrad,averageSqGrad]=adamstep_my(theta,grad_f,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay,1e-8);
    % % if  isempty(averageSqGrad)
    %     averageSqGrad = averageSqGrad1;
    % else
    %     averageSqGrad = max(averageSqGrad1,averageSqGrad);
    % end
    if mod(i,10)==0
        % temp = loss;
        loss = f(theta);
        % if loss >= 10*temp
        % 
        % end
        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=log(full(loss)),NormOfGradient=full(log(norm(grad_f))));
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
    end
end


