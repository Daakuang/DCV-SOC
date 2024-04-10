function theta=trainNNCV_1(net,x,y,w0,numIterationsPerEpoch)
% a batch process nn CV training process

import casadi.*

nn = net.nn;
% corrP = net.corrP;
% theta = net.theta;
num_x = size(x,1);
num_y = size(y,1);

num_theta = nn.numel_in - num_x - num_y;
w = MX.sym('theta',num_theta);
x_s = MX.sym('x',num_x);
y_s = MX.sym('y',num_y);

% L = 1000; MX.sym('L',num_x);
% L2 = MX.sym('L2',num_u)
b_s = MX.sym('b',num_x);
c_s = (nn(x_s,[y_s],w));
dcdu = jacobian(c_s,x_s);
Loss_s = norm(dcdu\(b_s-c_s));%+10*norm(dcdu - 1);
 
% Loss_s = norm(dcdu\(c_s));
% Xi_f_func = Function('loss',{uk_s,Xi_s},{Xi_f});
LossFunc = Function('loss',{b_s,x_s,y_s,w},{Loss_s});
% dcduFunc = Function('dcdu',{uk_s,Xi_s,Xi_f_s,w},{norm(dcdu+1)});
% cFunc = Function('dcdu',{uk_s,Xi_s,Xi_f_s,w},{c_s});

Loss = 0;
g=[];{};
lbg = [];
ubg = [];

% N = 0;
% for i=1:length(data)
%     N = N + size(data(i).u,2);
% end

% for i=1:length(data)
        Loss = mean(LossFunc(zeros(num_x,length(x)),x,y,w));

% end

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
       
   
        % Update the training progress monitor.
        recordMetrics(monitor,iteration,Loss=log(full(loss)),NormOfGradient=full(log(norm(grad_f))));
        updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
        monitor.Progress = 100 * iteration/numIterations;
    end
end


