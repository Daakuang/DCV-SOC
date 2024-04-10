function [theta,Losses,Grades]=trainNN_mse(net,Loss,sol,par,w0,miniBatchSize,numEpochs)
% a batch process nn training process

import casadi.*
nn = net.nn;
corrP = net.corrP;
% theta = net.theta;
num_u = nn.n_out;
num_Xi = nn.n_in; 

num_theta = nn.numel_in - num_Xi;
w = MX.sym('theta',num_theta);
% uk_s = MX.sym('uk',num_u);
Xi_s = MX.sym('Xi',num_Xi);
uk_s = (nn(Xi_s,w));


Loss_i = cellfun(@(sol)(sol.x(num_Xi+1)-nn(sol.x(1:num_Xi),w)).^2,sol,UniformOutput=false);
Loss_F = 0;
for i=1:length(sol)
Loss_F = Loss_F + Loss_i{i}/length(sol);
end



rng(10086)
if nargin<5 || isempty(w0) 
    w0 = [net.w0];
end

%% NLP
% opts = struct('ipopt',struct('max_iter',5000));
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
grad = Function('grad_f',{[w]},{jacobian(Loss_F,[w])});
f = Function('loss_f',{[w]},{Loss_F});
theta = w0; 


numObservations = length(sol);
% miniBatchSize = 300; %<numObservations
% numEpochs = floor(numObservations./miniBatchSize)*10;
numIterationsPerEpoch=1;%
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor(Metrics=["Loss","NormOfGradient"],Info="Epoch",XLabel="Iteration");

k=0;
iteration=0;
epoch = 0;
i=0; 
% Index = randperm(numObservations); 
% a=1;b=a+miniBatchSize;
loss=100;
while i < numIterations && ~monitor.Stop && loss>=0.01
    epoch = epoch + 1;
    
    % if b > numObservations
    %     b=b-numObservations;
    %     idx = Index([a:numObservations,1:b]);
    % else  
    %     idx = Index(a:b);%Index(((epoch - 1)*miniBatchSize + 1):(epoch*miniBatchSize));
    % end
    % 
    % a = b+1;
    % if a >= numObservations
    %     a=1;
    % end
    % b=a+miniBatchSize;

    % Loss = mean(LossFunc(zeros(num_u,length(idx)),U(:,idx),Y(:,idx),w));
    % grad = Function('grad_f',{[w]},{jacobian(Loss,[w])});
    % f = Function('loss_f',{[w]},{Loss});
    % 这里还可以优化 写在循环外面
    for j =1:numIterationsPerEpoch
        i = i + 1;
        iteration=iteration+1;

        grad_f=grad(theta)';
         % Update the network parameters using the Adam optimizer.
        [theta,averageGrad,averageSqGrad] = adamupdate(theta,grad_f,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay);
        if mod(i,10)==0
            loss = full(f(theta));
            k = k+1;
            Losses(k)=loss;
            Grades(k)=full(norm(grad_f));
            % Update the training progress monitor.
            recordMetrics(monitor,iteration,Loss=log(full(loss)),NormOfGradient=full(log(norm(grad_f))));
            updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
            monitor.Progress = 100 * iteration/numIterations;
        end
    end
end

    % Update the network parameters using the SGDM optimizer.
    % [theta,vel] = sgdmupdate(theta,grad_f,vel);
    % Update the network parameters using the Adam optimizer.
    % [theta,averageGrad,averageSqGrad]=adamstep_my(theta,grad_f,averageGrad,averageSqGrad,iteration,learnRate,gradDecay,sqGradDecay,1e-8);
