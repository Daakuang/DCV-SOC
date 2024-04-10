function theta=trainNNCV_minibatch(net,sys,par,U,Y,w0,miniBatchSize,numEpochs)
% a batch process nn CV training process

import casadi.*

nn = net.nn;
corrP = net.corrP;
% theta = net.theta;
num_u = size(U,1);
num_Xi = size(Y,1);


num_theta = nn.numel_in - num_u - num_Xi;
w = MX.sym('theta',num_theta);
uk_s = MX.sym('uk',num_u);
Xi_s = MX.sym('Xi',num_Xi);
b_s = MX.sym('b',num_u);

% L = 1000; MX.sym('L',num_u);
% L2 = MX.sym('L2',num_u)

% Xi_f= sys.F(Xi_s,[par.d0;uk_s],0,0,0,0);
try
    xk_s  = MX.sym(['X_' num2str(0)],par.nx);
    X = [xk_s];X_s = [xk_s];U_s = [];
    % xk_s = Xi_s;
    for k =0:par.H-1
        uk_s  = MX.sym(['U_' num2str(k)],par.nu);
        xk = sys.F(xk_s,[par.d0;uk_s],0,0,0,0);
        xk_s  = MX.sym(['X_' num2str(k+1)],par.nx);
        X = [X ; xk];
        X_s = [X_s ; xk_s];
        U_s = [U_s ; uk_s];
    end
    c_s = nn(U_s,X_s,w);
    dcdu = jacobian(c_s,U_s)+jacobian(c_s,X_s)*jacobian(X,U_s);
    Loss_s = norm(dcdu\(b_s-c_s));
    % Loss_s = norm((b_s-c_s));
    LossFunc = Function('loss',{b_s,U_s,X_s,w},{Loss_s});
catch
    c_s = (nn(uk_s,Xi_s,w));
    dcdu = jacobian(c_s,uk_s);%+jacobian(c_s,Xi_f_s)*jacobian(Xi_f,uk_s);
    Loss_s = norm(dcdu\(b_s-c_s));
    Loss_s = norm((b_s-c_s));
    LossFunc = Function('loss',{b_s,uk_s,Xi_s,w},{Loss_s});
end
% Xi_f_s = MX.sym('Xi_f',par.nx);

% Loss_s = norm(dcdu\(b_s-c_s));%+10*norm(dcdu - 1);
 
% try
%     LossFunc = Function('loss',{b_s,U_s,X_s,w},{Loss_s});
% catch
%     LossFunc = Function('loss',{b_s,uk_s,Xi_s,w},{Loss_s});
% end


Loss = mean(LossFunc(zeros(num_u,length(U)),U,Y,w));
for i =0:0.4:2
   Loss = Loss+ mean(LossFunc(ones(num_u,length(U))*i-U,ones(num_u,length(U))*i,Y,w));
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
grad = Function('grad_f',{[w]},{jacobian(Loss,[w])});
f = Function('loss_f',{[w]},{Loss});
theta = w0; 


numObservations = length(U);
% miniBatchSize = 300; %<numObservations
% numEpochs = floor(numObservations./miniBatchSize)*10;
numIterationsPerEpoch=10;%
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor(Metrics=["Loss","NormOfGradient"],Info="Epoch",XLabel="Iteration");

iteration=0;
epoch = 0;
i=0; 
Index = randperm(numObservations); 
a=1;b=a+miniBatchSize;
while i < numIterations && ~monitor.Stop
    epoch = epoch + 1;
    
    if b > numObservations
        b=b-numObservations;
        idx = Index([a:numObservations,1:b]);
    else  
        idx = Index(a:b);%Index(((epoch - 1)*miniBatchSize + 1):(epoch*miniBatchSize));
    end
    
    a = b+1;
    if a >= numObservations
        a=1;
    end
    b=a+miniBatchSize;

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
            loss = f(theta);
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
