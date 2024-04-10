function net = createNN_maxout(neurons_per_layer,num_max,par)
% create a CV NN
% nu is the dim of uk
% num_layers n e.g 5 including input and output layer
% neurons_per_layer  
% e.g. [4 2 3 4 2] first is the size of input  end is the  size of output
nu = par.nu;
lbu = par.lbu;
ubu = par.ubu;
num_layers = length(neurons_per_layer);

% Load CasADi
import casadi.*

% Create symbolic variables for learable paremters
corrP{num_layers-1} =[];startInd = 1;
for i =1:num_layers-1
    per_P_num = (neurons_per_layer(i)+1)*neurons_per_layer(i+1);
    corrP{i} = startInd:startInd+per_P_num-1 ;
    startInd = startInd+per_P_num;
end
toll_P_num = startInd-1;
theta = MX.sym('theta',toll_P_num,num_max);

% Create symbolic variables for inputs and outputs
uk = MX.sym('uk',nu);  % Input
Xi = MX.sym('Xi',  neurons_per_layer(1)-nu); % Output

out = [];
% Create hidden layers
for j = 1:num_max
    outLayer = [uk;Xi];
    for i = 1:num_layers - 1
        W = reshape(theta(corrP{i}(1:end-neurons_per_layer(i+1)),j),neurons_per_layer(i+1),neurons_per_layer(i));
        b = theta(corrP{i}(end-neurons_per_layer(i+1)+1:end),j);
        if i< num_layers - 1
            outLayer = max(W*outLayer+b,0); % Replace with desired activation function
        else
            % Create output layer (using linear activation)
            outLayer = W*outLayer+b;
            % outLayer = min(max(W*outLayer+b,lbu),ubu);
        end
    end
    out = [out ,outLayer];
end
maxout = [];
for k  = 1:nu 
    temp = out(k,1);
    for j = 2:num_max
        temp = max(out(k,j),temp);
    end
    maxout = [maxout;temp];
end
% Define the neural network as a CasADi function
nn = Function('nn', {uk,Xi,theta(:)}, {maxout});

net.nn =nn;
net.corrP = corrP;
end
