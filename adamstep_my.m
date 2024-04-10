function [theta,averageGrad,averageSqGrad]=adamstep_my(theta,grad_f,averageGrad,averageSqGrad,t,lr,beta1,beta2,epsilon)

if nargin < 6
lr = 0.001;
beta1 = 0.9;
beta2 = 0.999;
epsilon=1e-8;
end

averageGrad = beta1.*averageGrad + (1-beta1).*grad_f;
averageSqGrad = max(beta2.*averageSqGrad + (1-beta2).*grad_f.^2,averageSqGrad);
% averageSqGrad = beta2.*averageSqGrad + (1-beta2).*grad_f.^2;
averageGradCorrection = averageGrad./(1-beta1.^t);
averageSqGradCorrection = averageSqGrad./(1-beta2.^t);
theta = theta - lr .* averageGradCorrection./(sqrt(averageSqGradCorrection)+epsilon);

end