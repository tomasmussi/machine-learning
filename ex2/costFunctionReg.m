function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% First, cost function:
pos = find(y==1); neg = find(y == 0);
sig = sigmoid(X * theta);
posValues = -1 .* log(sig(pos,:));
negValues = -1 .* log(1 .- (sig(neg,:)));
J = (sum(posValues) + sum(negValues)) / m;

% Add regularization to cost function J(theta)
% Do not regularize theta[0] 
reg = [theta(2:rows(theta)) .^2]';
%printf("Reg %.2f \n",reg);
%printf("[%d] %.2f \n",lambda, ((lambda / (2*m)) * sum(reg)));
J = J + ((lambda / (2*m)) * sum(reg));

% Second, derivative of J(theta)
temp1 = sigmoid(X * theta) - y;
temp2 = temp1 .* X;
temp3 = temp2' * ones(m, 1);
grad = temp3 ./ m;
% Add regularization
%printf("theta %.2f \n",theta);
derReg = (lambda/m) .* [0, theta(2:rows(theta))']';
%printf("[%d] %.2f \n",lambda, derReg);
grad = grad + derReg;



% =============================================================

end
