function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% First, cost function:
pos = find(y==1); neg = find(y == 0);
sig = sigmoid(X * theta);
posValues = -1 .* log(sig(pos,:));
negValues = -1 .* log(1 .- (sig(neg,:)));
J = (sum(posValues) + sum(negValues)) / m;

temp1 = sigmoid(X * theta) - y;
temp2 = temp1 .* X;
temp3 = temp2' * ones(m, 1);
grad = temp3 ./ m;




% =============================================================

end
