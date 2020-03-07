function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% 1) Forward propagation
X = [ones(m, 1), X];
% Convert y vector to a matrix
Y = zeros(m, max(y));
for i = 1:m
  Y(i, y(i)) = 1;
endfor

% Do forward propagation
a1 = sigmoid(Theta1 * X'); % Compute a1
a1 = [ones(1, m); a1]; %Add zeros
a2 = sigmoid(Theta2 * a1)';
[max_vals, max_indexes] = max(a2, [], 2);

% Compute cost
for i = 1:m
  for k = 1:num_labels
    tmp = - Y(i,k) * log(a2(i,k)) - (1 - Y(i,k)) * log(1 - a2(i,k));
    J += tmp;
  endfor
endfor
J /= m;

% Add regularization to cost
tmp = 0;
for i = 1:size(Theta1, 1)
  for j = 2:size(Theta1, 2)
    tmp += Theta1(i,j)^2;
  endfor
endfor

for i = 1:size(Theta2, 1)
  for j = 2:size(Theta2, 2)
    tmp += Theta2(i,j)^2;
  endfor
endfor
J += (tmp * lambda) / (2 * m);

delta_2 = zeros(size(Theta1));
delta_3 = zeros(size(Theta2));

% 2) BACKPROPAGATION
% As suggested, implement a for loop with 5 steps
for t = 1:m
  % Step 1: Forward propagation
  a1 = X(t,:);
  z2 = Theta1 * a1';
  a2 = sigmoid(z2);
  a2z = [1 ; a2];
  z3 = Theta2 * a2z;
  a3 = sigmoid(z3);
   
  % Step 2: Compute error in layer 3
  % Y is a matrix that for each sample, it has a vector with 1 in corresponding column class
  y_vec = Y(t,:)';
  d3 = a3 - y_vec;
  
  % Step 3: now compute error in layer 2
  % Remove from Theta2 the first column which corresponds to Bias unit
  d2 = (Theta2(:,2:end))' * d3 .* sigmoidGradient(z2); % Gradient for sample t
  
  % Step 4: accumulate gradient
  delta_3 = delta_3 + d3 * [1; a2]';
  
  % Activations in first layer are the sample inputs
  
  temp_m = d2 * a1;
  % temp_m(:,1) = zeros(rows(temp_m), 1);
  % Add zeros to sum to gradient to avoid changing bias
  % temp_m = [zeros(25,1), temp_m];
  delta_2 = delta_2 + temp_m;
endfor

% Once iterated over all samples, update Theta_grads

% Average by all samples
delta_2 = delta_2 / m;
delta_3 = delta_3 / m;

% Regularization to delta matrices
regTheta1 = (lambda/m) .* Theta1;
regTheta1(:,1) = zeros(rows(Theta1),1);

regTheta2 = (lambda/m) .* Theta2;
regTheta2(:,1) = zeros(rows(Theta2),1);


Theta1_grad = Theta1_grad + delta_2 + regTheta1;
Theta2_grad = Theta2_grad + delta_3 + regTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
