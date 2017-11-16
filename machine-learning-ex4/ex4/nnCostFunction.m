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


A1 = [ones(m, 1), X];
J = 0;
for iFeature = 1:num_labels
  %do sht about the 10 which is 0 in fact
  correct = y == iFeature;
  predict_a2 = sigmoid(A1 * Theta1');
  predict_a2 = [ones(size(predict_a2, 1), 1), predict_a2];
  predict_a3 = sigmoid(predict_a2 * Theta2');
  predicted_feature = predict_a3(:, iFeature);
  semiJ = sum(-correct .* log(predicted_feature) - (1-correct) .* log(1 - predicted_feature));
  J = J + semiJ;
endfor
J = 1/m * J;
% We basically just summarise Theta across columns and then rows butt without the first column


% Vectorised solution ----- possibly faster, don#t care . I read mine sollution better
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
 Y(i, :)= I(y(i), :);
end

Z2 = A1 * Theta1';
A2 = [ones(size(Z2, 1), 1) sigmoid(Z2)];
Z3 = A2*Theta2';
H = A3 = sigmoid(Z3);
J = (1/m)*sum(sum((-Y).*log(H) - (1-Y).*log(1 - H), 2));

% Reguralisation
regPart = (lambda/(2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2), 2) + (sum(sum(Theta2(:, 2:end) .^ 2), 2)));
J = J + regPart;
% -------------------------------------------------------------

% =========================================================================

% going to compute it separately for all X
% sig3 = zeros(m, num_labels);
% sig2 = zeros(m, hidden_layer_size + 1);
% delta1 = zeros(m, num_labels);
% delta2 = zeros(m, hidden_layer_size + 1);
% for t = 1:1
%   a1 = A1(t, :);
%   z2 = a1 * Theta1';
%   a2 = [1 sigmoid(z2)];
%   z3 = a2 * Theta2';
%   a3 = sigmoid(z3);
%   yk = 1:10 == y(t);
%   sig3(t, :) = a3 - yk;
%   sig2(t, :) = (sig3(t, :) * Theta2) .* sigmoidGradient([1, z2]);
%   delta1 = delta1 + sig2(t, 2:end)' * a1;
%   delta2 = delta2 + sig3(t)' * a2;
% endfor

% Theta1_grad = (1/m) .* delta1;
% Theta2_grad = (1/m) .* delta2;

% vectorised
Sigma3 = A3 - Y;
Sigma2 = (Sigma3 * Theta2) .* sigmoidGradient([ones(size(Z2, 1), 1) Z2]);
Sigma2 = Sigma2(:, 2:end);

% accumulate gradients
delta_1 = (Sigma2'*A1);
delta_2 = (Sigma3'*A2);
% calculate regularized gradient
p1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:, 2:end)];
p2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:, 2:end)];

Theta1_grad = delta_1./m + p1;
Theta2_grad = delta_2./m + p2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];