function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters, lambda)

% Initialize some J_history for plotting learning curve, gradients and other useful values
% Size of theta = 9 x 1
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
grad = zeros(size(theta));

% Calculate gradients
for iter = 1:num_iters
	for i = 1:size(theta, 1)
	grad(i,1) = (1/m) * sum(((X*theta) - y).*X(:, i));
	end
	grad(2:size(theta, 1),1) = grad(2:size(theta, 1),1) + (lambda/m) * theta(2:size(theta, 1),1);
	theta = theta - (alpha * grad);
	J_history(iter) = linearRegCostFunction(X, y, theta, lambda);

end

end