function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, alpha, epochs, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%

% Number of training examples
[m n] = size(X);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

j = 0;

for i = 1:m
theta = zeros(n, 1);
[theta, J_history] = gradientDescentMulti(X(1:i, :), y(1:i, :), theta, alpha, epochs, lambda);
J_train = linearRegCostFunction(X(1:i, :), y(1:i, :), theta, 0);
error_train(i,1) = J_train;
J_cv = linearRegCostFunction(Xval, yval, theta, 0);
error_val(i,1) = J_cv;

j = j+1;
end






% -------------------------------------------------------------

% =========================================================================

end
