

%% Initialization
clear ; close all; clc

%% 1. Load the Data


fprintf('Loading Data ...\n')


% Load concrete dataset into X, y, Xval, yval, Xtest, ytest: 

data = load('Concrete_Data.csv');
X = data(1:721 , 1:8);
y = data(1:721 , 9);

Xval = data(722:928 , 1:8);
yval = data(722:928 , 9);

Xtest = data(929:1030 , 1:8);
ytest = data(929:1030 , 9);

% m = Number of examples
[m n] = size(X);

%Initializa theta
theta = zeros(n + 1, 1);

fprintf(['Training set: %f loaded \nCross validation set: %f loaded \nTest set: %f loaded \n'], round(size(X, 1)), 
			round(size(Xval, 1)), round(size(Xtest, 1)));
fprintf('Program paused. Press enter to continue.\n');
pause;


%% 2. Feature normalization procedure

[X, mu, sigma] = featureNormalize(X);
[Xval, mu, sigma] = featureNormalize(Xval);
[Xtest, mu, sigma] = featureNormalize(Xtest);
X = [ones(size(X, 1), 1) X];
Xval = [ones(size(Xval, 1), 1) Xval];
Xtest = [ones(size(Xtest, 1), 1) Xtest];

%% 2. Create regularized linear regression cost function

J = linearRegCostFunction(Xval, yval, theta, 1);

fprintf(['Cost at initial theta = %f \n'], J);


fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 4: Train Linear Regression =============
%  Train linear regression with lambda = 0
lambda = 0;
alpha = 0.007
epochs = 1000
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, epochs, lambda);
plot(1:size(J_history, 1), J_history);
title('Learning rate')
xlabel('Epochs')
ylabel('Cost')
fprintf(['Training cost = %f\n Cross validation cost = %f\n'], J_history(epochs, 1), linearRegCostFunction(Xval, yval, theta, lambda));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% 5. Learning Curve for Linear Regression
%% You can use the learning curves to diagnose whether the model is suffering from high bias or high variation 

fprintf('Please wait while the learning curves are generated.\n');

lambda = 0;
[error_train, error_val] = learningCurve(X, y, Xval, yval, alpha, epochs, lambda);
figure;
plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

fprintf('Training error = %f, Cross validation error = %f\n', error_train(end), error_val(end));


fprintf('Program ended.\n');
pause;