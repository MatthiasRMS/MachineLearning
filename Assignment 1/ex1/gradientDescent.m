function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % assigning to x the second column of X
    x = X(:,2);

    % in this exercise, we have a simple version where we only have two features
    h = theta(1) + (theta(2)*x);

    % theta has two components here:
    theta_zero = theta(1) - alpha * (1/m) * sum(h-y);
    theta_one  = theta(2) - alpha * (1/m) * sum((h - y) .* x);

    % store in theta the above results
    theta = [theta_zero; theta_one];

      J_history(iter) = computeCost(X, y, theta);

  end

end
