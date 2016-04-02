function [J, grad] = costFunctionReg(theta, X, y, lambda)

%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

  % Initialize some useful values
  % number of training examples
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  sz = size(theta)(1);

  A = (-y) .* log(sigmoid(X * theta));
  B = (1 - y) .* log(1 - sigmoid(X * theta));

  J = ((1/m) * sum(A - B)) + ((lambda/(2*m)) * sum(theta(2:sz,:) .^2));

  grad =((sigmoid(X * theta) - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;

end
