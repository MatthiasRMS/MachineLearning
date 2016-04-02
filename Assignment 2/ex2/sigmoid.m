
% arrayfun applies to each element of an array the given function

function g = sigmoid(z)
  g = arrayfun(@(z) 1/(1+exp(-z)), z)
end
