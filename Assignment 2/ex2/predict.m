function p = predict(theta, X)
  % predict whether the label is 0 or 1 using learned logistic
  % regression parameters theta
  % p = PREDICT(theta, X) computes the predictions for X using a
  % threshold at 0.5

  % Number of training examples
  m = size(X, 1);

  % set the prediction size
  p = zeros(m, 1);

  for i = 1:m
    prediction = sigmoid(theta' * X'(:,i))
    if (prediction < 0.5)
      p(i) = 0
    endif

    if (prediction >= 0.5)
      p(i) = 1
    endif
  end

end
