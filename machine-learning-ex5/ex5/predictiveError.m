function[] = predictiveError(Xtest, Ytest, theta)
  
  %% Don't want reguralisation parameter to test this
  linearRegCostFunction(Xtest, Ytest, theta, 0)
end