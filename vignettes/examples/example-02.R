library(MASS)
data(Boston)

X <- Boston[,2:14]
y <- Boston[,1]
 
deepLearnR::TensorFlowDNNRegressor(modelTag="tfdnnr-01", X=X, y=y, steps=5000)
pred <- deepLearnR::TensorFlow.regressorEval(modelTag="tfdnnr-01")
mse <- rPython::python.get("mse")
r2 <- rPython::python.get("r2")