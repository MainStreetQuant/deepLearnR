library(testthat)
library(deepLearnR)
#
library(MASS)
data(Boston)
X <- Boston[,2:14]
y <- Boston[,1]
#
testthat::test_that("Creates deepLearnR::TensorFlowDNNRegressor Model", {
  result <- deepLearnR::TensorFlowDNNRegressor(modelTag="test-02", X=X, y=y, steps=5000)
  testthat::expect_that(result, testthat::matches("test-02"))
})
#
testthat::test_that("deepLearnR::TensorFlowDNNRegressor Model : Predicts Test Dataset w/ Regression Model; Check MSE", {
  pred <- deepLearnR::TensorFlow.regressorEval(modelTag="test-02")
  mse <- python.get("mse")
  testthat::expect_less_than(mse, 100)
})
