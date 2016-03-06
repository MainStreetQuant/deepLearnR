library(testthat)
library(deepLearnR)
#
Y <- deepLearnR::titanic.data$Survived
X <- deepLearnR::titanic.data[,c("Age","SibSp","Fare","Pclass")]
X$Age[is.na(X$Age)] <- mean(X$Age,na.rm=TRUE)
#
set.seed(512)
inTrain <- sample(1:nrow(X), trunc(nrow(X)*0.8))
X.Train <- X[inTrain,]
Y.Train <- Y[inTrain]
X.Test <- X[-inTrain,]
Y.Test <- Y[-inTrain]
#
testthat::test_that("Creates deepLearnR::TensorFlow.Classifier Linear Model", {
  result <- deepLearnR::TensorFlow.Classifier(modelTag="test-01", X=X.Train, Y=Y.Train, steps=5000)
  testthat::expect_that(result, testthat::matches("test-01"))
})
#
testthat::test_that("deepLearnR::TensorFlow.Classifier : Predicts Test Dataset w/ Linear Model", {
  pred <- deepLearnR::TensorFlow.predict(modelTag="test-01", X=X.Test, Y=Y.Test)
  accuracy <- sum(pred == Y.Test)/length(Y.Test)
  testthat::expect_more_than(accuracy, 0.5)
})
#
testthat::test_that("Creates deepLearnR::TensorFlow.Classifier dnn-default ReLU Model", {
  result = deepLearnR::TensorFlow.Classifier(modelTag="test-dnn-01", X=X.Train, Y=Y.Train,
                                  hiddenUnits=c(10,20,20,10), steps=5000, nnType="dnn")
  testthat::expect_that(result, testthat::matches("test-dnn-01"))
})
#
testthat::test_that("deepLearnR::TensorFlow.Classifier : Predicts Test Dataset w/ dnn-default ReLU Model", {
  pred <- deepLearnR::TensorFlow.predict(modelTag="test-dnn-01", X=X.Test, Y=Y.Test)
  accuracy <- sum(pred == Y.Test)/length(Y.Test)
  testthat::expect_more_than(accuracy, 0.65)
})
#
testthat::test_that("Creates deepLearnR::TensorFlow.Classifier dnn-tanh Model", {
  result = deepLearnR::TensorFlow.Classifier(modelTag="test-dnn-02", X=X.Train, Y=Y.Train,
                                             hiddenUnits=c(10,20,20,10), steps=5000, nnType="dnn", netType="tanh")
  testthat::expect_that(result, testthat::matches("test-dnn-02"))
})
#
testthat::test_that("deepLearnR::TensorFlow.Classifier : Predicts Test Dataset w/ dnn-tanh Model", {
  pred <- deepLearnR::TensorFlow.predict(modelTag="test-dnn-02", X=X.Test, Y=Y.Test)
  accuracy <- sum(pred == Y.Test)/length(Y.Test)
  testthat::expect_more_than(accuracy, 0.65)
})
