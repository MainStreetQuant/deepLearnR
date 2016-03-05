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
testthat::test_that("Creates deepLearnR::TensorFlow.Classifier Model", {
  result <- deepLearnR::TensorFlow.Classifier(modelTag="test-01", X=X.Train, Y=Y.Train, steps=5000)
  testthat::expect_that(result, testthat::matches("test-01"))
})
#
testthat::test_that("deepLearnR::TensorFlow.Classifier : Predicts Test Dataset", {
  pred <- deepLearnR::TensorFlow.predict(modelTag="test-01", X=X.Test, Y=Y.Test)
  accuracy <- sum(pred == Y.Test)/length(Y.Test)
  testthat::expect_more_than(accuracy, 0.5)
})

