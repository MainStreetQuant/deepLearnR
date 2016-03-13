Y <- deepLearnR::titanic.data$Survived
X <- deepLearnR::titanic.data[,c("Age","SibSp","Fare","Pclass")]
X$Age[is.na(X$Age)] <- mean(X$Age,na.rm=TRUE)
set.seed(512)
inTrain <- sample(1:nrow(X), trunc(nrow(X)*0.8))
X.Train <- X[inTrain,]
Y.Train <- Y[inTrain]
X.Test <- X[-inTrain,]
Y.Test <- Y[-inTrain]
deepLearnR::TensorFlow.Classifier(modelTag="tflr-03",X=X.Train,Y=Y.Train,steps=5000)
pred <- deepLearnR::TensorFlow.predict(modelTag="tflr-03",X=X.Test,Y=Y.Test)
accuracy <- sum(pred == Y.Test)/length(Y.Test)
print(accuracy) # Should be ~ 0.6312849
pred <-  deepLearnR::TensorFlow.predict(modelTag="tflr-03",X=X,Y=Y)
accuracy <- sum(pred == Y)/length(Y)
print(accuracy) # Should be ~ 0.6397306