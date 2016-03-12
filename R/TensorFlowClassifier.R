# Functions to create linear and classifier models with TensorFlow
# 2/28/16
#
# Ref:
# [1] rPython and data in and out of pandas
#     https://statcompute.wordpress.com/2013/10/13/rpython-r-interface-to-python/
# [2] some python code refactored from skflow examples in Tutorials (1,2 & 3) by Illia Polosukhin
#     https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1#.njjgnw8yh
#     https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92#.xxksiy8gg
#     https://medium.com/@ilblackdragon/tensorflow-tutorial-part-3-c5fc0662bc08#.md7qum553
# [3] python code from skflow examples https://github.com/tensorflow/skflow/tree/master/examples
#
# library("rPython")
##
## Begin ksankar code
##
#
# Capability to set & get state sariables
pkgGlobals <- new.env( parent=emptyenv())
#
pkgSetGlobal <- function(name, value) {
  pkgGlobals[[name]] <- value
}
#
pkgGetGlobal <- function(name) {
  pkgGlobals[[name]]
}
#
# Imports
#' @importFrom rPython python.exec python.assign python.get
#
# TensorFlow.init() is private, not exported
#
TensorFlow.init <- function() {
  if (is.null(pkgGetGlobal("tensorFlow.initialized"))) pkgSetGlobal("tensorFlow.initialized",FALSE)
  if (!pkgGetGlobal("tensorFlow.initialized")) {
    print("Initializing ...")
    #
    python.exec("
                import sys
                sys.argv = ['']
                import tensorflow as tf
                import skflow
                import numpy as np
                import random
                import pandas
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score
                from sklearn.utils import check_array
                from sklearn.cross_validation import train_test_split
                #
                models = {}
                ")
    python.exec("ver = tf.__version__")
    tFlow.ver <- python.get("ver")
    pkgSetGlobal("tensorFlow.ver",tFlow.ver)
    print (sprintf("Tensorflow Version = %s",pkgGetGlobal("tensorFlow.ver")))
    pkgSetGlobal("tensorFlow.initialized",TRUE)
  }
}
##
## End ksankar code
##
#
##
## Begin ksankar code
##
#
#' Create  Classifier model based on the parameters
#' 
#' @param modelTag Tag for this model - can be referenced in other calls like prediction
#' @param XTrain The X Matrix for training
#' @param YTrain The Y Matrix for training
#' @param nClasses The number of classes
#' @param miniBatchSize Batch Size for the mini batch for optimization algorithms like SGD
#' @param steps Number of epochs for training
#' @param optimizer The Optimizer algorithm = "SGD", "Adam","Adagrad" (only "SGD" tested, others ignored)
#' @param learningRate The learning rate for optimize algorithm
#' @param hiddenUnits The number and architecture of hidden unit layers for dnn e.g. [10,20,10]
#' @param rnnSize The size of the rnn cell, e.g. size of your word embeddings
#' @param nnType The network type = "linear", "dnn", "rnn"
#'     ("rnn" is not implemented, but included for completeness of the interface & future implementation)
#' @param netType The network type for the final round = "ReLU","tanh"
#' @param cellType The cell type for rnn network = "rnn","gru","lstm" (not implemented, but included for completeness of the interface & future implementation)
#' @export
#' @examples
#' {
#'  Y <- titanic.data$Survived
#'  X <- deepLearnR::titanic.data[,c("Age","SibSp","Fare","Pclass")]
#'  X$Age[is.na(X$Age)] <- mean(X$Age,na.rm=TRUE)
#'  set.seed(512)
#'  inTrain <- sample(1:nrow(X), trunc(nrow(X)*0.8))
#'  X.Train <- X[inTrain,]
#'  Y.Train <- Y[inTrain]
#'  X.Test <- X[-inTrain,]
#'  Y.Test <- Y[-inTrain]
#'  deepLearnR::TensorFlow.Classifier(modelTag="tflr-03",X=X.Train,Y=Y.Train,steps=5000)
#'  pred <- deepLearnR::TensorFlow.predict(modelTag="tflr-03",X=X.Test,Y=Y.Test)
#'  accuracy <- sum(pred == Y.Test)/length(Y.Test)
#'  print(accuracy) # Should be ~ 0.6312849
#'  pred <-  deepLearnR::TensorFlow.predict(modelTag="tflr-03",X=X,Y=Y)
#'  accuracy <- sum(pred == Y)/length(Y)
#'  print(accuracy) # Should be ~ 0.6397306
#' }
#'
#'@seealso \code{\link{TensorFlow.predict}}
TensorFlow.Classifier <- function(modelTag, XTrain, YTrain, nClasses=2, miniBatchSize=128,
                                            steps=500, optimizer="SGD", learningRate=0.05,
                                            hiddenUnits=c(10,20,10), rnnSize=100,
                                            nnType="linear", netType="ReLU", cellType="lstm") {
  # validate parameters
  if (missing(modelTag)) {
    stop("deepLearnR.TensorFlowClassifier : Parameter modelTag missing")
  }
  if (missing(XTrain)) {
    stop("deepLearnR.TensorFlowClassifier : Parameter XTrain missing")
  }
  if (missing(YTrain)) {
    stop("deepLearnR.TensorFlowClassifier : Parameter YTrain missing")
  }
  optimizer <- match.arg(arg = optimizer, choices = c("SGD", "Adam", "Adagrad")) # ignored w/ SGD as default
  nnType <- match.arg(arg = nnType, choices = c("linear", "dnn", "rnn","skit")) # "linear" & "dnn" implemented
  netType <- match.arg(arg = netType, choices = c("ReLU","tanh"))
  cellType <- match.arg(arg = cellType, choices = c("rnn","gru","lstm")) # not implemented
  # initialize and imports
  TensorFlow.init()
  # set of assigns
  python.assign("modelTag",modelTag)
  python.assign("nClasses",nClasses)
  python.assign("miniBatchSize",miniBatchSize)
  python.assign("steps",steps)
  python.assign("learningRate",learningRate)
  python.assign("XTrain",XTrain)
  python.assign("YTrain",YTrain)
  #
  python.exec("X_train = pandas.DataFrame(XTrain)")
  python.exec("y_train = pandas.DataFrame(YTrain)")
  # do work
  python.exec("
              print 'parameters : n_classes = %d batch_size = %d steps = %d learning_rate = %f' % (nClasses, \
                miniBatchSize, steps, learningRate) 
              ")
  if (nnType == "linear") {
    python.exec("
              tflr = skflow.TensorFlowLinearClassifier(n_classes=nClasses,
                        batch_size=miniBatchSize, 
                        steps=steps, learning_rate=learningRate)
              models[modelTag] = tflr
              tflr.fit(X_train, y_train)
              ")
  } else if (nnType == "dnn") {
    python.assign("hiddenUnits", hiddenUnits)
    python.exec("
              print 'dnn parameter : hidden_units = %s' % (hiddenUnits)
              ")
    print (sprintf("dnn parameter : netType=%s",netType))
    if (netType == "ReLU") {
      python.exec("
              tfdnn_r = skflow.TensorFlowDNNClassifier(hidden_units=hiddenUnits,
                n_classes=nClasses,batch_size=miniBatchSize,
                steps=steps, learning_rate=learningRate)
              models[modelTag] = tfdnn_r
              tfdnn_r.fit(X_train, y_train)
              ")
      } else { # tanh
        python.exec("
              def dnn_tanh(X, y):
                layers = skflow.ops.dnn(X, hiddenUnits, tf.tanh)
                return skflow.models.logistic_regression(layers, y)
              tfdnn_t = skflow.TensorFlowEstimator(model_fn=dnn_tanh,
                  n_classes=nClasses,batch_size=miniBatchSize,
                  steps=steps, learning_rate=learningRate)
              models[modelTag] = tfdnn_t
              tfdnn_t.fit(X_train, y_train)
              ")
      }
## We didn't get time to implement & test all combinations fully.
## But we kept the code as a reference for future implemenation 
#   } else if (nnType == "rnn") { # with num_layers=1
#     python.assign("rnnSize",rnnSize)
#     python.assign("cellType",cellType)
#     python.exec("
#               print 'rnn parameter : rnnSize = %d cellType = %s' % (rnnSize, cellType)
#               ")
#     python.exec("
#               tfrnn = skflow.TensorFlowRNNClassifier(rnn_size=rnnSize,
#                 cell_type=cellType, n_classes=nClasses, batch_size=miniBatchSize,
#                 steps=steps, learning_rate=learningRate, bidirectional=False)
#               models[modelTag] = tfrnn
#               tfrnn.fit(X_train, y_train)
#               ")
  } else {
    stop(sprintf("deepLearnR.TensorFlowClassifier : nnType %s not implemented",nnType))
  }
  # set of gets - as required
#   W <- python.get("slope.tolist()")
#   b <- python.get("intercept.tolist()")
#   return(list(W=W,b=b))
  return(modelTag)
}
##
## End ksankar code
##
#
##
## Begin ksankar code
##
#' Predict using a model(modelTag) the Yvalues for the X Matrix
#' 
#' @param modelTag Tag for this model - referenced in the model ceate calls like TensorFlow.Classifier
#' @param XTest The X Matrix for test or prediction
#' @param YTest The Y Matrix for test, to calculate the accuracy
#' @param calculateAccuracy Yes/No to calculate the accuracy from skflow. As a check
#' @export
#' @seealso \code{\link{TensorFlow.Classifier}}
TensorFlow.predict <- function(modelTag, XTest, YTest=NULL, calculateAccuracy=TRUE) {
  # validate parameters
  if (missing(modelTag)) {
    stop("deepLearnR.TensorFlow.predict : Parameter modelTag missing")
  }
  if (missing(XTest)) {
    stop("deepLearnR.TensorFlow.predict : Parameter XTest missing")
  }
  if (calculateAccuracy && is.null(YTest)){
    stop("deepLearnR.TensorFlow.predict : Parameter YTest missing to calculate accuracy")
  }
  python.assign("modelTag",modelTag)
  python.assign("XTest",XTest)
  python.assign("YTest",YTest)
  python.exec("X_test = pandas.DataFrame(XTest)")
  python.exec("y_test = pandas.DataFrame(YTest)")
  python.exec("
              mdl  = models[modelTag]
              y_pred = mdl.predict(X_test)
              ")
  if (calculateAccuracy) {
    python.exec("
              print(accuracy_score(y_pred, y_test))
              ")
  }
  y_pred <- python.get("y_pred.tolist()")
  return(y_pred)
}
##
## End ksankar code
##
#