# Functions to create Regressor models with TensorFlow
# 3/9/16
#
##
## Begin tdazad code
##
#' Create  Classifier model based on the parameters
#' 
#' @param modelTag Tag for this model - can be referenced in other calls like prediction
#' @param X The X Matrix for training
#' @param y The y Matrix for training
#' @param test_size The division of the dataset into training vs test
#' @param steps Number of epochs for training
#' @param learningRate The learning rate for optimize algorithm
#' @param batchSize Batch Size for the mini batch for optimization algorithms like SGD
#' @export
#' @examples
#' {
#' library(MASS)
#' data(Boston)
#' X <- Boston[,2:14]
#' y <- Boston[,1]
#' 
#' TensorFlowDNNRegressor(modelTag="tfdnnr-01", X=X, y=y, steps=5000)
#' pred <- TensorFlow.regressorEval(modelTag="tfdnnr-01")
#' mse <- rPython::python.get("mse")
#' r2 <- rPython::python.get("r2")
#' }
#'
TensorFlowDNNRegressor <- function(modelTag, X, y, test_size = 0.2, steps=5000, learningRate=0.1, batchSize = 1) {
  # validate parameters
  if (missing(modelTag)) {
    stop("TensorFlowDNNRegressor : Parameter modelTag missing")
  }
  if (missing(X)) {
    stop("TensorFlowDNNRegressor : Parameter X missing")
  }
  if (missing(y)) {
    stop("TensorFlowDNNRegressor: Parameter y missing")
  }
  # initialize and imports
  TensorFlow.init()
  python.exec('
              from sklearn import datasets, cross_validation, metrics
              from sklearn import preprocessing
              ')
  
  #Series of assignments in Python
  python.exec("
              boston = datasets.load_boston()
              X, y = boston.data, boston.target
              ")
  python.assign("modelTag",modelTag)
  python.assign("X", X)
  python.exec('
              from pandas import DataFrame
              X = DataFrame(X)')
  python.assign("y", y)
  python.assign("test_size", test_size)
  python.exec("X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = test_size, random_state=23)")
  python.assign("steps",steps)
  python.assign("learningRate",learningRate)
  python.assign("batchSize",batchSize)
  
  #Perform scaler preprocessing
  python.exec("scaler = preprocessing.StandardScaler()")
  python.exec("X_train = scaler.fit_transform(X_train)")
  
  #Call skflow DNN Regressor and fit it
  python.exec("
              regressor = skflow.TensorFlowDNNRegressor(hidden_units=[10,10],batch_size=batchSize,
              steps=steps, learning_rate=learningRate)
              models[modelTag] = regressor
              regressor.fit(X_train, y_train)
              ")
  #Print model ID
  return(modelTag)
}

##
## End tdazad code
##

##
## Begin tdazad code
##
#' Predict using a model(modelTag) the Yvalues for the X Matrix
#' 
#' @param modelTag Tag for this model - referenced in the model ceate calls like TensorFlow.Classifier
#' @param calculateMSE Yes/No to calculate the MSE
#' @param calculateR2 Yes/No to compute R2 
#' @export
#' 
TensorFlow.regressorEval <- function(modelTag, calculateMSE=TRUE, calculateR2 = TRUE) {
  # validate parameters
  if (missing(modelTag)) {
    stop("TensorFlow.regressorEval : Parameter modelTag missing")
  }
  python.assign("modelTag",modelTag)
  python.exec("
              mdl  = models[modelTag]
              ")
  python.exec("
              y_pred = mdl.predict(scaler.fit_transform(X_test))
              ")
  if (calculateMSE) {
    #Produce MSE
    python.exec("
                print(metrics.mean_squared_error(mdl.predict(scaler.fit_transform(X_test)), y_test))
                ")
  }
  if (calculateR2) {
    #Produce R2
    python.exec("
                print(metrics.r2_score(mdl.predict(scaler.fit_transform(X_test)), y_test))
                ")
  }
  python.exec("
              mse = metrics.mean_squared_error(mdl.predict(scaler.fit_transform(X_test)), y_test)
              r2 = metrics.r2_score(mdl.predict(scaler.fit_transform(X_test)), y_test)
              ")
  y_pred <- python.get("y_pred.tolist()")
  return(y_pred)
  
  }
##
## End tdazad code
##
#