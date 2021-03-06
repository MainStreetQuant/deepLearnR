#' Deep Learnning interface to TensorFlow from R
#' Leverage the distributed multicore capabilites of TensorFlow
#'
#' Functions to create deepLearning architectures and associated datasets. 
#'  Requires tensorFlow 0.7.0 and rPython installed.
#'  Works with default python, not anaconda python installations.
#'  See examples for functions
#'     \code{\link{TensorFlow.Classifier}}, \code{\link{TensorFlow.predict}},
#'     \code{\link{TensorFlowDNNRegressor}}, \code{\link{TensorFlow.regressorEval}} and
#'     \code{\link{TensorFlow.SystemLinReg}}.
#' @seealso \code{\link{TensorFlow.Classifier}}
#' @seealso \code{\link{TensorFlow.predict}}
#' @seealso \code{\link{TensorFlowDNNRegressor}}
#' @seealso \code{\link{TensorFlow.regressorEval}}
#' @seealso \code{\link{TensorFlow.SystemLinReg}}
#' @docType package
#' @references [1] rPython and data in and out of pandas \url{https://statcompute.wordpress.com/2013/10/13/rpython-r-interface-to-python/}
#' @references [2] some python code refactored from skflow examples in Tutorials (1,2 & 3) by Illia Polosukhin
#' @references      \url{https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1#.njjgnw8yh}
#' @references      \url{https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92#.xxksiy8gg}
#' @references      \url{https://medium.com/@ilblackdragon/tensorflow-tutorial-part-3-c5fc0662bc08#.md7qum553}
#' @references [3] python code from skflow examples \url{https://github.com/tensorflow/skflow/tree/master/examples}
#' @name deepLearnR
NULL