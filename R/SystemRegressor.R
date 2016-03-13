# Functions to create linear regression models with TensorFlow utilizing system 
# calls and JSON serialization, currently at proof-of-concept stage, works for
# 1-D independent variable vector.  Next steps are generalization to multi-dimensional
# regression and simple neural networks.  Further developments occur almost
# entirly in the python coding and will require straightforward adjustments to the R 
# framework developed herein.
# 
# Last update:  3/12/16
#
##
## Begin wvreelan code
##
# Imports
# Was using jsonlite here, but this causes a note on R CMD INSTALL as rPython
# already relies on RJSONIO
#
#' Serialize data to JSON and write to file, internal function used by deepLearnR
#' @importFrom RJSONIO fromJSON toJSON
#' @param data data to serialize
#' @param file.name output file name
 
SerializeTFData <- function(data, file.name) {
  # If file already exists, get timestamp, else set timestamp to date in the past
  if(file.exists(file.name)) {
    initial.file.timestamp <- file.mtime(file.name)
  } else {
    initial.file.timestamp <- as.POSIXct("2000-01-01")
  }

  # Serialize to JSON
  data.json <- RJSONIO::toJSON(data)
  
  # Try writing data
  write.result <- tryCatch({
    write(data.json, file.name)
  }, error = function(err) {
    stop('Failed to save JSON data.  Check file permissions.')
  })

  # Check that timestamp was updated
  if(file.exists(file.name)) {
    if(initial.file.timestamp > file.mtime(file.name)) {
      warning('JSON file exists but does not appear to be newer than a previous file of the same name.')
    }
  } else {
    stop('JSON file does not exist after attempting to save.')
  }
}

#' Get serialized data values back following python execution, internal function used by deepLearnR
#' @param file.name file name from which to read data

GetSerializedTFData <- function(file.name) {
  # Check if file exists
  if(!file.exists(file.name)) {
    warning(paste0(file.name, " does not exist.  Returning NA."))
    NA
  } else {
    # Check if [Nan] was returned from python
    if(readChar(con = file.name, nchars = 5) == "[NaN]") {
      val <- NA
      warning(paste0("NA returned for ", file.name, ". Try lowering learning rate."))
    } else {
      val <- RJSONIO::fromJSON(file.name)
    }
    val
  } 
}

#' Generate a TensorFlow Python script, internal function used by deepLearnR
#' 
#' @param epochs number of epochs to use in the model
#' @param learning.rate learning rate to use in the model
GeneratePythonScript <- function(epochs = 100000, learning.rate = 0.0001) {
python.init <- 
"#python
# Initialize
import sys
import os
import json
sys.argv = ['']
import tensorflow as tf
import numpy as np\n"

# Set passed parameters for use below
python.set.parms <- 
  paste0("learning_rate = ", format(learning.rate, scientific=F), "\n", 
         "epochs = ", format(epochs, scientific=F), "\n")

python.body <- 
"# Read JSON files
with open('x_vals.json') as json_file:
  x_data = json.load(json_file)

with open('y_vals.json') as json_file:
  y_data = json.load(json_file)

# Run model
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for step in xrange(epochs):
  sess.run(train)
  if step % 10000 == 0:
    print(step, sess.run(W), sess.run(b))
    slope = sess.run(W)
    intercept = sess.run(b)

# Serialize results for importing back to R
with open('slope.json', 'w') as outfile:
  json.dump(slope.tolist(), outfile)

with open('intercept.json', 'w') as outfile:
  json.dump(intercept.tolist(), outfile)
"

# Glue code together
python.code <- paste0(python.init, python.set.parms, python.body)

# Write script to file
script.file <- file("generated_python_script.py")
writeLines(python.code, script.file)
close(script.file)
}

#' TensoFlow linear regression implementation
#' @param X the dependent variables in the model (currently only supports 1-D numeric vector)
#' @param Y the independent variable in the model
#' @param epochs number of epochs to use in the model
#' @param learning.rate learning rate to use in the model
#' @export
#' @examples
#' {
#' x.vals <- seq(1:100)
#' y.vals <- 0.3 + 0.5 * x.vals
#' lm.tf.fit <- deepLearnR::TensorFlow.SystemLinReg(X = x.vals, Y = y.vals, 
#'                           epochs = 100000, learning.rate = .00005)
#' lm.tf.fit
#' }
#' 
TensorFlow.SystemLinReg <- function(X, Y, epochs = 100000, learning.rate = 0.0001) {
  # Check that data are numeric vectors and hyperparameters are numeric before starting
  if(!(is.numeric(X) & is.numeric(Y)
       & is.vector(X) & is.vector(Y)
       & is.numeric(epochs) & is.numeric(learning.rate))) {
    print('Error: input X and Y must be numeric vectors.  Hyperparameters must be numeric values.  Aborting regression.')
    return(NA)
  } else {
    # Serialize data
    SerializeTFData(X, 'x_vals.json')
    SerializeTFData(Y, 'y_vals.json')
    
    # Generate python code 
    # possible TODO - pass file names here instead?
    GeneratePythonScript(epochs = epochs, learning.rate = learning.rate)
    
    # Run python script
    tf.script <- 'generated_python_script.py'
    py.command <- 'python'
    tf.out <- system2(py.command, args = tf.script, stdout = TRUE)
    
    # Get serialized output values
    slope <- GetSerializedTFData(file.name = 'slope.json')
    intercept <- GetSerializedTFData(file.name = 'intercept.json')

    list(slope = slope, intercept = intercept)
  }
}
  




