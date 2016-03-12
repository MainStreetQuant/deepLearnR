# Functions to create linear regression models with TensorFlow utilizing system 
# calls and JSON serialization
# 3/10/16
#
##
## Begin wvreelan code
##
#' Serialize data to JSON and write to file
#' 
#' @param data data to serialize
#' @param file.name output file name
SerializeData <- function(data, file.name) {
  # TODO - needs additional checks (does file exist, was file updated, etc.)
  
  data.json <- jsonlite::toJSON(data)
  write.check <- write(data.json, file.name)
}

#' Generate a TensorFlow Python script
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
  paste0("learning_rate = ", learning.rate, "\n", 
         "epochs = ", epochs, "\n")

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

#' Run a TensorFlow linear regression
#' 
#' @param X the dependent variables in the model
#' @param Y the independent variable in the model
#' @param epochs number of epochs to use in the model
#' @param learning.rate learning rate to use in the model
#' @export
#' @examples
#' {
#' set.seed(123)
#' x.vals <- seq(1:100)
#' y.vals <- 0.3 + 0.5*(rnorm(n = 100, mean = 10, sd = 5) + x.vals)
#' lm.tf.fit <- TensorFlow.SystemLinReg(X = x.vals, Y = y.vals, epochs = 100000, learning.rate = .5)
#' lm.tf.fit
#' }
#' 
TensorFlow.SystemLinReg <- function(X, Y, epochs = 100000, learning.rate = 0.0001) {
  # TODO - add checks on data types
  
  # Serialize data
  SerializeData(X, 'x_vals.json')
  SerializeData(Y, 'y_vals.json')
  
  # Generate python code 
  # TODO - add passing x and y file names?
  GeneratePythonScript(epochs = epochs, learning.rate = learning.rate)
  
  # Run python script
  tf.script <- 'generated_python_script.py'
  py.command <- 'python'
  tf.out <- system2(py.command, args = tf.script, stdout = TRUE)
  
  # Get serialized output values
  # TODO - move to separate reader function that includes error checking
  warning.flag <- 0
  if(readChar(con = "intercept.json", nchars = 5) == "[NaN]") {
    intercept <- NA
    warning.flag <- warning.flag + 1
  } else {
    intercept <- jsonlite::fromJSON('intercept.json')
  }
  if(readChar(con = "slope.json", nchars = 5) == "[NaN]") {
    slope <- NA
    warning.flag <- warning.flag + 1
  } else {
    slope <- jsonlite::fromJSON('slope.json')
  }
  if (warning.flag > 0) {
    warning("NA(s) returned.  Consider lowering learning rate.")
  }
  list(slope = slope, intercept = intercept)
}
  








