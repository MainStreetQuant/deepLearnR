x.vals <- seq(1:100)
y.vals <- 0.3 + 0.5 * x.vals
lm.tf.fit <- deepLearnR::TensorFlow.SystemLinReg(X = x.vals, Y = y.vals, 
                                     epochs = 100000, learning.rate = .00005)
lm.tf.fit
