library(testthat)
library(deepLearnR)

set.seed(123)
x.vals <- seq(1:100)
y.vals <- 0.3 + 0.5 * x.vals

# > lm.fit <- lm(y.vals ~ x.vals)
# > summary(lm.fit)
# 
# Call:
#   lm(formula = y.vals ~ x.vals)
# 
# Residuals:
#   Min         1Q     Median         3Q        Max 
# -1.493e-13 -1.047e-15  2.473e-15  2.894e-15  3.001e-14 
# 
# Coefficients:
#   Estimate Std. Error   t value Pr(>|t|)    
# (Intercept) 3.000e-01  3.149e-15 9.527e+13   <2e-16 ***
#   x.vals    5.000e-01  5.414e-17 9.236e+15   <2e-16 ***

testthat::test_that("Check that TensorFlow model outputs same slope as 'lm'", {
  lm.tf.fit <- TensorFlow.SystemLinReg(X = x.vals, Y = y.vals, epochs = 100000, learning.rate = .00005)
  lm.fit <- lm(y.vals ~ x.vals)
  testthat::expect_equal(lm.tf.fit$slope, expected = 0.5, tolerance = 0.1)
})


