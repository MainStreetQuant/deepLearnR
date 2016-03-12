library(testthat)
library(deepLearnR)

set.seed(123)
x.vals <- seq(1:100)
y.vals <- 0.3 + 0.5*(rnorm(n = 100, mean = 10, sd = 5) + x.vals)

# > lm.fit <- lm(y.vals ~ x.vals)
# > summary(lm.fit)
# 
# Call:
#   lm(formula = y.vals ~ x.vals)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -6.1339 -1.3809 -0.0866  1.6212  5.2372 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  5.20899    0.46072   11.31   <2e-16 ***
#   x.vals       0.50628    0.00792   63.92   <2e-16 ***

testthat::test_that("Check that TensorFlow model outputs same slope as 'lm'", {
  lm.tf.fit <- TensorFlow.SystemLinReg(X = x.vals, Y = y.vals, epochs = 100000, learning.rate = .00001)
  lm.fit <- lm(y.vals ~ x.vals)
  testthat::expect_equal(lm.tf.fit$slope, expected = 0.506, tolerance = 0.1)
})


