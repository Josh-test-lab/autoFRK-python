set.seed(0)
n <- 150
s <- 5
grid1 <- grid2 <- seq(0, 1, l = 30)
grids <- expand.grid(grid1, grid2)
fn <- matrix(0, 900, 2)
fn[, 1] <-
  cos(sqrt((grids[, 1] - 0) ^ 2 + (grids[, 2] - 1) ^ 2) * pi)
fn[, 2] <-
  cos(sqrt((grids[, 1] - 0.75) ^ 2 + (grids[, 2] - 0.25) ^ 2) * 2 * pi)

#### single realization simulation example
w <- c(rnorm(1, sd = 5), rnorm(1, sd = 3))
y <- fn %*% w
obs <- sample(900, n)
z <- y[obs] + rnorm(n) * sqrt(s)
X <- grids[obs,]
obsData <- rnorm(n)

# Example1
single_realization_object <- autoFRK(data = z, loc = X, maxK = 15)
yhat_example1 <- predict(single_realization_object, newloc = grids)
yhat_without_newloc_example1 <- predict(single_realization_object)
yhat_se_without_newloc_example1 <-
  predict(single_realization_object, se.report = TRUE)
yhat_se_with_obsData_example1 <-
  predict(single_realization_object,
          obsData = obsData,
          se.report = TRUE)
yhat_se_with_obsData_obsloc_example1 <-
  predict(
    single_realization_object,
    obsloc = X,
    obsData = obsData,
    se.report = TRUE
  )
# Example2
G <- mrts(X, 15)
Gpred <- predict(G, newx = grids)
user_defined_basis_object <- autoFRK(data = z, loc = X, G = G)
yhat_example2 <-
  predict(user_defined_basis_object,
          newloc = grids,
          basis = Gpred)

# Example 3
wt <- matrix(0, 2, 20)
for (tt in 1:20) {
  wt[, tt] <- c(rnorm(1, sd = 5), rnorm(1, sd = 3))
}
yt <- fn %*% wt
obs <- sample(900, n)
zt <- yt[obs,] + matrix(rnorm(n * 20), n, 20) * sqrt(s)
X <- grids[obs,]

zt[1:10, 1] <- NA
multi_realization_object <- autoFRK(
  data = zt,
  loc = X,
  maxK = 15,
  finescale = TRUE
)
yhat_example3 <- predict(multi_realization_object$G, newx = grids)

# Example4
G <- mrts(X, 3)
Gpred <- predict(G, newx = grids)
user_defined_basis_object <- autoFRK(data = z, loc = X, G = G)
yhat_example4 <-
  predict(user_defined_basis_object,
          newloc = grids,
          basis = Gpred)

# Example5
finescale_object <-
  autoFRK(
    data = z,
    loc = X,
    maxK = 15,
    finescale = TRUE
  )
yhat_example5 <- predict(finescale_object, newloc = grids)
yhat_se_example5 <-
  predict(finescale_object, newloc = grids, se.report = TRUE)
yhat_without_newloc_obsloc_example5 <-
  predict(finescale_object, se.report = TRUE)
yhat_with_obsloc_example5 <-
  predict(
    finescale_object,
    obsloc = grids,
    newloc = NULL,
    se.report = TRUE
  )
yhat_with_obsData_example5 <-
  predict(finescale_object, obsData = obsData, se.report = TRUE)
yhat_with_obsData_obsloc_example5 <-
  predict(
    finescale_object,
    obsloc = X,
    obsData = obsData,
    se.report = TRUE
  )

# Example6
z[c(1, 3, 5)] <- NA
missing_value_object <-
  autoFRK(
    data = z,
    loc = X,
    maxK = 15,
    method = "EM"
  )
yhat_example6 <- predict(missing_value_object, newloc = grids)
missing_value_finescale_object <-
  autoFRK(
    data = z,
    loc = X,
    maxK = 15,
    method = "EM",
    finescale = TRUE
  )
yhat_finescale_example6 <-
  predict(missing_value_finescale_object,
          newloc = grids,
          se.report = TRUE)

tolerance <- 1e-4
# Test
test_that("Automatic selection and prediction", {
  expect_lte(abs(mean(yhat_example1$pred.value) + 3.029648), tolerance)
  expect_lte(abs(sum(yhat_example1$pred.value) + 2726.683), tolerance)
  expect_null(yhat_example1$se)
  expect_equal(length(yhat_example1$pred.value), 900)
  expect_lte(abs(mean(
    yhat_without_newloc_example1$pred.value
  ) + 3.022685), tolerance)
  expect_lte(abs(
    mean(yhat_se_without_newloc_example1$pred.value) + 3.022685
  ), tolerance)
  expect_lte(abs(mean(yhat_se_without_newloc_example1$se) - 0.1629801), tolerance)
  expect_lte(abs(
    mean(yhat_se_with_obsData_example1$pred.value) - 0.09285895
  ), tolerance)
  expect_lte(abs(
    mean(yhat_se_with_obsData_obsloc_example1$pred.value) - 0.09285895
  ), tolerance)
})

test_that("User-specified basis function", {
  expect_lte(abs(mean(yhat_example2$pred.value) + 3.024821), tolerance)
  expect_lte(abs(sum(yhat_example2$pred.value) + 2722.339), tolerance)
  expect_null(yhat_example2$se)
  expect_equal(length(yhat_example2$pred.value), 900)
})

test_that("Independent multi-realization", {
  expect_lte(abs(norm(yhat_example3, "F") - 107.6848), tolerance)
  expect_lte(abs(max(yhat_example3) - 3.998665), tolerance)
  expect_true(any("matrix" %in% class(yhat_example3)))
  expect_equal(dim(yhat_example3), c(900, 13))
})

test_that("User-specified basis function with kstar = 0", {
  expect_lte(abs(mean(yhat_example4$pred.value) + 2.9857), tolerance)
  expect_lte(abs(sum(yhat_example4$pred.value) + 2687.1301), tolerance)
  expect_null(yhat_example4$se)
  expect_equal(length(yhat_example4$pred.value), 900)
  expect_equal(predict(G), G)
})

test_that("autoFRK object with finescale", {
  expect_lte(abs(mean(yhat_example5$pred.value) + 4.415521), tolerance)
  expect_lte(abs(sum(yhat_example5$pred.value) + 3973.9687), tolerance)
  expect_null(yhat_example5$se)
  expect_equal(yhat_example5$pred.value, yhat_se_example5$pred.value)
  expect_lte(abs(mean(yhat_se_example5$se) - 0.4575002), tolerance)
  expect_lte(abs(sum(yhat_se_example5$se) - 411.7502), tolerance)
  expect_lte(abs(
    mean(yhat_without_newloc_obsloc_example5$pred.value) + 4.477185
  ), tolerance)
  expect_lte(abs(mean(
    yhat_without_newloc_obsloc_example5$se
  ) - 0.4588948), tolerance)
  expect_lte(abs(mean(yhat_with_obsloc_example5$pred.value) + 4.477185), tolerance)
  expect_lte(abs(mean(yhat_with_obsloc_example5$se) - 0.4588948), tolerance)
  expect_lte(abs(mean(
    yhat_with_obsData_example5$pred.value
  ) - 0.05621855), tolerance)
  expect_lte(abs(mean(yhat_with_obsData_example5$se) - 0.4588948), tolerance)
  expect_lte(abs(
    mean(yhat_with_obsData_obsloc_example5$pred.value) - 0.05621855
  ), tolerance)
  expect_lte(abs(mean(yhat_with_obsData_obsloc_example5$se) - 0.4588948), tolerance)
})

test_that("autoFRK object with missing values", {
  expect_lte(abs(mean(yhat_example6$pred.value) + 2.907614), tolerance)
  expect_lte(abs(sum(yhat_example6$pred.value) + 2616.8529), tolerance)
  expect_lte(abs(mean(yhat_finescale_example6$pred.value) + 3.667199), tolerance)
  expect_lte(abs(sum(yhat_finescale_example6$pred.value) + 3300.47903), tolerance)
  expect_lte(abs(mean(yhat_finescale_example6$se) - 0.4602187), tolerance)
  expect_lte(abs(sum(yhat_finescale_example6$se) - 414.1968), tolerance)
})
