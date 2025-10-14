set.seed(0)
n <- 100
s <- 5
grid1 <- grid2 <- seq(0, 1, l = 30)
location <- expand.grid(grid1, grid2)
obs <- sample(n * s, n)
X <- location[obs,]
z <- z_na <- rnorm(n)
dummy_array <- array(1:n, s)

z_na[1] <- NA
LK_obj <- initializeLKnFRK(
  data = z_na,
  loc = X,
  nlevel = 3,
  n.neighbor = 3,
  nu = 0
)

tolerance <- 1e-4
# Test
test_that("initializeLKnFRK with NA", {
  expect_lte(abs(sum(LK_obj$x) - 77.82759), tolerance <- 1e-4)  
  expect_lte(abs(sum(LK_obj$z) + 6.329831), tolerance <- 1e-4)  
  expect_equal(LK_obj$n, 99)
  expect_lte(abs(mean(LK_obj$alpha) - 0.3333333), tolerance <- 1e-4)  
  expect_true(any(LK_obj$loc == X))
  expect_equal(LK_obj$nlevel, 3)
  expect_equal(LK_obj$pick[1], 2)  
  expect_equal(mean(LK_obj$weights), 1)  
  expect_lte(abs(mean(LK_obj$alpha) - 0.3333333), tolerance <- 1e-4)
})
test_that("NC with different levels", {
  expect_equal(setNC(z, location, 0), 9)  
  expect_equal(setNC(z, location, 1), 10)
  expect_equal(setNC(z, location, 2), 4)
})

test_that("shiftArray with different shifts", {
  expect_equal(shiftArray(dummy_array, c(-1, 0, 1))[1], 2)  
  expect_equal(shiftArray(dummy_array, 1)[1], 5)
  expect_error(shiftArray(dummy_array, 10),
               "shift exceeds array dimensions")
})