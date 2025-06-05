set.seed(0)
n <- 150
obs <- sample(900, n)
grid1 <- grid2 <- seq(0, 1, l = 30)
grids <- expand.grid(grid1, grid2)
X <- grids[obs,]
G <- mrts(X, 15)
tolerance <- 1e-4

# Test
test_that("mrts", {
  expect_equal(class(G), "mrts")
  expect_equal(mean(G[, 1]), 1)
  expect_lte(max(colSums(G[,-1], 2)), tolerance)
  expect_error(mrts(seq(0, 1, l = 30), 1),
               "k-1 can not be smaller than the number of dimensions!")
  expect_error(mrts(1, 3), "nev must satisfy 1 <= nev <= n - 1, n is the size of matrix")
})
