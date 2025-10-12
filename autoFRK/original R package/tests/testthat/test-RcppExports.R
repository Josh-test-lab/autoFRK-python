tolerance <- 1e-6
matrix <- matrix(c(1, 2, 2, 1), 2, 2)
result <- eigenDecompose(matrix)
test_that("Eigen Decomposition", {
  expect_lte(norm(result$vector, "F") - 1.414214, tolerance)
  expect_equal(sum(result$value), 2)
})

test_that("Square Root Matrix", {
  expect_equal(
    sum(getSquareRootMatrix(diag(c(4, 4))) - matrix(c(2, 0, 2, 0), 2, 2)),
    0
  )
})

test_that("Inverse Square Root Matrix", {
  expect_equal(
    sum(
      getInverseSquareRootMatrix(
        diag(c(4, 4)),
        diag(2)
      ) - matrix(c(0.5, 0, 0.5, 0), 2, 2)
    ),
    0
  )
})
