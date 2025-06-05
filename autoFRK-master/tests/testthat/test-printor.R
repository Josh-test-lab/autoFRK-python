set.seed(1234)
tolerance <- 1e-6

mrts_message <- capture_output(print.mrts(mrts(1, 2)), print = TRUE)
test_that("Print mrts", {
  expect_error(print.mrts(1), "Invalid object! Please enter an `mrts` object")
  expect_equal(mrts_message, "[1]   1 NaN")
  expect_equal(sum(print.mrts(mrts(matrix(
    1:10, 2
  ), 6))), 2)
})

FRK_message <-
  capture_output(print.FRK(autoFRK(
    data = rnorm(10),
    loc = 1:10,
    maxK = 3
  )))
test_that("Print FRK", {
  expect_error(print.FRK(1), "Invalid object! Please enter an `FRK` object")
  expect_equal(FRK_message, "[1] \"a 10 by 2 mrts matrix\"")
})
