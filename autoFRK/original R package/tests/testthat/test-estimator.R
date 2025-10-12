set.seed(1234)
n <- 150
s <- 5
grid1 <- grid2 <- seq(0, 1, l = 30)
grids <- expand.grid(grid1, grid2)
Fk <- matrix(0, 900, 2)
Fk[, 1] <- cos(sqrt((grids[, 1] - 0) ^ 2 + (grids[, 2] - 1) ^ 2) * pi)
Fk[, 2] <-
  cos(sqrt((grids[, 1] - 0.75) ^ 2 + (grids[, 2] - 0.25) ^ 2) * 2 * pi)
w <- matrix(c(rnorm(2, sd = 5), rnorm(2, sd = 3)), 2, 2)
y <- Fk %*% w
obs <- sample(900, n)
epsilon <- rexp(n) * sqrt(s)
data <- y[obs] + epsilon

cMLEsp_result <- cMLEsp(Fk[obs,], data, diag(epsilon), TRUE)
em0miss_result <- EM0miss(Fk[obs,], data, diag(epsilon), 10, 1e-4)
em0miss_result_message <- capture_output(
  EM0miss(Fk[obs,], data, diag(epsilon), 10, 1e-4), print = TRUE)
em0miss_w_result <- EM0miss(Fk[obs,], data, diag(epsilon), 10, 1e-4, wSave = TRUE)

missing_data <- data
missing_data[1:10] <- NA
indeMLE_na_result <- indeMLE(missing_data, Fk[obs,])

set.seed(1234)
test_matrix <- matrix(rnorm(25), 5, 5)
test_matrix <- (test_matrix + t(test_matrix)) / 2
mle1 <- cMLE(Fk,
             4,
             5,
             diag(1, 5),
             test_matrix,
             wSave = TRUE)
mle2 <- cMLE(Fk,
             4,
             5,
             diag(1, 5),
             test_matrix,
             wSave = FALSE,
             onlylogLike = FALSE,)
mle3 <- cMLE(Fk,
             4,
             5,
             diag(1, 5),
             test_matrix,
             wSave = TRUE,
             s = 10)

test_cmle_object <-
  cMLEimat(Fk[obs], data, s, FALSE, diag(150), onlylogLike = FALSE)

tolerance <- 1e-6
# Test
test_that("cMLEsp", {
  expect_lte(abs(norm(cMLEsp_result$M, "F") - 46.896965660), tolerance)
  expect_lte(abs(cMLEsp_result$s - 2.088664), tolerance)
  expect_lte(abs(cMLEsp_result$negloglik - 592.7296927), tolerance)
  expect_lte(abs(sum(cMLEsp_result$w) + 436.4577698), tolerance)
  expect_lte(abs(norm(cMLEsp_result$V, "F") - 3890.7169745), tolerance)
})

test_that("cMLE", {
  expect_lte(abs(mle1$v - 0.003904234), tolerance)
  expect_lte(abs(norm(mle1$M, "F") - 1.373454), tolerance)
  expect_equal(mle1$s, 0)
  expect_lte(abs(mle1$negloglik--9710.933343), tolerance)
  expect_lte(abs(norm(mle1$L, "F") - 27.275983), tolerance)
  expect_null(mle2$L)
  expect_true(all(mle3$L == 0))
})

test_that("cMLEimat", {
  expect_lte(abs(test_cmle_object$M[1] - 0), tolerance)
  expect_lte(abs(test_cmle_object$v - 35.089146), tolerance)
  expect_lte(abs(test_cmle_object$negloglik - 979.347403), tolerance)
})

test_that("EM0miss", {
  expect_lte(abs(sum(em0miss_result$M) - 27.0142), tolerance)
  expect_lte(abs(em0miss_result$s - 2.095243), tolerance)
  expect_lte(abs(em0miss_result$negloglik - 593.1121768), tolerance)
  expect_equal(em0miss_result_message, "Number of iteration:  3 \n$M\n      [,1]      [,2]\n 44.680115 -9.944163\n -9.944163  2.222411\n\n$s\n[1] 2.095243\n\n$negloglik\n[1] 593.1122\n")
  expect_lte(abs(sum(em0miss_w_result$w) + 5.195894), tolerance)
  expect_lte(abs(sum(em0miss_w_result$V) - 0.01688807), tolerance)
})

test_that("indeMLE", {
  expect_lte(abs(indeMLE_na_result$negloglik - 732.866333), tolerance)
})
