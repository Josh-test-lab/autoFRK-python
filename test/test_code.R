library(autoFRK)

# generate data
set.seed(0)
ns <- 100
nt <- 30
s <- 5

grid1 <- grid2 <- seq(0, 1, l = 30)
grids <- expand.grid(grid1, grid2)
fn <- matrix(0, 900, 2)
fn[, 1] <- cos(sqrt((grids[, 1] - 0)^2 + (grids[, 2] - 1)^2) * pi)
fn[, 2] <- cos(sqrt((grids[, 1] - 0.75)^2 + (grids[, 2] - 0.25)^2) * 2 * pi)

wt <- matrix(0, 2, nt)
for (tt in 1:nt) wt[, tt] <- c(rnorm(1, sd = 5), rnorm(1, sd = 3))
yt <- fn %*% wt
obs <- sample(900, ns)
zt <- yt[obs, ] + matrix(rnorm(ns * nt), ns, nt) * sqrt(s)
X <- grids[obs, ]

# Fit the model
fit <- autoFRK(data = zt, loc = X)
PRED <- predict(fit)

# MSE
options(digits = 15)
res <- PRED$pred.value - zt
squ_res <- res^2
cat("MSE(ALL TIME):", mean(squ_res))

for (i in 1:nt) {
  # mse
  temp <- mean(squ_res[, i])
  plot(PRED$pred.value[, i],
    zt[, i],
    main = paste("MSE(TIME", i, ") =", round(temp, 4)),
    xlab = "Predicted",
    ylab = "Observed"
  )
  abline(0, 1)
}
