source("fn_cpp.R")
#---generate data and setting
lon = 100 ; lat = 100 ; N = lon*lat ; sigma2 = 0.5 ; n = 100 ; T = 1
bigK = 50 
grid2=seq(-180,176,l=lon)
grid1=seq(-90,87,l=lat)
grids_180 = expand.grid(grid1,grid2)
grids = matrix(0,N,2)
grids[,1]=grids_180$Var1 ; grids[,2]=grids_180$Var2
cset = c(0.01,2) ; sigma_y_set = c(1,2) ; d = 2 ; dd = 3
ccc = 0.1 ; vy = 1
cov.mat2 = cpp_exp(grids, grids, N, N,ccc, vy)
y = mvrnorm(T, seq(0,0,length.out = N), cov.mat2, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)
obs=sample(N,n)
z=y[obs]+rnorm(n,0,sigma2^(1/2))
X=grids[obs,]


# generate the basis functions
# X for control points ; bigK for the number of the basis functions ; grids for the data points
system.time({mrts = mrts_sphere(grids, bigK, grids)$mrts})
dim(mrts)

# # linear model
# # mrlm = function(knot,bigK,obsdata,obsloc,newloc)
# res_lm = mrlm(X,30,z,X,grids)
# 
# # MSPE
# mean((y-res_lm$pred)^2)
# mean((y-res_lm$pred_cal)^2)


# Need some modification because of fields package
# mrmm = function(knot,bigK,obsdata,obsloc,newloc,a,param) a is the taper para
# res_mm = mrmm(X,30,z,X,grids,0.05,c(0.1,0.1,0.1))
# mean((y-res_mm$pred)^2)
# mean((y-res_mm$pred_cal)^2)

