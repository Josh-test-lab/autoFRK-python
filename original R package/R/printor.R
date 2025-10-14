
#'
#' Internal function: print an FRK object
#'
#' @keywords internal
#' @param x An FRK object
#' @param ... Not used directly
#' @method print FRK
#'
print.FRK <- function(x, ...) {
  if (!inherits(x, "FRK")) {
    stop("Invalid object! Please enter an `FRK` object")
  }
  attr(x, "pinfo") <- NULL
  if (!is.null(x$LKobj)) {
    x$LKobj <- x$LKobj$summary
  }
  out <- paste0("a ", NROW(x$G), " by ", NCOL(x$G), " mrts matrix")
  return(print(out))
}

#'
#' Internal function: print an mrts object
#'
#' @keywords internal
#' @param x An mrts object
#' @param ... Not used directly
#' @method print mrts
#'
print.mrts <- function(x, ...) {
  if (!inherits(x, "mrts")) {
    stop("Invalid object! Please enter an `mrts` object")
  }
  
  if (NCOL(x) == 1) {
    return(c(x))
  } else {
    return(x[, 1:NCOL(x)])
  }
}
