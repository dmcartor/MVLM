#' Multivariate Linear Model with Analytic p-values
#'
#' The \code{MVLM} package is used to fit linear models with a multivariate
#' outcome. It utilizes the asymptotic null distribution of the multivariate
#' linear model test statistic to compute p-values (McArtor et al., under
#' review). It therefore alleviates the need to use approximate p-values based
#' Wilks Lambda, Pillai's Trace, the Hotelling-Lawley Trace, and Roy's Greatest
#' Root.
#'
#' @section Usage:
#' To access this package's tutorial, type the following line into the console:
#'
#' \code{vignette("mvlm-vignette")}
#'
#' There is one primary function that comprises this package:
#' \code{vignette('mvlm-vignette')}
#' There is one primary functions that comprise this package:
#' \code{\link{mvlm}}, which regresses a multivariate outcome onto a set of
#' predictors. Standard functions like \code{summary}, \code{fitted},
#' \code{residuals}, and \code{predict} can be called on a \code{mvlm} output
#' object.
#'
#' @references Davies, R. B. (1980). The Distribution of a Linear Combination of
#'  chi-square Random Variables. Journal of the Royal Statistical Society.
#'  Series C (Applied Statistics), 29(3), 323-333.
#'
#'  Duchesne, P., & De Micheaux, P.L. (2010). Computing the distribution of
#'  quadratic forms: Further comparisons between the Liu-Tang-Zhang
#'  approximation and exact methods. Computational Statistics and Data
#'  Analysis, 54(4), 858-862.
#'
#'  McArtor, D. B., Grasman, R. P. P. P., Lubke, G. H., & Bergeman, C. S.
#'  (under review). The asymptotic null distribution of the multivariate linear
#'  model test statistic. Manuscript submitted for publication.
#'
#' @examples
#'data(mvlmdata)
#'Y <- as.matrix(Y.mvlm)
#'mvlm.res <- mvlm(Y ~ Cont + Cat + Ord, data = X.mvlm)
#'summary(mvlm.res)
#'
#' @importFrom CompQuadForm davies
#' @importFrom  parallel mclapply
#'
#' @docType package
#' @name MVLM-package
#' @aliases MVLM
NULL
