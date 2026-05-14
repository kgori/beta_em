# R/em_beta_mixture.R
#
# Beta mixture model fitted by Expectation-Maximisation.
#
# The computationally intensive E-step, M-step and main EM loop are
# implemented in C++ (src/em_beta_mixture.cpp) and called via Rcpp.
# This file provides:
#
#   - Thin R wrappers for the exported C++ helpers so that callers who
#     use the helper functions directly are unaffected by the move to C++.
#   - The per-iteration plotting helper (plot_estimation_step), which must
#     remain in R as it uses R graphics.
#   - estimate_mixture(), the main user-facing function.  When plot = FALSE
#     (the common case) it delegates entirely to estimate_mixture_cpp().
#     When plot = TRUE it steps one iteration at a time from R, calling the
#     C++ E-step and M-step at each iteration and plotting the result.
#   - Beta distribution helpers: beta_ab, beta_ll, beta_mode, beta_mean,
#     beta_boundary, and get_init (thin wrapper over get_init_cpp).

# ---------------------------------------------------------------------------
# Beta distribution helpers
# ---------------------------------------------------------------------------

#' Find a and b such that Mean(Beta(a,b)) = mn and Var(Beta(a,b)) = va
#'
#' @param mn Target mean.
#' @param va Target variance.
#' @return Numeric vector c(a, b).
#' @export
beta_ab <- function(mn, va) {
    a <- ((1 - mn) / va - 1 / mn) * mn^2
    b <- a * (1 / mn - 1)
    c(a, b)
}

#' Calculate log-likelihood of data given Beta(a, b)
#'
#' @param data Numeric vector of observations in [0, 1].
#' @param a    Beta shape parameter a.
#' @param b    Beta shape parameter b.
#' @return Numeric vector of log-densities.
#' @export
beta_ll <- function(data, a, b) {
    dbeta(data, a, b, log = TRUE)
}

#' Find mode of a Beta distribution
#'
#' @param a Beta shape parameter a.
#' @param b Beta shape parameter b.
#' @return Mode value.
#' @export
beta_mode <- function(a, b) {
    if (a > 1 & b > 1) return((a - 1) / (a + b - 2))
    if (a == 1 & b == 1) return(0.5)
    if (a == 1 & b > 1)  return(0)
    if (a > 1 & b == 1)  return(1)
    stop(paste("Beta distribution has no mode: a =", a, "b =", b))
}

#' Find mean of a Beta distribution
#'
#' @param a Beta shape parameter a.
#' @param b Beta shape parameter b.
#' @return Mean value.
#' @export
beta_mean <- function(a, b) {
    if (a < 0 | b < 0) stop(paste("Beta parameters out of range: a =", a, "b =", b))
    a / (a + b)
}

#' Find the boundary between two adjacent Beta components
#'
#' Locates the x value in (mode1, mode2) where the weighted densities of two
#' Beta distributions are equal (i.e. the classification boundary).
#'
#' @param a1   Shape parameter a of component 1.
#' @param b1   Shape parameter b of component 1.
#' @param a2   Shape parameter a of component 2.
#' @param b2   Shape parameter b of component 2.
#' @param p1   Mixing proportion of component 1.
#' @param p2   Mixing proportion of component 2.
#' @param log  Logical; use log-scale computation (default TRUE).
#' @param plot Logical; plot intermediate results (default FALSE).
#' @return Scalar x value of the classification boundary.
#' @export
beta_boundary <- function(a1, b1, a2, b2, p1, p2, log = TRUE, plot = FALSE) {
    mode1 <- tryCatch(beta_mode(a1, b1), error = function(e) {
        warning(paste("Fall back to Beta mean for parameters a =", a1, "b =", b1))
        beta_mean(a1, b1)
    })
    mode2 <- tryCatch(beta_mode(a2, b2), error = function(e) {
        warning(paste("Fall back to Beta mean for parameters a =", a2, "b =", b2))
        beta_mean(a2, b2)
    })
    psum <- p1 + p2
    p1 <- p1 / psum
    p2 <- p2 / psum
    x <- seq(mode1, mode2, length.out = 1001)
    if (log) {
        df <- data.frame(
            dens1 = log(p1) + dbeta(x, a1, b1, log = TRUE),
            dens2 = log(p2) + dbeta(x, a2, b2, log = TRUE)
        )
        y <- apply(df, 1, function(row) {
            log(abs(diff(exp(row - max(row))))) + max(row)
        })
    } else {
        df <- data.frame(
            dens1 = p1 * dbeta(x, a1, b1),
            dens2 = p2 * dbeta(x, a2, b2)
        )
        y <- apply(df, 1, function(row) abs(diff(row)))
    }
    result <- x[which.min(y)]
    if (plot) {
        plot(x, y, type = "l", ylim = c(min(c(y, df$dens1, df$dens2)), max(y)))
        lines(x, df$dens1, lty = 2)
        lines(x, df$dens2, lty = 2)
        abline(v = result, col = "red")
    }
    result
}

# ---------------------------------------------------------------------------
# R wrappers for C++ helper functions
# ---------------------------------------------------------------------------

#' EM E-step: compute posterior component weights for each observation
#'
#' Thin R wrapper around \code{\link{get_weights_cpp}}.
#'
#' @inheritParams get_weights_cpp
#' @return A list with elements \code{weights} (ndata x ncomp matrix) and
#'   \code{ll} (total observed-data log-likelihood).
#' @export
get_weights <- function(data, params, pi) {
    get_weights_cpp(data, params, pi)
}

#' EM M-step: update Beta parameters and mixing proportions from weights
#'
#' Thin R wrapper around \code{\link{get_updated_params_cpp}}.
#'
#' @inheritParams get_updated_params_cpp
#' @return A list with elements \code{params} (ncomp x 2 matrix) and
#'   \code{pi} (numeric vector of mixing proportions).
#' @export
get_updated_params <- function(data, weights) {
    get_updated_params_cpp(data, weights)
}

#' Generate initial parameter estimates for the Beta mixture model
#'
#' Thin R wrapper around \code{\link{get_init_cpp}}.
#'
#' @inheritParams get_init_cpp
#' @return A list with elements \code{params} (k x 2 matrix) and \code{pi}
#'   (length-k vector of initial mixing proportions).
#' @export
get_init <- function(v, k = 3) {
    get_init_cpp(v, k)
}

# ---------------------------------------------------------------------------
# Convergence helpers (used in the plot = TRUE path)
# ---------------------------------------------------------------------------

# Maximum scaled absolute difference between vectors va and vb.
max_relerror <- function(va, vb) {
    if (all(va == vb)) return(0)
    numer <- abs(va - vb)
    denom <- apply(matrix(c(va, vb), ncol = 2), 1, max)
    max(numer / denom)
}

# Return largest relative parameter change across a, b and pi.
get_delta <- function(new_params, old_params, new_pi, old_pi) {
    max(
        max_relerror(old_pi,          new_pi),
        max_relerror(old_params[, 1], new_params[, 1]),
        max_relerror(old_params[, 2], new_params[, 2])
    )
}

# ---------------------------------------------------------------------------
# Per-iteration plot (plot = TRUE path only)
# ---------------------------------------------------------------------------

#' Plot a single iteration of the mixture model estimation procedure
#'
#' Visualises the current state of the EM algorithm: histogram of data with
#' fitted component densities, classification uncertainty, and a parameter
#' table.
#'
#' @importFrom grid      popViewport pushViewport grid.draw
#' @importFrom gridExtra tableGrob ttheme_minimal
#' @importFrom gridBase  baseViewports
#'
#' @param data   Numeric vector of observations.
#' @param params Current ncomp x 2 matrix of Beta parameters.
#' @param pi     Current mixing proportions.
#' @param step   Current iteration number (used in plot title).
#' @export
plot_estimation_step <- function(data, params, pi, step) {
    weights_ll <- get_weights_cpp(data, params, pi)
    weights    <- weights_ll$weights
    ll         <- weights_ll$ll

    x           <- seq(0, 1, length.out = 1001)
    w           <- get_weights_cpp(x, params, pi)$weights
    uncertainty <- 1 - apply(w, 1, max)

    bounds <- c()
    for (i in seq_len(nrow(params) - 1)) {
        bounds <- append(
            bounds,
            beta_boundary(
                params[i,   1], params[i,   2],
                params[i+1, 1], params[i+1, 2],
                pi[i], pi[i+1], log = FALSE
            )
        )
    }

    layout(matrix(c(1, 1, 1, 3,
                    1, 1, 1, 1,
                    2, 2, 2, 2), nrow = 3, byrow = TRUE))
    par(mar = c(-0.1, 4, 4, 2) + 0.1)
    hist(data,
         breaks      = seq(0, 1, length.out = 51),
         border      = "white",
         col         = grDevices::hcl(240, 50, 80),
         probability = TRUE,
         main        = paste("step", step, "ll =", round(ll, 4)),
         xaxt        = "n", xlab = NA)

    dens <- t(t(apply(params, 1, function(r) dbeta(x, r[1], r[2]))) * pi)
    apply(dens, 2, function(col) lines(x, col, lty = 3,
                                        col = grDevices::hcl(240, 80, 50)))
    lines(x, rowSums(dens), col = grDevices::hcl(240, 100, 30), lwd = 2)

    abline(v = bounds, col = "red")

    par(mar = c(5, 4, -0.1, 2) + 0.1)
    plot(x, uncertainty, ylim = c(0, 1), type = "l", bty = "n",
         xlab = "VAF", ylab = "Uncertainty")
    abline(v = bounds, col = "red")

    df      <- data.frame(a = params[, 1], b = params[, 2], pi = pi)
    frame()
    vps     <- gridBase::baseViewports()
    grid::pushViewport(vps$inner, vps$figure, vps$plot)
    mytheme <- gridExtra::ttheme_minimal(
        core    = list(fg_params = list(cex = 0.7)),
        colhead = list(fg_params = list(cex = 0.6)),
        rowhead = list(fg_params = list(cex = 0.6))
    )
    grob <- gridExtra::tableGrob(round(df, 2), theme = mytheme)
    grid::grid.draw(grob)
    grid::popViewport(3)
}

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

#' Fit a Beta mixture model by Expectation-Maximisation
#'
#' Optimises the parameters of a mixture of Beta distributions to fit
#' observations in \[0, 1\] using the EM algorithm.
#'
#' When \code{plot = FALSE} (the default) the entire EM loop runs in C++ via
#' \code{\link{estimate_mixture_cpp}} for maximum speed.  When
#' \code{plot = TRUE} the loop runs in R one iteration at a time, calling the
#' C++ E-step and M-step at each iteration, so that
#' \code{\link{plot_estimation_step}} can be invoked between steps.
#'
#' @param data        Numeric vector of observations in \[0, 1\].
#' @param init_params \code{ncomp x 2} matrix of initial Beta \code{(a, b)}
#'   parameters.  All values must be strictly positive.
#' @param init_pi     Numeric vector of initial mixing proportions (must sum
#'   to 1).
#' @param max_steps   Maximum number of EM iterations (default 1000).
#' @param tol         Convergence tolerance (default 1e-5).
#' @param plot        Logical; if \code{TRUE}, plot the model state after each
#'   iteration to the active graphics device (default FALSE).
#'
#' @return A list with elements:
#'   \describe{
#'     \item{params}{\code{ncomp x 2} matrix of optimised Beta parameters,
#'       rows sorted by ascending component mean.}
#'     \item{pi}{Optimised mixing proportions (sorted).}
#'     \item{usedsteps}{Number of EM iterations performed.}
#'     \item{means}{Mean of each fitted Beta component (sorted ascending).}
#'     \item{classification}{Integer vector giving, for each observation, the
#'       index (1-based) of the most probable component.}
#'     \item{uncertainty}{Numeric vector; \code{1 - max(weight)} per
#'       observation.}
#'     \item{data}{The input data vector.}
#'     \item{ll}{Observed-data log-likelihood at convergence.}
#'     \item{bic}{Bayesian Information Criterion.}
#'   }
#' @export
estimate_mixture <- function(data, init_params, init_pi,
                              max_steps = 1000, tol = 1e-5, plot = FALSE) {
    # --- Input validation (mirrors C++ checks; done here for early, readable errors) ---
    if (!isTRUE(all.equal(sum(init_pi), 1)))
        stop(paste0("init_pi must sum to 1: (", paste(init_pi, collapse = ", "), ")"))
    if (any(init_params <= 0))
        stop("init_params must all be > 0")
    if (nrow(init_params) != length(init_pi))
        stop("Number of rows of init_params must equal length of init_pi")

    # --- Fast path: entire loop in C++ ---
    if (!plot) {
        return(estimate_mixture_cpp(data, init_params, init_pi, max_steps, tol))
    }

    # --- Diagnostic path: R loop with per-iteration plotting ---
    ncomp  <- nrow(init_params)
    params <- init_params
    pi     <- init_pi
    step   <- 0L

    plot_estimation_step(data, params, pi, step)

    for (i in seq_len(max_steps)) {
        params_old <- params
        pi_old     <- pi

        # E-step (C++)
        w_out   <- get_weights_cpp(data, params, pi)
        weights <- w_out$weights

        # M-step (C++)
        p_out  <- get_updated_params_cpp(data, weights)
        params <- p_out$params
        pi     <- p_out$pi

        step  <- step + 1L
        delta <- get_delta(params, params_old, pi, pi_old)
        if (step %% 10 == 0) message(step, "  ", delta)

        plot_estimation_step(data, params, pi, step)

        if (delta < tol) break
    }

    ll  <- get_weights_cpp(data, params, pi)$ll
    bic <- log(length(data)) * ncomp - 2 * ll

    # Sort by ascending component mean
    means  <- params[, 1] / rowSums(params)
    ord    <- order(means)
    means  <- means[ord]
    params <- params[ord, , drop = FALSE]
    pi     <- pi[ord]
    weights <- weights[, ord, drop = FALSE]

    list(
        params         = params,
        pi             = pi,
        usedsteps      = step,
        means          = means,
        classification = apply(weights, 1, which.max),
        uncertainty    = 1 - apply(weights, 1, max),
        data           = data,
        ll             = ll,
        bic            = bic
    )
}
