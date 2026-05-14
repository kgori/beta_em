// em_beta_mixture.cpp
//
// C++ implementation of Expectation-Maximisation for Beta mixture models.
// Based on the R implementation in cnpipe by Kevin Gori, itself based on:
//   Schröder, Christopher, and Sven Rahmann. 2017. Algorithms for Molecular
//   Biology: AMB 12 (August): 21.
//   https://stackoverflow.com/a/43561339

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Convert mean (mn) and variance (va) to Beta(a, b) parameters via method
// of moments:  a = mn * ( mn*(1-mn)/va - 1 ),  b = a * (1/mn - 1)
static vec beta_ab_cpp(double mn, double va) {
    double a = ((1.0 - mn) / va - 1.0 / mn) * mn * mn;
    double b = a * (1.0 / mn - 1.0);
    return vec({a, b});
}

// Element-wise log-likelihood of data under Beta(a, b)
static vec beta_ll_cpp(const vec& data, double a, double b) {
    int n = data.n_elem;
    vec out(n);
    for (int i = 0; i < n; i++) {
        out(i) = R::dbeta(data(i), a, b, /*give_log=*/1);
    }
    return out;
}

// Numerically stable log-sum-exp for a single row vector
static double log_sum_exp(const rowvec& x) {
    double mx = x.max();
    if (!std::isfinite(mx)) return mx;
    return mx + std::log(sum(exp(x - mx)));
}

// Maximum scaled absolute difference between two vectors:
//   max( |va - vb| / max(|va|, |vb|) )
// Returns 0 if va == vb exactly.
static double max_relerror_cpp(const vec& va, const vec& vb) {
    if (approx_equal(va, vb, "absdiff", 0.0)) return 0.0;
    vec numer = abs(va - vb);
    vec denom(va.n_elem);
    for (uword i = 0; i < va.n_elem; i++) {
        denom(i) = std::max(std::abs(va(i)), std::abs(vb(i)));
    }
    return max(numer / denom);
}

// ---------------------------------------------------------------------------
// Exported: E-step  —  get_weights_cpp
// ---------------------------------------------------------------------------

//' EM E-step: compute posterior component weights for each observation
//'
//' For each data point, compute the probability that it came from each mixture
//' component given the current parameters and mixing proportions.
//'
//' @param data   Numeric vector of observations in \[0, 1\].
//' @param params Numeric matrix with one row per component; columns are
//'   \code{a} and \code{b} (Beta shape parameters).
//' @param pi     Numeric vector of mixture proportions (must sum to 1).
//'
//' @return A list with two elements:
//'   \describe{
//'     \item{weights}{An \code{ndata × ncomp} matrix of posterior weights.}
//'     \item{ll}{The total observed-data log-likelihood.}
//'   }
//' @export
// [[Rcpp::export]]
List get_weights_cpp(const arma::vec& data,
                     const arma::mat& params,
                     const arma::vec& pi) {
    int ndata = data.n_elem;
    int ncomp = params.n_rows;

    if ((int)pi.n_elem != ncomp)
        stop("length(pi) must equal nrow(params)");

    // Build log-likelihood matrix: ll[i, k] = log(pi[k]) + logdbeta(x_i; a_k, b_k)
    mat ll(ndata, ncomp);
    for (int k = 0; k < ncomp; k++) {
        ll.col(k) = std::log(pi(k)) + beta_ll_cpp(data, params(k, 0), params(k, 1));
    }

    // Row-wise log-sum-exp for normalisation and total ll
    vec row_lse(ndata);
    for (int i = 0; i < ndata; i++) {
        row_lse(i) = log_sum_exp(ll.row(i));
    }
    double total_ll = sum(row_lse);

    // Weights = exp(ll - row_lse)
    mat weights(ndata, ncomp);
    for (int k = 0; k < ncomp; k++) {
        weights.col(k) = exp(ll.col(k) - row_lse);
    }

    // Fix NaN / Inf rows: observations that caused numerical under/overflow.
    // Assign such points to the leftmost component if x < 0.5, otherwise to
    // the rightmost component (mirrors the R implementation).
    for (int i = 0; i < ndata; i++) {
        bool bad = false;
        for (int k = 0; k < ncomp; k++) {
            if (!std::isfinite(weights(i, k))) { bad = true; break; }
        }
        if (bad) {
            weights.row(i).zeros();
            if (data(i) < 0.5) {
                weights(i, 0) = 1.0;
            } else {
                weights(i, ncomp - 1) = 1.0;
            }
        }
    }

    return List::create(
        Named("weights") = wrap(weights),
        Named("ll")      = total_ll
    );
}

// ---------------------------------------------------------------------------
// Exported: M-step  —  get_updated_params_cpp
// ---------------------------------------------------------------------------

//' EM M-step: update Beta parameters and mixing proportions from weights
//'
//' Given the current posterior weights, compute new estimates of the Beta
//' shape parameters \code{(a, b)} and mixture proportions \code{pi} for each
//' component via moment matching.
//'
//' @param data    Numeric vector of observations in \[0, 1\].
//' @param weights Numeric matrix of posterior weights (\code{ndata × ncomp}),
//'   as returned by \code{get_weights_cpp}.
//'
//' @return A list with two elements:
//'   \describe{
//'     \item{params}{Updated \code{ncomp × 2} matrix of Beta parameters.}
//'     \item{pi}{Updated numeric vector of mixture proportions.}
//'   }
//' @export
// [[Rcpp::export]]
List get_updated_params_cpp(const arma::vec& data,
                             const arma::mat& weights) {
    int ncomp = weights.n_cols;
    int ndata  = weights.n_rows;

    vec col_sums = sum(weights, 0).t();   // ncomp
    vec new_pi   = col_sums / ndata;       // colMeans(weights)

    // Weighted means: mean_k = sum(x_i * w_ik) / sum(w_ik)
    vec means(ncomp);
    for (int k = 0; k < ncomp; k++) {
        means(k) = dot(data, weights.col(k)) / col_sums(k);
    }

    // Weighted variances: var_k = sum((x_i - mean_k)^2 * w_ik) / sum(w_ik)
    vec vars(ncomp);
    for (int k = 0; k < ncomp; k++) {
        vec diff = data - means(k);
        vars(k) = dot(diff % diff, weights.col(k)) / col_sums(k);
    }

    // Convert (mean, var) -> (a, b) for each component
    mat new_params(ncomp, 2);
    for (int k = 0; k < ncomp; k++) {
        vec ab = beta_ab_cpp(means(k), vars(k));
        new_params(k, 0) = ab(0);
        new_params(k, 1) = ab(1);
    }

    return List::create(
        Named("params") = wrap(new_params),
        Named("pi")     = wrap(new_pi)
    );
}

// ---------------------------------------------------------------------------
// Exported: Initialisation  —  get_init_cpp
// ---------------------------------------------------------------------------

//' Generate initial parameter estimates for the Beta mixture model
//'
//' Divides \code{v} into \code{k} equal-width bins over \[0, 1\] and estimates
//' initial Beta parameters and mixing proportions for each bin via method of
//' moments.
//'
//' @param v Numeric vector of observations in \[0, 1\].
//' @param k Number of mixture components (default 3).
//'
//' @return A list with two elements:
//'   \describe{
//'     \item{params}{\code{k × 2} matrix of initial Beta parameters.}
//'     \item{pi}{Length-\code{k} vector of initial mixing proportions.}
//'   }
//' @export
// [[Rcpp::export]]
List get_init_cpp(const arma::vec& v, int k = 3) {
    mat params(k, 2, fill::zeros);
    vec pi_vec(k, fill::zeros);

    for (int i = 0; i < k; i++) {
        double lo = (double)i       / k;
        double hi = (double)(i + 1) / k;

        // Collect data points in (lo, hi] (first bin includes 0)
        vec bin;
        for (uword j = 0; j < v.n_elem; j++) {
            if (v(j) > lo && v(j) <= hi) {
                bin.resize(bin.n_elem + 1);
                bin(bin.n_elem - 1) = v(j);
            }
        }
        // Edge: include exactly 0 in the first bin
        if (i == 0) {
            for (uword j = 0; j < v.n_elem; j++) {
                if (v(j) == 0.0) {
                    bin.resize(bin.n_elem + 1);
                    bin(bin.n_elem - 1) = v(j);
                }
            }
        }

        double mn, va;
        if (bin.n_elem >= 2) {
            mn = mean(bin);
            va = var(bin);
        } else if (bin.n_elem == 1) {
            mn = bin(0);
            va = 1e-6;          // tiny variance to avoid degenerate parameters
        } else {
            // Empty bin: use bin midpoint and a small variance
            mn = (lo + hi) / 2.0;
            va = 1e-6;
        }

        // Guard against mean = 0 or 1 (beta_ab undefined there)
        mn = std::max(1e-4, std::min(1.0 - 1e-4, mn));
        // Guard against variance >= mn*(1-mn) (would give a <= 0)
        double max_va = mn * (1.0 - mn) - 1e-6;
        if (max_va <= 0.0) max_va = 1e-6;
        va = std::max(1e-8, std::min(max_va, va));

        vec ab = beta_ab_cpp(mn, va);
        params(i, 0) = ab(0);
        params(i, 1) = ab(1);
        pi_vec(i) = (double)bin.n_elem;
    }

    // Normalise pi
    double pi_sum = sum(pi_vec);
    if (pi_sum == 0.0) {
        pi_vec.fill(1.0 / k);
    } else {
        pi_vec /= pi_sum;
    }

    // Clamp params to a minimum of 0.01 (mirrors R: params[params < 0.01] <- 0.01)
    params.transform([](double x) { return x < 0.01 ? 0.01 : x; });

    return List::create(
        Named("params") = wrap(params),
        Named("pi")     = wrap(pi_vec)
    );
}

// ---------------------------------------------------------------------------
// Exported: Full EM  —  estimate_mixture_cpp
// ---------------------------------------------------------------------------

//' Fit a Beta mixture model by Expectation-Maximisation
//'
//' Runs the EM algorithm to optimise the fit of a mixture of \code{ncomp}
//' Beta distributions to observations in \[0, 1\]. The entire iterative loop
//' executes in C++ for maximum speed.
//'
//' @param data        Numeric vector of observations in \[0, 1\].
//' @param init_params \code{ncomp × 2} matrix of initial Beta \code{(a, b)}
//'   parameters; all values must be strictly positive.
//' @param init_pi     Numeric vector of initial mixing proportions; must sum
//'   to 1 and have length \code{ncomp}.
//' @param max_steps   Maximum number of EM iterations (default 1000).
//' @param tol         Convergence tolerance: stop when the maximum relative
//'   parameter change across all \code{a}, \code{b} and \code{pi} values is
//'   below this threshold (default 1e-5).
//'
//' @return A list with elements:
//'   \describe{
//'     \item{params}{\code{ncomp × 2} matrix of optimised Beta parameters,
//'       rows sorted by increasing component mean.}
//'     \item{pi}{Optimised mixing proportions (sorted).}
//'     \item{usedsteps}{Number of EM iterations performed.}
//'     \item{means}{Mean of each fitted Beta component (sorted ascending).}
//'     \item{classification}{Integer vector; for each observation, the index
//'       of the most probable component (1-based).}
//'     \item{uncertainty}{Numeric vector; \code{1 - max(weight)} for each
//'       observation.}
//'     \item{data}{The input data vector (returned for convenience).}
//'     \item{ll}{Observed-data log-likelihood at convergence.}
//'     \item{bic}{Bayesian Information Criterion: \code{log(n)*ncomp - 2*ll}.}
//'   }
//' @export
// [[Rcpp::export]]
List estimate_mixture_cpp(const arma::vec& data,
                           const arma::mat& init_params,
                           const arma::vec& init_pi,
                           int    max_steps = 1000,
                           double tol       = 1e-5) {
    int n     = data.n_elem;
    int ncomp = init_params.n_rows;

    // --- Input validation ---
    if ((int)init_pi.n_elem != ncomp)
        stop("length(init_pi) must equal nrow(init_params)");

    double pi_sum = sum(init_pi);
    if (std::abs(pi_sum - 1.0) > 1e-6)
        stop("init_pi must sum to 1");

    if (init_params.min() <= 0.0)
        stop("init_params must all be > 0");

    // --- Initialise ---
    mat  params = init_params;
    vec  pi     = init_pi;
    mat  weights(n, ncomp);
    double ll = -arma::datum::inf;

    // --- EM loop ---
    int step = 0;
    for (int iter = 0; iter < max_steps; iter++) {
        mat  params_old = params;
        vec  pi_old     = pi;

        // E-step
        List w_out  = get_weights_cpp(data, params, pi);
        weights     = as<mat>(w_out["weights"]);
        ll          = as<double>(w_out["ll"]);

        // M-step
        List p_out  = get_updated_params_cpp(data, weights);
        mat  new_params = as<mat>(p_out["params"]);
        vec  new_pi     = as<vec>(p_out["pi"]);

        step++;

        // Convergence check: max relative error across a, b and pi
        double delta = std::max({
            max_relerror_cpp(params_old.col(0), new_params.col(0)),
            max_relerror_cpp(params_old.col(1), new_params.col(1)),
            max_relerror_cpp(pi_old,            new_pi)
        });

        params = new_params;
        pi     = new_pi;

        if (step % 10 == 0) {
            Rcpp::Rcout << "step " << step << "  delta " << delta << "\n";
        }

        if (delta < tol) break;
    }

    // BIC = log(n) * ncomp - 2 * ll
    double bic = std::log((double)n) * ncomp - 2.0 * ll;

    // Component means = a / (a + b); sort by ascending mean
    vec means(ncomp);
    for (int k = 0; k < ncomp; k++) {
        means(k) = params(k, 0) / (params(k, 0) + params(k, 1));
    }
    uvec order = sort_index(means);        // ascending order
    means  = means(order);
    params = params.rows(order);
    pi     = pi(order);
    weights = weights.cols(order);

    // Classification (1-based) and uncertainty
    uvec classification(n);
    vec  uncertainty(n);
    for (int i = 0; i < n; i++) {
        uword  best_k = weights.row(i).index_max();
        double max_w  = weights(i, best_k);
        classification(i) = best_k + 1;    // R is 1-indexed
        uncertainty(i)    = 1.0 - max_w;
    }

    return List::create(
        Named("params")         = wrap(params),
        Named("pi")             = wrap(pi),
        Named("usedsteps")      = step,
        Named("means")          = wrap(means),
        Named("classification") = wrap(conv_to<ivec>::from(classification)),
        Named("uncertainty")    = wrap(uncertainty),
        Named("data")           = wrap(data),
        Named("ll")             = ll,
        Named("bic")            = bic
    );
}
