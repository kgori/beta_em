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
// Internal scalar helpers
// ---------------------------------------------------------------------------

// Fix 1: return std::pair<double,double> instead of vec to avoid a 2-element
// heap allocation at every call site.
//
// Convert mean (mn) and variance (va) to Beta(a, b) parameters via method
// of moments:  a = mn * ( mn*(1-mn)/va - 1 ),  b = a * (1/mn - 1)
static std::pair<double,double> beta_ab_cpp(double mn, double va) {
    double a = ((1.0 - mn) / va - 1.0 / mn) * mn * mn;
    double b = a * (1.0 / mn - 1.0);
    return {a, b};
}

// Clamp a scalar to [lo, hi]
static inline double clamp_val(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
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

// wrap(arma::vec) produces a dim = c(n, 1) R matrix, not a plain vector.
// Use this helper for all pi returns so callers always receive a 1-D vector.
static inline Rcpp::NumericVector to_rvec(const arma::vec& v) {
    return Rcpp::NumericVector(v.begin(), v.end());
}

// ---------------------------------------------------------------------------
// Fix 6 + Fix 4 + Fix 5: Internal E-step
//
// Writes posterior weights and total log-likelihood directly into caller-
// supplied arma objects — no Rcpp::List packing/unpacking, no SEXP copies.
//
// Fix 4: log-likelihoods written column-by-column (no intermediate vec of
//        length n per component; column-major sequential writes).
//
// Fix 5: row-wise log-sum-exp computed via two column-major passes, avoiding
//        both the rowvec temporaries and the stride-n row accesses that the
//        old log_sum_exp(ll.row(i)) loop caused.
// ---------------------------------------------------------------------------
static void e_step_internal(const vec& data,
                              const mat& params,
                              const vec& pi,
                              mat& weights,
                              double& ll_out) {
    const int ndata = data.n_elem;
    const int ncomp = params.n_rows;

    // Fix 4: build ll column-by-column (column-major, no intermediate vec)
    mat ll(ndata, ncomp);
    for (int k = 0; k < ncomp; k++) {
        double log_pi_k = std::log(pi(k));
        double a = params(k, 0), b = params(k, 1);
        for (int i = 0; i < ndata; i++) {
            ll(i, k) = log_pi_k + R::dbeta(data(i), a, b, /*give_log=*/1);
        }
    }

    // Fix 5: row-wise log-sum-exp via two column-major passes.
    // Pass 1: row-wise maximum
    vec row_max(ndata);
    row_max.fill(-arma::datum::inf);
    for (int k = 0; k < ncomp; k++) {
        for (int i = 0; i < ndata; i++) {
            if (ll(i, k) > row_max(i)) row_max(i) = ll(i, k);
        }
    }
    // Pass 2: accumulate exp(ll[i,k] - row_max[i]) into per-row sums
    vec row_sum_exp(ndata, fill::zeros);
    for (int k = 0; k < ncomp; k++) {
        for (int i = 0; i < ndata; i++) {
            if (std::isfinite(row_max(i)))
                row_sum_exp(i) += std::exp(ll(i, k) - row_max(i));
        }
    }
    // row_lse[i] = row_max[i] + log(row_sum_exp[i])
    double total_ll = 0.0;
    vec row_lse(ndata);
    for (int i = 0; i < ndata; i++) {
        row_lse(i) = std::isfinite(row_max(i))
                         ? row_max(i) + std::log(row_sum_exp(i))
                         : row_max(i);   // propagate -inf / NaN
        total_ll += row_lse(i);
    }
    ll_out = total_ll;

    // Weights = exp(ll[i,k] - row_lse[i]), column-by-column (cache-friendly)
    for (int k = 0; k < ncomp; k++) {
        for (int i = 0; i < ndata; i++) {
            weights(i, k) = std::exp(ll(i, k) - row_lse(i));
        }
    }

    // Fix bad rows (NaN/Inf from numerical under/overflow): assign point to
    // leftmost component if x < 0.5, rightmost otherwise (mirrors R code).
    for (int i = 0; i < ndata; i++) {
        bool bad = false;
        for (int k = 0; k < ncomp; k++) {
            if (!std::isfinite(weights(i, k))) { bad = true; break; }
        }
        if (bad) {
            for (int k = 0; k < ncomp; k++) weights(i, k) = 0.0;
            weights(i, data(i) < 0.5 ? 0 : ncomp - 1) = 1.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Fix 6 + Fix 2: Internal M-step
//
// Updates params and pi without any Rcpp::List round-trips.
//
// Fix 2: weighted variance uses a scalar accumulator loop (column-major access
//        to weights.col(k)), eliminating the two n-element heap allocations
//        from the old  vec diff = data - means(k);  vars(k) = dot(diff % diff, ...)
// ---------------------------------------------------------------------------
static void m_step_internal(const vec& data,
                              const mat& weights,
                              mat& params,
                              vec& pi) {
    const int ncomp = weights.n_cols;
    const int ndata  = weights.n_rows;

    vec col_sums = sum(weights, 0).t();   // length ncomp
    pi           = col_sums / ndata;       // colMeans(weights)

    // Weighted means: mean_k = dot(data, weights[:,k]) / col_sums[k]
    vec means(ncomp);
    for (int k = 0; k < ncomp; k++) {
        means(k) = dot(data, weights.col(k)) / col_sums(k);
    }

    // Fix 2: weighted variance via scalar accumulator (column-major access).
    // var_k = sum_i (x_i - mean_k)^2 * w_ik  /  sum_i w_ik
    params.set_size(ncomp, 2);
    for (int k = 0; k < ncomp; k++) {
        double mk   = means(k);
        double cs   = col_sums(k);
        double wvar = 0.0;
        for (int i = 0; i < ndata; i++) {
            double d = data(i) - mk;
            wvar += d * d * weights(i, k);
        }
        wvar /= cs;
        auto ab = beta_ab_cpp(mk, wvar);
        params(k, 0) = ab.first;
        params(k, 1) = ab.second;
    }
}

// ---------------------------------------------------------------------------
// Fix 6 + Fix 3: Internal constrained M-step
//
// Updates params and pi for the symmetric 3-component model without any
// Rcpp::List round-trips.
//
// Fix 3: the old code called join_cols twice (creating 2n-element temporaries)
//        and computed arma diff vectors.  Replaced with scalar accumulator
//        loops throughout.  The weight sums and outer weighted mean are fused
//        into a single pass over the data.
// ---------------------------------------------------------------------------
static void m_step_constrained_internal(const vec& data,
                                          const mat& weights,
                                          mat& params,
                                          vec& pi) {
    const int n = data.n_elem;

    // Single pass: component weight sums + outer augmented weighted mean
    // augmented outer dataset = { (x_i, w1_i), (1-x_i, w3_i) }
    double sum_w1 = 0.0, sum_w2 = 0.0, sum_w3 = 0.0;
    double s1_outer = 0.0;
    for (int i = 0; i < n; i++) {
        double xi = data(i);
        double w1 = weights(i, 0), w2 = weights(i, 1), w3 = weights(i, 2);
        sum_w1   += w1;
        sum_w2   += w2;
        sum_w3   += w3;
        s1_outer += xi * w1 + (1.0 - xi) * w3;
    }
    double total_ow = sum_w1 + sum_w3;

    // --- Mixing proportions ---
    double x_new = total_ow > 0.0 ? total_ow / (2.0 * n) : 1.0 / 3.0;
    double y_new = sum_w2 / n;
    x_new = clamp_val(x_new, 1e-6, 0.5 - 1e-6);
    y_new = clamp_val(y_new, 1e-6, 1.0 - 2e-6);
    double pi_total = 2.0 * x_new + y_new;
    x_new /= pi_total;
    y_new /= pi_total;

    // --- Outer parameters (a, b) ---
    double m_outer, v_outer;
    if (total_ow > 0.0) {
        m_outer = s1_outer / total_ow;
        // Second pass for variance (needs mean)
        double s2_outer = 0.0;
        for (int i = 0; i < n; i++) {
            double d1 = data(i)       - m_outer;
            double d3 = (1.0 - data(i)) - m_outer;
            s2_outer += d1 * d1 * weights(i, 0) + d3 * d3 * weights(i, 2);
        }
        v_outer = s2_outer / total_ow;
    } else {
        m_outer = 0.25;
        v_outer = 0.01;
    }
    m_outer = clamp_val(m_outer, 1e-4, 1.0 - 1e-4);
    double max_va_outer = m_outer * (1.0 - m_outer) - 1e-6;
    v_outer = clamp_val(v_outer, 1e-8, max_va_outer);

    auto ab  = beta_ab_cpp(m_outer, v_outer);
    double a = ab.first, b = ab.second;
    // Ensure component 1 is left-tail (a <= b => mean <= 0.5)
    if (a > b) std::swap(a, b);
    a = std::max(0.01, a);
    b = std::max(0.01, b);

    // --- Middle parameter (c) ---
    // Beta(c, c) has mean = 0.5; estimate c from weighted variance around 0.5.
    // Var[Beta(c,c)] = 1/(4*(2c+1))  =>  c = (1 - 4v) / (8v)
    double c_val;
    if (sum_w2 > 0.0) {
        double v_mid = 0.0;
        for (int i = 0; i < n; i++) {
            double d = data(i) - 0.5;
            v_mid += d * d * weights(i, 1);
        }
        v_mid /= sum_w2;
        v_mid  = clamp_val(v_mid, 1e-8, 0.25 - 1e-6);
        c_val  = (1.0 - 4.0 * v_mid) / (8.0 * v_mid);
    } else {
        c_val = 1.0;
    }
    c_val = std::max(0.01, c_val);

    // --- Assemble ---
    params.set_size(3, 2);
    params(0, 0) = a;      params(0, 1) = b;
    params(1, 0) = c_val;  params(1, 1) = c_val;
    params(2, 0) = b;      params(2, 1) = a;

    pi = { x_new, y_new, x_new };
}

// ===========================================================================
// Exported functions
// ===========================================================================

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

    mat weights(ndata, ncomp);
    double ll;
    e_step_internal(data, params, pi, weights, ll);

    return List::create(
        Named("weights") = wrap(weights),
        Named("ll")      = ll
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
    mat new_params;
    vec new_pi;
    m_step_internal(data, weights, new_params, new_pi);

    return List::create(
        Named("params") = wrap(new_params),
        Named("pi")     = to_rvec(new_pi)
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

        // Single-pass accumulation: avoids repeated arma::vec reallocation.
        // The first bin also includes x == 0 (mirrors the R implementation).
        int    count = 0;
        double s1 = 0.0, s2 = 0.0;   // sum and sum-of-squares
        for (uword j = 0; j < v.n_elem; j++) {
            double x = v(j);
            if ((x > lo && x <= hi) || (i == 0 && x == 0.0)) {
                count++;
                s1 += x;
                s2 += x * x;
            }
        }

        double mn, va;
        if (count >= 2) {
            mn = s1 / count;
            // Sample variance (matches R's var(), divides by count-1)
            va = (s2 - s1 * s1 / count) / (count - 1);
        } else if (count == 1) {
            mn = s1;
            va = 1e-6;
        } else {
            mn = (lo + hi) / 2.0;
            va = 1e-6;
        }

        // Guard against mean = 0 or 1 (beta_ab undefined there)
        mn = std::max(1e-4, std::min(1.0 - 1e-4, mn));
        // Guard against variance >= mn*(1-mn) (would give a <= 0)
        double max_va = mn * (1.0 - mn) - 1e-6;
        if (max_va <= 0.0) max_va = 1e-6;
        va = std::max(1e-8, std::min(max_va, va));

        auto ab = beta_ab_cpp(mn, va);
        params(i, 0) = ab.first;
        params(i, 1) = ab.second;
        pi_vec(i) = (double)count;
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
        Named("pi")     = to_rvec(pi_vec)
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
//'     \item{bic}{Bayesian Information Criterion: \code{log(n)*(3*ncomp-1) - 2*ll}.
//'       The penalty uses \code{3*ncomp - 1} free parameters: \code{2*ncomp} Beta shape
//'       parameters plus \code{ncomp - 1} free mixing weights.}
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
    mat    params  = init_params;
    vec    pi      = init_pi;
    mat    weights(n, ncomp);   // pre-allocated; reused every iteration
    double ll      = -arma::datum::inf;

    // --- EM loop ---
    int step = 0;
    for (int iter = 0; iter < max_steps; iter++) {
        mat params_old = params;
        vec pi_old     = pi;

        // Fix 6: call internal functions directly — no Rcpp::List round-trips
        e_step_internal(data, params, pi, weights, ll);
        m_step_internal(data, weights, params, pi);

        step++;

        // Convergence check: max relative error across a, b and pi
        double delta = std::max({
            max_relerror_cpp(params_old.col(0), params.col(0)),
            max_relerror_cpp(params_old.col(1), params.col(1)),
            max_relerror_cpp(pi_old,            pi)
        });

        if (step % 10 == 0) {
            Rcpp::Rcout << "step " << step << "  delta " << delta << "\n";
        }

        if (delta < tol) break;
    }

    // BIC: k = 3*ncomp - 1 free parameters
    // (2*ncomp Beta shape params + ncomp-1 free mixing weights)
    double bic = std::log((double)n) * (3 * ncomp - 1) - 2.0 * ll;

    // Component means = a / (a + b); sort by ascending mean
    vec means(ncomp);
    for (int k = 0; k < ncomp; k++) {
        means(k) = params(k, 0) / (params(k, 0) + params(k, 1));
    }
    uvec order = sort_index(means);
    means   = means(order);
    params  = params.rows(order);
    pi      = pi(order);
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
        Named("pi")             = to_rvec(pi),
        Named("usedsteps")      = step,
        Named("means")          = wrap(means),
        Named("classification") = wrap(conv_to<ivec>::from(classification)),
        Named("uncertainty")    = wrap(uncertainty),
        Named("data")           = wrap(data),
        Named("ll")             = ll,
        Named("bic")            = bic
    );
}

// ===========================================================================
// Constrained 3-component model:
//   params = { Beta(a,b),  Beta(c,c),  Beta(b,a) }
//   pi     = { x,          y,          x         }   with 2x + y = 1
// ===========================================================================

// ---------------------------------------------------------------------------
// Exported: Constrained initialisation  —  get_init_constrained_cpp
// ---------------------------------------------------------------------------

//' Generate constrained initial parameter estimates
//'
//' Uses \code{\link{get_init_cpp}} with \code{k = 3} to obtain unconstrained
//' starting values, then symmetrises them to satisfy the constraints
//' \eqn{\text{params} = \{(a,b),\,(c,c),\,(b,a)\}},
//' \eqn{\pi = \{x,\,y,\,x\}}.
//'
//' @param v Numeric vector of observations in \[0, 1\].
//'
//' @return A list with elements:
//'   \describe{
//'     \item{params}{\code{3 x 2} matrix satisfying the symmetry constraint.}
//'     \item{pi}{Length-3 mixing proportions satisfying \eqn{\pi_1 = \pi_3}.}
//'   }
//' @export
// [[Rcpp::export]]
List get_init_constrained_cpp(const arma::vec& v) {
    // Start from unconstrained 3-bin init
    List init    = get_init_cpp(v, 3);
    mat  params0 = as<mat>(init["params"]);
    vec  pi0     = as<vec>(init["pi"]);

    // Component 1 params: use left-bin (a, b) as-is
    double a = params0(0, 0);
    double b = params0(0, 1);

    // Middle component: symmetrise by averaging its two shape params
    double c = (params0(1, 0) + params0(1, 1)) / 2.0;
    c = std::max(0.01, c);

    // Outer mixing weight: average of left- and right-bin proportions
    double x = (pi0(0) + pi0(2)) / 2.0;
    double y = pi0(1);
    double total = 2.0 * x + y;
    if (total <= 0.0) { x = 1.0 / 3.0; y = 1.0 / 3.0; }
    else              { x /= total;     y /= total;     }

    // Enforce a <= b so component 1 is the left-tail component
    if (a > b) std::swap(a, b);
    a = std::max(0.01, a);
    b = std::max(0.01, b);

    mat new_params(3, 2);
    new_params(0, 0) = a;   new_params(0, 1) = b;
    new_params(1, 0) = c;   new_params(1, 1) = c;
    new_params(2, 0) = b;   new_params(2, 1) = a;

    vec new_pi = { x, y, x };

    return List::create(
        Named("params") = wrap(new_params),
        Named("pi")     = to_rvec(new_pi)
    );
}

// ---------------------------------------------------------------------------
// Exported: Constrained M-step  —  get_updated_params_constrained_cpp
// ---------------------------------------------------------------------------

//' Constrained EM M-step for the symmetric Beta mixture model
//'
//' Updates the four free parameters \eqn{(a, b, c, x)} of the constrained
//' 3-component model
//' \eqn{\pi_1 \text{Beta}(a,b) + \pi_2 \text{Beta}(c,c) + \pi_1 \text{Beta}(b,a)}
//' (where \eqn{\pi_1 = \pi_3 = x} and \eqn{\pi_2 = y = 1 - 2x}) given the
//' current posterior weights.
//'
//' @details
//' \strong{Outer parameters \eqn{(a, b)}}:  fitted by moment-matching to an
//' augmented dataset containing the original observations \eqn{x_i} weighted
//' by \eqn{w_{i1}} (component 1) together with the reflected observations
//' \eqn{1 - x_i} weighted by \eqn{w_{i3}} (component 3).  This correctly
//' exploits the symmetry \eqn{\text{Beta}(x; b, a) = \text{Beta}(1-x; a, b)}.
//'
//' \strong{Middle parameter \eqn{c}}: the \eqn{\text{Beta}(c, c)} distribution
//' is constrained to have mean 0.5, so \eqn{c} is estimated solely from the
//' weighted variance of observations around 0.5, via
//' \eqn{c = (1 - 4v) / (8v)} where
//' \eqn{v = \sum_i w_{i2}(x_i - 0.5)^2 / \sum_i w_{i2}}.
//'
//' @param data    Numeric vector of observations in \[0, 1\].
//' @param weights \code{ndata x 3} matrix of posterior weights from the
//'   E-step, column order: component 1, 2, 3.
//'
//' @return A list with elements:
//'   \describe{
//'     \item{params}{\code{3 x 2} matrix of updated Beta parameters satisfying
//'       the constraint: row 1 = \eqn{(a,b)}, row 2 = \eqn{(c,c)},
//'       row 3 = \eqn{(b,a)}.}
//'     \item{pi}{Updated length-3 mixing proportions with
//'       \eqn{\pi_1 = \pi_3}.}
//'   }
//' @export
// [[Rcpp::export]]
List get_updated_params_constrained_cpp(const arma::vec& data,
                                         const arma::mat& weights) {
    if ((int)weights.n_rows != (int)data.n_elem)
        stop("length(data) must equal nrow(weights)");
    if ((int)weights.n_cols != 3)
        stop("weights must have exactly 3 columns");

    mat new_params;
    vec new_pi;
    m_step_constrained_internal(data, weights, new_params, new_pi);

    return List::create(
        Named("params") = wrap(new_params),
        Named("pi")     = to_rvec(new_pi)
    );
}

// ---------------------------------------------------------------------------
// Exported: Constrained full EM  —  estimate_mixture_constrained_cpp
// ---------------------------------------------------------------------------

//' Fit the constrained symmetric Beta mixture model by EM
//'
//' Runs the EM algorithm for the 3-component model
//' \eqn{x\,\text{Beta}(a,b) + y\,\text{Beta}(c,c) + x\,\text{Beta}(b,a)}
//' where \eqn{2x + y = 1}.  The entire loop runs in C++.
//'
//' Initial parameters must satisfy the symmetry constraints; use
//' \code{\link{get_init_constrained_cpp}} to obtain suitable starting values.
//'
//' @param data        Numeric vector of observations in \[0, 1\].
//' @param init_params \code{3 x 2} matrix of initial Beta parameters
//'   satisfying the constraint (row 1: \eqn{(a,b)}, row 2: \eqn{(c,c)},
//'   row 3: \eqn{(b,a)}).
//' @param init_pi     Length-3 mixing proportions with \eqn{\pi_1 = \pi_3}
//'   (must sum to 1).
//' @param max_steps   Maximum EM iterations (default 1000).
//' @param tol         Convergence tolerance on the maximum relative parameter
//'   change (default 1e-5).
//'
//' @return A list with the same structure as
//'   \code{\link{estimate_mixture_cpp}}, with the additional element
//'   \code{constrained = TRUE}.
//' @export
// [[Rcpp::export]]
List estimate_mixture_constrained_cpp(const arma::vec& data,
                                       const arma::mat& init_params,
                                       const arma::vec& init_pi,
                                       int    max_steps = 1000,
                                       double tol       = 1e-5) {
    int n = data.n_elem;

    // --- Input validation ---
    if ((int)init_params.n_rows != 3)
        stop("init_params must have exactly 3 rows");
    if ((int)init_pi.n_elem != 3)
        stop("init_pi must have length 3");
    if (std::abs(sum(init_pi) - 1.0) > 1e-6)
        stop("init_pi must sum to 1");
    if (init_params.min() <= 0.0)
        stop("init_params must all be > 0");
    if (std::abs(init_pi(0) - init_pi(2)) > 1e-6)
        stop("init_pi[1] must equal init_pi[3] (constrained model requires x = pi[1] = pi[3])");

    // --- Initialise ---
    mat    params  = init_params;
    vec    pi      = init_pi;
    mat    weights(n, 3);   // pre-allocated; reused every iteration
    double ll      = -arma::datum::inf;

    // Extract the 4 free parameters (a, b, c, x) for convergence tracking
    auto unique_params = [](const mat& p, const vec& q) -> vec {
        return vec({ p(0,0), p(0,1), p(1,0), q(0) });
    };

    // --- EM loop ---
    int step = 0;
    for (int iter = 0; iter < max_steps; iter++) {
        vec old_unique = unique_params(params, pi);

        // Fix 6: call internal functions directly — no Rcpp::List round-trips
        e_step_internal(data, params, pi, weights, ll);
        m_step_constrained_internal(data, weights, params, pi);

        step++;

        vec    new_unique = unique_params(params, pi);
        double delta      = max_relerror_cpp(old_unique, new_unique);

        if (step % 10 == 0) {
            Rcpp::Rcout << "step " << step << "  delta " << delta << "\n";
        }
        if (delta < tol) break;
    }

    // BIC: 4 free parameters (a, b, c, x)
    double bic = std::log((double)n) * 4.0 - 2.0 * ll;

    // Sort components by ascending mean
    vec means(3);
    for (int k = 0; k < 3; k++) {
        means(k) = params(k, 0) / (params(k, 0) + params(k, 1));
    }
    uvec order  = sort_index(means);
    means       = means(order);
    params      = params.rows(order);
    pi          = pi(order);
    weights     = weights.cols(order);

    // Classification and uncertainty
    uvec classification(n);
    vec  uncertainty(n);
    for (int i = 0; i < n; i++) {
        uword  best_k = weights.row(i).index_max();
        double max_w  = weights(i, best_k);
        classification(i) = best_k + 1;
        uncertainty(i)    = 1.0 - max_w;
    }

    return List::create(
        Named("params")         = wrap(params),
        Named("pi")             = to_rvec(pi),
        Named("usedsteps")      = step,
        Named("means")          = wrap(means),
        Named("classification") = wrap(conv_to<ivec>::from(classification)),
        Named("uncertainty")    = wrap(uncertainty),
        Named("data")           = wrap(data),
        Named("ll")             = ll,
        Named("bic")            = bic,
        Named("constrained")    = true
    );
}
