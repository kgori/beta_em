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

// ===========================================================================
// Constrained 3-component model:
//   params = { Beta(a,b),  Beta(c,c),  Beta(b,a) }
//   pi     = { x,          y,          x         }   with 2x + y = 1
// ===========================================================================

// ---------------------------------------------------------------------------
// Internal helper: clamp a scalar to [lo, hi]
// ---------------------------------------------------------------------------
static inline double clamp_val(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

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
        Named("pi")     = wrap(new_pi)
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
    int n = data.n_elem;
    if ((int)weights.n_rows != n)
        stop("length(data) must equal nrow(weights)");
    if ((int)weights.n_cols != 3)
        stop("weights must have exactly 3 columns");

    // --- Mixing proportions -------------------------------------------
    // Each outer component gets half of the combined outer weight.
    double sum_outer = accu(weights.col(0)) + accu(weights.col(2));
    double x_new     = sum_outer / (2.0 * n);
    double y_new     = accu(weights.col(1)) / n;
    // Clamp to avoid degenerate pi
    x_new = clamp_val(x_new, 1e-6, 0.5 - 1e-6);
    y_new = clamp_val(y_new, 1e-6, 1.0 - 2e-6);
    // Re-normalise so that 2x + y = 1 exactly
    double pi_total = 2.0 * x_new + y_new;
    x_new /= pi_total;
    y_new /= pi_total;

    // --- Outer component parameters (a, b) ----------------------------
    // Build augmented dataset:
    //   original data[i] with weight w1[i]  → models Beta(a, b)
    //   reflected (1-data[i]) with weight w3[i] → models Beta(a, b) via symmetry
    vec aug_data    = join_cols(data, 1.0 - data);
    vec aug_weights = join_cols(weights.col(0), weights.col(2));
    double total_ow = sum(aug_weights);

    double m_outer, v_outer;
    if (total_ow > 0.0) {
        m_outer = dot(aug_data, aug_weights) / total_ow;
        vec diff = aug_data - m_outer;
        v_outer  = dot(diff % diff, aug_weights) / total_ow;
    } else {
        m_outer = 0.25;
        v_outer = 0.01;
    }
    m_outer = clamp_val(m_outer, 1e-4, 1.0 - 1e-4);
    double max_va = m_outer * (1.0 - m_outer) - 1e-6;
    v_outer = clamp_val(v_outer, 1e-8, max_va);

    vec ab = beta_ab_cpp(m_outer, v_outer);
    double a = ab(0), b = ab(1);
    // Ensure component 1 remains the left-tail (a <= b => mean <= 0.5)
    if (a > b) std::swap(a, b);
    a = std::max(0.01, a);
    b = std::max(0.01, b);

    // --- Middle component parameter (c) --------------------------------
    // Beta(c, c) is constrained to mean = 0.5; estimate c from the
    // variance of data around 0.5 using component-2 weights.
    // Var[Beta(c,c)] = 1 / (4*(2c+1))  =>  c = (1 - 4v) / (8v)
    double total_mw = accu(weights.col(1));
    double c_val;
    if (total_mw > 0.0) {
        vec diff_mid = data - 0.5;
        double v_mid = dot(diff_mid % diff_mid, weights.col(1)) / total_mw;
        v_mid  = clamp_val(v_mid, 1e-8, 0.25 - 1e-6);
        c_val  = (1.0 - 4.0 * v_mid) / (8.0 * v_mid);
    } else {
        c_val = 1.0;
    }
    c_val = std::max(0.01, c_val);

    // --- Assemble result ----------------------------------------------
    mat new_params(3, 2);
    new_params(0, 0) = a;      new_params(0, 1) = b;
    new_params(1, 0) = c_val;  new_params(1, 1) = c_val;
    new_params(2, 0) = b;      new_params(2, 1) = a;

    vec new_pi = { x_new, y_new, x_new };

    return List::create(
        Named("params") = wrap(new_params),
        Named("pi")     = wrap(new_pi)
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
    mat  params = init_params;
    vec  pi     = init_pi;
    mat  weights(n, 3);
    double ll = -arma::datum::inf;

    // Extract unique parameters for convergence tracking: (a, b, c, x)
    auto unique_params = [](const mat& p, const vec& q) -> vec {
        return vec({ p(0,0), p(0,1), p(1,0), q(0) });
    };

    // --- EM loop ---
    int step = 0;
    for (int iter = 0; iter < max_steps; iter++) {
        vec old_unique = unique_params(params, pi);

        // E-step (unconstrained: works with any params/pi)
        List w_out = get_weights_cpp(data, params, pi);
        weights    = as<mat>(w_out["weights"]);
        ll         = as<double>(w_out["ll"]);

        // M-step (constrained)
        List p_out     = get_updated_params_constrained_cpp(data, weights);
        mat  new_params = as<mat>(p_out["params"]);
        vec  new_pi     = as<vec>(p_out["pi"]);

        step++;
        params = new_params;
        pi     = new_pi;

        // Convergence: max relative change in the 4 free parameters
        vec new_unique = unique_params(params, pi);
        double delta   = max_relerror_cpp(old_unique, new_unique);

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
        Named("pi")             = wrap(pi),
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
