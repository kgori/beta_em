Installation:

    git clone https://github.com/kgori/beta_em
    cd beta_em
    R CMD INSTALL .

Example use:

    library(betaem)

    # Make some mock data with two components
    mix <- c(rbeta(10900, 24,45), rbeta(32700, 35, 36))

    # Estimate the initial parameters
    starting_guess <- get_init(mix, k=2)

    # Optimise (plotting is optional, and it's faster without)
    est <- estimate_mixture(init_params=starting_guess$params, init_pi=starting_guess$pi, data=mix, plot=FALSE)

    # Look at the result
    est$params

             [,1]     [,2]
    [1,] 22.63647 41.98124
    [2,] 35.25560 36.21190

    # component means:
    est$means

              [,1]
    [1,] 0.3503138
    [2,] 0.4933095

    # component modes:
    beta_mode(est$params[1, 1], est$params[1, 2]) # 0.3455328 
    beta_mode(est$params[2, 1], est$params[2, 2]) # 0.4931169
