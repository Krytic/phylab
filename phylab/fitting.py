"""Curve-fitting and regression routines used throughout the phylab courses.

This module provides a small Levenberg-Marquardt non-linear least-squares
fitter (:func:`nonlinft`) together with a weighted polynomial regression
helper (:func:`regress`). These were originally written for MATLAB and
ported to Python; the numerical algorithm is unchanged, but the code has
been cleaned up and modernised so it runs under current NumPy/SciPy.
"""

import numpy as np

__all__ = ["nonlinft", "regress"]


def _nonlin2(mfunc, x, y, sy, p, delta, lamb):
    """Compute the alpha matrix and beta vector for one L-M iteration.

    This is an internal helper used exclusively by :func:`nonlinft`; it is
    not part of the public API.
    """
    m = p.shape[0]
    n = x.shape[0]

    # beta = -0.5 * grad(chi^2), evaluated by numerical differentiation.
    d_chisq_da = np.zeros(m)
    for i in range(m):
        p[i] += 0.5 * delta[i]
        chisq_plus = np.sum((np.abs(y - mfunc(x, p)) / sy) ** 2)
        p[i] -= delta[i]
        chisq_minus = np.sum((np.abs(y - mfunc(x, p)) / sy) ** 2)
        p[i] += 0.5 * delta[i]
        d_chisq_da[i] = (chisq_plus - chisq_minus) / delta[i]
    beta = -0.5 * d_chisq_da

    # alpha = curvature matrix.
    alpha = np.zeros((m, m))
    y0 = mfunc(x, p)
    dy_da = np.zeros((n, m))
    for i in range(m):
        p[i] += delta[i]
        y1 = mfunc(x, p)
        p[i] -= delta[i]
        dy_da[:, i] = (y1 - y0) / (sy * delta[i])
    for j in range(n):
        j_dy_da = dy_da[j, :]
        alpha += np.outer(j_dy_da, j_dy_da)

    for i in range(m):
        alpha[i, i] *= 1 + lamb

    return alpha, beta


def nonlinft(mfunc, x, y, sy, pt, v, chi_cut=0.01, max_iter=100, verbose=True):
    """Fit a non-linear model to data using Levenberg-Marquardt regression.

    Parameters
    ----------
    mfunc : callable
        ``mfunc(x, p)`` returns the model's predicted y-values for the
        independent variable ``x`` and parameter vector ``p``. It must not
        be complex-valued.
    x : array_like
        Independent variable values (e.g. time).
    y : array_like
        Dependent variable values (e.g. counts recorded).
    sy : array_like
        Standard error on each value of ``y``.
    pt : array_like
        Initial estimate of the parameters to be fitted. All parameters
        must be non-zero, and are assumed to be real.
    v : array_like
        A 0/1 vector the same length as ``pt``: 1 means the corresponding
        parameter is varied (fitted), 0 means it is held fixed. For
        example ``[1, 0, 1, 1]`` fixes the second of four parameters.
    chi_cut : float, optional
        Convergence tolerance on the change in chi-squared between
        iterations. Defaults to 0.01.
    max_iter : int, optional
        Maximum number of iterations before giving up. Defaults to 100.
    verbose : bool, optional
        If True (the default), print a short fit report (best-fit
        parameters and normalised chi-squared) once the fit converges.

    Returns
    -------
    pbest : numpy.ndarray
        Best-fit parameters.
    perror : numpy.ndarray
        Standard error on each fitted parameter.
    nchi2 : float
        Normalised chi-squared (chi-squared / degrees of freedom). A
        good fit gives a value close to 1.

    Notes
    -----
    Written 4/12/95 by Michael Fleming, University of Auckland.
    Translated to Python on 4/2/14 by Maarten Hoogerland.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)
    v = np.asarray(v, dtype=float)
    p = np.array(pt, dtype=float)

    lamb = 0.001
    step_size = 0.001
    chi_old = 1e30

    n_params = p.shape[0]
    n_points = x.shape[0]
    delta = p * step_size  # step used for numerical derivatives

    dof = int(np.sum(v == 1))
    dof = n_points - dof

    y_model = mfunc(x, p)
    chi_sqr = np.sum((np.abs(y - y_model) / sy) ** 2)
    if chi_sqr / float(dof) > 5000.0:
        print("You have made a bad choice of initial parameters")

    n_iter = 0
    while (abs(chi_old - chi_sqr) > chi_cut) and (n_iter < max_iter):
        n_iter += 1
        chi_old = chi_sqr
        alpha, beta = _nonlin2(mfunc, x, y, sy, p, delta, lamb)
        if np.linalg.det(alpha) == 0.0:
            raise RuntimeError("No convergence - try a different set of parameters")
        dp = np.linalg.solve(alpha, beta) * v
        p = p + dp
        chi_sqr = np.sum((np.abs(y - mfunc(x, p)) / sy) ** 2)

        while chi_sqr > (chi_old + chi_cut):
            p = p - dp
            n_iter += 1
            lamb *= 10.0
            alpha, beta = _nonlin2(mfunc, x, y, sy, p, delta, lamb)
            if np.linalg.det(alpha) == 0:
                raise RuntimeError(
                    "No convergence - try a different set of parameters"
                )
            dp = np.linalg.solve(alpha, beta) * v
            p = p + dp
            chi_sqr = np.sum((np.abs(y - mfunc(x, p)) / sy) ** 2)

        lamb = 0.1 * lamb

    if n_iter == max_iter:
        print("Maximum number of iterations exceeded - convergence not achieved")

    # Standard errors from the final curvature matrix.
    alpha, _ = _nonlin2(mfunc, x, y, sy, p, delta, 0.0)
    sp = np.sqrt(np.diagonal(np.linalg.inv(alpha))) * v
    nchi2 = chi_sqr / dof

    if verbose:
        print("Parameters +/- errors")
        for i in range(n_params):
            print(p[i], "+/-", sp[i])
        print("Normalised Chi Squared is ", nchi2)

    return p, sp, nchi2


def regress(x, y, sy, n):
    """Weighted least-squares fit of a degree-``n`` polynomial to data.

    Parameters
    ----------
    x : array_like
        Independent variable values.
    y : array_like
        Dependent variable values.
    sy : array_like
        Standard error on each value of ``y``, used as the fit weights.
    n : int
        Degree of the polynomial to fit.

    Returns
    -------
    p : numpy.ndarray
        Polynomial coefficients, highest order first, as in
        ``numpy.polyval``.
    sp : numpy.ndarray
        Standard error on each coefficient.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sy = np.asarray(sy, dtype=float)

    design = np.ones(x.size)
    for i in range(n):
        design = np.column_stack((x ** (i + 1), design))

    weights = np.diag(1.0 / (sy * sy))
    design_t = np.transpose(design)

    b = design_t @ weights @ design
    b_inv = np.linalg.inv(b)
    c = design_t @ (weights @ y)

    p = b_inv @ c
    sp = np.sqrt(np.diagonal(b_inv))
    return p, sp
