"""Generic numerical ODE integration helpers."""

import numpy as np

__all__ = ["rk4"]


def rk4(f, t0, tfinal, y0, h, varargs=()):
    """Integrate an ODE with the classic fourth-order Runge-Kutta method.

    Parameters
    ----------
    f : callable
        Right-hand side of the ODE. Called as ``f(t, y)``, or as
        ``f(t, y, varargs)`` if ``varargs`` is non-empty.
    t0 : float
        Start time.
    tfinal : float
        End time.
    y0 : float or numpy.ndarray
        Initial condition.
    h : float
        Step size.
    varargs : tuple, optional
        Extra arguments passed through to ``f``.

    Returns
    -------
    tout : numpy.ndarray
        Times at which the solution was evaluated.
    yout : numpy.ndarray
        Solution values at each time in ``tout``.
    """
    y = y0
    tout = np.arange(t0, tfinal, h)
    if tout[-1] < tfinal:
        tout = np.append(tout, tfinal)
    n_times = tout.size

    is_vector = isinstance(y0, np.ndarray)
    if is_vector:
        yout = np.zeros([n_times, y0.size])
        yout[0, :] = y0
    else:
        yout = np.zeros([n_times])
        yout[0] = y0

    for i in range(n_times - 1):
        t = tout[i]
        if len(varargs) == 0:
            k1 = h * f(t, y)
            k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
            k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
            k4 = h * f(t + h, y + k3)
        else:
            k1 = h * f(t, y, varargs)
            k2 = h * f(t + 0.5 * h, y + 0.5 * k1, varargs)
            k3 = h * f(t + 0.5 * h, y + 0.5 * k2, varargs)
            k4 = h * f(t + h, y + k3, varargs)
        y = y + (k1 + 2 * (k2 + k3) + k4) / 6.0
        if is_vector:
            yout[i + 1, :] = y
        else:
            yout[i + 1] = y

    return tout, yout
