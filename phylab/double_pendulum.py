"""Tools for the e315 double-pendulum experiment: data acquisition,
energy/momentum analysis, and Lyapunov-exponent estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

__all__ = ["acquire", "findlyap", "mom", "pendcut", "pendlyap", "period"]


def acquire(directory, show_fig):
    """Load double-pendulum angle data recorded to a text file.

    Parameters
    ----------
    directory : str
        Path to the recorded ``.txt`` data file.
    show_fig : bool or int
        If truthy, also plot ``p1`` and ``p2`` against time.

    Returns
    -------
    t : numpy.ndarray
        Real time, in seconds.
    p1 : numpy.ndarray
        Angular displacement of pendulum 1, in radians.
    p2 : numpy.ndarray
        Angular displacement of pendulum 2, in radians.
    sampling_period : float
        The sampling period, in milliseconds.
    myfig : matplotlib figure or str
        The displayed figure if ``show_fig`` is truthy, otherwise the
        string ``'Figure not shown.'``.
    """
    sample_n, p1, p2 = np.loadtxt(
        directory, delimiter=",", skiprows=6, unpack=True
    )

    sampling_period = np.genfromtxt(
        directory,
        usecols=3,
        skip_header=3,
        skip_footer=len(sample_n) + 1,
    )

    t = sample_n * sampling_period * 1e-3

    if show_fig:
        plt.figure()
        plt.plot(t, p1, "b", label=r"$\theta_1$")
        plt.plot(t, p2, "g", label=r"$\theta_2$")
        plt.xlabel("sample time (s)")
        plt.ylabel("angle (radians)")
        plt.title("Angle v Sample number", size=14)
        plt.legend(loc=0, fontsize=14)
        myfig = plt.show()
        return t, p1, p2, sampling_period, myfig

    return t, p1, p2, sampling_period, "Figure not shown."


def findlyap(p2array, maxdiffv, t):
    """Estimate the dominant Lyapunov exponent from trajectory separation.

    Parameters
    ----------
    p2array : numpy.ndarray
        Array of shape ``(m, n)`` with ``m`` data points for each of
        ``n`` runs.
    maxdiffv : sequence of float
        Maximum trajectory separations (in degrees) at which to estimate
        the exponent.
    t : numpy.ndarray
        Sample times corresponding to the rows of ``p2array``.

    Returns
    -------
    None
        Displays a plot of the estimated exponent vs. maximum separation.
    """
    n_ind = np.amin(p2array.shape)
    ly1v = []

    for maxdiff in maxdiffv:
        alpha = []
        for i in range(n_ind - 1):
            for j in range(i, n_ind):
                d = np.abs(p2array[:, i] - p2array[:, j])

                # Keep only separations below the threshold.
                ind = np.cumsum(d > maxdiff / 180 * np.pi) == 0
                d = d[ind]
                n = len(d)

                # Replace zeros with the smallest measurable angle.
                d[d == 0] = 2 * np.pi / 4000
                ld = np.log(d)
                t1 = t[ind]

                alpha.append(
                    (np.sum(t1 * ld) * n - np.sum(ld) * np.sum(t1))
                    / (np.sum(t1 * t1) * n - (np.sum(t1)) ** 2)
                )
        ly1v.append(np.mean(alpha))

    plt.figure()
    plt.plot(maxdiffv, ly1v, "o")
    plt.xlabel("Maximum difference between trajectories (degrees)")
    plt.ylabel("Estimate of Lyapunov exponent")
    plt.title("Estimate of dominant Lyapunov exponent")
    plt.show()


def mom(t, p1, p2, sampling_period, theta, show_fig):
    """Compute angular momentum and total energy of the double pendulum.

    Parameters
    ----------
    t : numpy.ndarray
        Sample times, in seconds.
    p1, p2 : numpy.ndarray
        Angular displacements theta1, theta2, in radians.
    sampling_period : float
        Sampling period (used to compute angular velocities).
    theta : float
        Angle of the pendulum from the vertical, in degrees. Use 90 to
        study zero-gravity motion.
    show_fig : bool or int
        If truthy, plot ``P1 / sqrt(E)`` and ``theta2`` against time.

    Returns
    -------
    P1 : numpy.ndarray
        Total angular momentum.
    E : numpy.ndarray
        Total energy.
    myfig : matplotlib figure or str
        The displayed figure if ``show_fig`` is truthy, otherwise the
        string ``'Figure not shown.'``.
    """
    p1d = np.gradient(p1) * sampling_period
    p2d = np.gradient(p2) * sampling_period

    g = 9.799403 * np.cos(theta * np.pi / 180)

    m1 = 315.50e-3  # kg
    m2 = 76.53e-3  # kg

    l1 = 25.5e-3  # m
    l2 = 23.5e-3  # m
    l3 = 57e-3  # m

    i1 = 9.07e-4  # kg m^2
    i2 = 6.90e-5  # kg m^2

    k1 = (i1 + m2 * l3 ** 2) / 2
    k2 = i2 / 2
    k3 = m2 * l2 * l3

    nu1 = (m1 * l1 + m2 * l3) * g
    nu2 = m2 * l2 * g

    c2 = np.cos(p2)

    p1_mom = 2 * p1d * (k1 + k2 + k3 * c2) + p2d * (2 * k2 + k3 * c2)
    p2_mom = p1d * (2 * k2 + k3 * c2) + 2 * p2d * k2

    energy = (
        p1_mom * p1_mom * k2
        + p2_mom * p2_mom * (k1 + k2 + k3 * c2)
        - p1_mom * p2_mom * (2 * k2 + k3 * c2)
    ) / (4 * k1 * k2 - k3 * k3 * c2 * c2)
    energy = energy + nu1 * (1 - np.cos(p1)) + nu2 * (1 - np.cos(p1 + p2))

    if show_fig:
        plt.figure()
        plt.subplot(211)
        plt.plot(t, np.abs(p1_mom / np.sqrt(energy)))
        plt.xlabel("time [s]")
        plt.ylabel(r"$P_1/\sqrt{E} \, [m \sqrt{kg}]$")
        plt.title(r"Plot of $P_1/\sqrt{E}$ and of $\theta_2$ motion", fontsize=14)
        plt.grid()
        plt.subplot(212)
        plt.plot(t, p2)
        plt.xlabel("t [s]")
        plt.ylabel(r"$\theta_2$ [rad]")
        plt.grid()
        myfig = plt.show()
        return p1_mom, energy, myfig

    return p1_mom, energy, "Figure not shown."


def pendcut(t, p1, p2, threshold):
    """Trim leading near-stationary samples from double-pendulum data.

    Cuts ``p1`` and ``p2`` so that time zero corresponds to the first
    point where the angular velocity exceeds ``threshold`` events/ms,
    padding the remainder with zeros, and plots the result.

    Parameters
    ----------
    t : numpy.ndarray
        Sample times, in seconds.
    p1, p2 : numpy.ndarray
        Angular displacements theta1, theta2, in radians.
    threshold : float
        Velocity threshold, in events/ms. Start with 1 and increase to
        remove more of the stationary lead-in.

    Returns
    -------
    p1cut, p2cut : numpy.ndarray
        The trimmed and zero-padded displacement arrays.
    myfig : matplotlib figure
        The displayed figure.
    """
    dt = np.mean(np.diff(t))
    length = len(p1)

    d1 = np.abs(np.diff(p1) / dt) > ((threshold * 2 * np.pi / 4000) / 0.001)
    idx1 = np.nonzero(d1 > 0)[0]

    d2 = np.abs(np.diff(p2) / dt) > ((threshold * 2 * np.pi / 4000) / 0.001)
    idx2 = np.nonzero(d2 > 0)[0]

    idx = np.array([np.amin(idx1), np.amin(idx2)])

    p1cut = np.zeros(p1.shape)
    p2cut = np.zeros(p2.shape)

    if len(idx) > 0:
        start = idx[0] + 1
        p1cut[0:length - start + 1] = p1[start - 1:length]
        p2cut[0:length - start + 1] = p2[start - 1:length]

    plt.figure()
    plt.plot(t, p1cut, "r-", t, p2cut, "g-")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\theta_1$ (red), $\theta_2$ (green) [radian]")
    plt.title("Motion of double pendulum")
    myfig = plt.show()

    return p1cut, p2cut, myfig


def _double_pend_ode(t, theta, params):
    """Right-hand side of the double-pendulum ODE, plus its Jacobian
    acting on the four orthonormal tangent vectors used for the
    Lyapunov-exponent calculation in :func:`pendlyap`.
    """
    k1, k2, k3, nu1, nu2 = params
    d_theta = np.zeros((20, 1))

    a1 = k3 * nu2 * np.cos(theta[1])
    a2 = 2 * k2 * nu1
    a3 = 2 * k2 * k3 * (theta[2] + theta[3]) ** 2 + k3 ** 2 * theta[2] ** 2 * np.cos(theta[1])

    b1 = 2 * k1 * nu2 + k3 * nu2 * np.cos(theta[1])
    b2 = 2 * k2 * nu1 + k3 * nu1 * np.cos(theta[1])
    b3 = (2 * k2 * k3 + k3 ** 2 * np.cos(theta[1])) * (theta[2] + theta[3]) ** 2 + (
        2 * k1 * k3 + k3 ** 2 * np.cos(theta[1])
    ) * theta[2] ** 2

    num3 = a1 * np.sin(theta[0] + theta[1]) - a2 * np.sin(theta[0]) + a3 * np.sin(theta[1])
    num4 = b1 * np.sin(theta[0] + theta[1]) - b2 * np.sin(theta[0]) + b3 * np.sin(theta[1])
    denom = 4 * k1 * k2 - k3 ** 2 * np.cos(theta[1]) ** 2

    d_theta[0] = theta[2]
    d_theta[1] = theta[3]
    d_theta[2] = num3 / denom
    d_theta[3] = num4 / (-denom)

    v_matrix = theta[4:].reshape((4, 4), order="F").copy()
    jac = np.zeros((4, 4))

    jac[0, 2] = 1
    jac[1, 3] = 1

    jac[2, 0] = (a1 * np.cos(theta[0] + theta[1]) - a2 * np.cos(theta[0])) / denom
    jac[3, 0] = (b1 * np.cos(theta[0] + theta[1]) - b2 * np.cos(theta[0])) / (-denom)

    jac[2, 1] = (
        (
            -k3 * nu2 * np.sin(theta[1]) * np.sin(theta[0] + theta[1])
            + a1 * np.cos(theta[0] + theta[1])
            - k3 ** 2 * theta[2] ** 2 * np.sin(theta[1]) ** 2
            + a3 * np.cos(theta[1])
        )
        * denom
        - num3 * (k3 ** 2 * np.sin(2 * theta[1]))
    ) / denom ** 2

    jac[3, 1] = -(
        (
            -k3 * nu2 * np.sin(theta[1]) * np.sin(theta[0] + theta[1])
            + b1 * np.cos(theta[0] + theta[1])
            + k3 * nu1 * np.sin(theta[1]) * np.sin(theta[0])
            - k3 ** 2 * ((theta[2] + theta[3]) ** 2 + theta[2] ** 2) * np.sin(theta[1]) ** 2
            + b3 * np.cos(theta[1])
        )
        * denom
        - num4 * (k3 ** 2 * np.sin(2 * theta[1]))
    ) / denom ** 2

    jac[2, 2] = (
        4 * k2 * k3 * (theta[2] + theta[3]) + 2 * k3 ** 2 * theta[2] * np.cos(theta[1])
    ) * np.sin(theta[1]) / denom
    jac[2, 3] = (4 * k2 * k3 * (theta[2] + theta[3])) * np.sin(theta[1]) / denom
    jac[3, 2] = -(
        2 * (2 * k2 * k3 + k3 ** 2 * np.cos(theta[1])) * (theta[2] + theta[3])
        + 2 * (2 * k1 * k3 + k3 ** 2 * np.cos(theta[1])) * theta[2]
    ) * np.sin(theta[1]) / denom
    jac[3, 3] = (
        -2 * (2 * k2 * k3 + k3 ** 2 * np.cos(theta[1])) * (theta[2] + theta[3])
        * np.sin(theta[1]) / denom
    )

    v_matrix = jac @ v_matrix
    d_theta[4:] = v_matrix.reshape((16, 1), order="F").copy()
    return d_theta


def pendlyap(tstep, initcond, maxiter):
    """Estimate the Lyapunov exponents of the double pendulum.

    Integrates the double-pendulum equations of motion together with
    the tangent-space (Jacobian) equations, applying a Gram-Schmidt
    re-orthonormalisation every ``tstep`` seconds (the standard method
    of Wolf et al., Physica D 16, 285 (1985)).

    Parameters
    ----------
    tstep : float
        Time step, in seconds, between re-orthonormalisations. A
        typical value is 0.05 (50 ms).
    initcond : sequence of length 4
        Initial condition ``[theta1, theta2, dtheta1/dt, dtheta2/dt]``,
        in radians and radians/s. A typical value is
        ``[-110, 110, 0, 0]`` (in degrees, converted to radians).
    maxiter : int
        Number of re-orthonormalisation steps. A typical value is 1000.

    Returns
    -------
    Lyap : numpy.ndarray
        Array of shape ``(maxiter + 1, 4)`` with the running estimate of
        each of the four Lyapunov exponents at each step.
    """
    m1 = 315.50e-3  # kg
    m2 = 76.53e-3  # kg

    l2 = 23.5e-3  # m
    l3 = 57.0e-3  # m
    l1 = 25.5e-3  # m

    i1 = 9.07e-4  # kg m^2
    i2 = 6.90e-5  # kg m^2

    g = 9.799403  # m/s^2

    k1 = (i1 + m2 * l3 ** 2) / 2
    k2 = i2 / 2
    k3 = m2 * l2 * l3

    nu1 = (m1 * l1 + m2 * l3) * g
    nu2 = m2 * l2 * g

    params = (k1, k2, k3, nu1, nu2)

    y = np.zeros(20)
    y[0:4] = initcond
    y[4:] = np.eye(4).reshape(16, order="F")

    tab_y = np.zeros((maxiter + 1, 4))
    tab_y[0, :] = np.asarray(initcond)

    tab_lyap = np.zeros((maxiter + 1, 4))

    for it in range(1, maxiter + 1):
        if it % 100 == 0:
            print("iteration # %d" % it)

        integrator = ode(
            lambda t, theta: _double_pend_ode(t, theta, params)
        ).set_integrator("dopri5", method="adams")

        t_start = 0.0
        t_final = tstep
        delta_t = 0.5 * tstep
        num_steps = int(np.floor((t_final - t_start) / delta_t)) + 1

        integrator.set_initial_value(y, t_start)

        trajectory = np.zeros((num_steps, 20))
        k = 1
        while integrator.successful() and k < num_steps:
            integrator.integrate(integrator.t + delta_t)
            trajectory[k, :] = integrator.y
            k += 1

        # Solution at t = tstep (index 2: initial value, half-step, full step).
        y1 = trajectory[2, :]
        tab_y[it, 0:4] = y1[0:4]

        v_matrix = y1[4:].reshape(4, 4, order="F").copy()

        # Gram-Schmidt re-orthonormalisation (GSR); see Eq. (4) of
        # Wolf et al., Physica D 16, 285 (1985).
        t_norm = np.zeros(4)
        for k in range(4):
            if k > 1:
                v_matrix[:, k] = v_matrix[:, k] - (
                    v_matrix[:, k].T @ v_matrix[:, 1:k]
                ) @ v_matrix[:, 1:k].T
            t_norm[k] = np.linalg.norm(v_matrix[:, k])
            v_matrix[:, k] = v_matrix[:, k] / t_norm[k]

        y = np.zeros(20)
        y[0:4] = y1[0:4]
        y[4:] = v_matrix.flatten(order="F")

        tab_lyap[it, :] = tab_lyap[it - 1, :] + np.log(t_norm)

    t = np.arange(0.0, maxiter + 1.0, 1) * tstep
    for k in range(4):
        tab_lyap[:, k] = tab_lyap[:, k] / np.where(t == 0, 1, t)

    ax1 = plt.subplot(211)
    ax1.plot(
        t,
        tab_y[0:maxiter + 1, 0] * 180 / np.pi,
        t,
        tab_y[0:maxiter + 1, 1] * 180 / np.pi,
    )
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax1.set_title("Motion of double pendulum")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel(r"$\theta_1$ (blue), $\theta_2$ (green) [radians]")
    ax1.grid(True)

    ax2 = plt.subplot(212)
    ax2.plot(t, tab_lyap[:, 0], "r")
    ax2.plot(t, tab_lyap[:, 1], "b")
    ax2.plot(t, tab_lyap[:, 2], "g")
    ax2.plot(t, tab_lyap[:, 3], "k")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("Lyapunov exponents")

    plt.tight_layout()
    plt.show()

    return tab_lyap.copy()


def period(t, pn):
    """Estimate the period of oscillatory double-pendulum motion.

    The period is obtained as the mean spacing between zero crossings
    with positive slope (values differing from the mean by more than
    5% are rejected).

    Parameters
    ----------
    t : numpy.ndarray
        Time vector.
    pn : numpy.ndarray
        The theta1 or theta2 motion vector to analyse.

    Returns
    -------
    float
        The estimated period, in the same units as ``t``.
    """
    length = len(t)

    pshift = np.delete(np.hstack((np.zeros(1), pn)), length)

    i1 = np.nonzero((pn >= 0) & (pshift < 0))[0]

    t0 = t[i1 - 2] - pn[i1 - 2] * (t[i1 - 1] - t[i1 - 2]) / (pn[i1 - 1] - pn[i1 - 2])

    d = np.diff(t0)
    d2 = np.abs(d - np.mean(d)) / np.mean(d)

    return np.mean(d[d2 < 0.05])
