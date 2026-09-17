"""Tools for the e314 chaotic-pendulum experiment, including the
logistic-map demonstrations used to introduce chaotic dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.integrate import odeint

__all__ = [
    "pendeq",
    "bifurc",
    "ellipmodel",
    "pendulum",
    "ginput",
    "logistic",
    "logisticmap",
]


def pendeq(v, t, p):
    """Right-hand side of the damped, driven pendulum ODE.

    Parameters
    ----------
    v : sequence
        State variables ``[theta, dtheta/dt]``.
    t : float
        Time.
    p : sequence
        Parameters ``[g, Q, w]`` (driving force, quality factor, and
        driving angular frequency).

    Returns
    -------
    list
        The time derivative of ``v``.
    """
    y0, y1 = v
    g, q, w = p
    damping = 1.0 / q
    return [y1, -damping * y1 - np.sin(y0) + g * np.cos(w * t)]


def bifurc():
    """Plot a bifurcation diagram for the chaotic pendulum.

    Solves the driven-pendulum ODE for a range of driving forces and
    samples the angular velocity once per drive cycle (a Poincare
    section), producing the classic bifurcation plot used in the e314
    lab handout.
    """
    q = 2  # friction
    tmin = 500  # allow transients to settle
    tmax = 2000  # increase for a cleaner Poincare plot
    w = 0.666667  # driving frequency

    omega = []
    g_range = np.arange(0.5, 1.601, 0.01)

    for j, g in enumerate(g_range, start=1):
        print(j)

        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = 5000.0
        numpoints = 25 * int(stoptime)

        t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        p = [g, q, w]
        v0 = [0, 0]

        ysol = odeint(pendeq, v0, t, args=(p,), atol=abserr, rtol=relerr)

        ya = ysol[:, 0]
        ya = np.mod(ya, 2 * np.pi)  # displacement wrapped to +/- pi
        ya = ya - (ya > np.pi) * 2 * np.pi
        yb = ysol[:, 1]
        n_samples = ysol[:, 0].size

        if tmax > n_samples:
            raise ValueError(
                "Time range exceeds matrix dimensions. Try increasing numpoints."
            )
        vel = yb[tmin:tmax]
        t = np.array(t[tmin:tmax])

        # Sample velocity at a fixed phase of the driving cycle.
        i = np.where(np.cos(w * t) > 0.9999)
        omega.append(vel[i])

    plt.plot(g_range, omega, "r.")
    plt.xlabel("driving force")
    plt.ylabel("angular velocity")
    plt.title("Bifurcation graph for pendulum")

    # omega contains many zeros which would otherwise draw a spurious
    # line along the x-axis; overplot white markers to hide them.
    plt.plot(g_range, np.zeros(g_range.size), "w.")
    plt.grid()
    plt.show()


def ellipmodel(c, t, y):
    """Objective function for fitting the pendulum's elliptic-integral model.

    Intended to be minimised (e.g. with :func:`scipy.optimize.fmin`) to
    find the best-fit parameters of a damped pendulum whose period is
    described via the complete elliptic integral of the first kind.

    Parameters
    ----------
    c : sequence of length 5
        Fit parameters.
    t : array_like
        Time values.
    y : array_like
        Measured pendulum position values.

    Returns
    -------
    float
        Sum of squared residuals between the data and the model.
    """
    m = (np.sin(np.exp(-c[1] * t) * c[0] / 2.0)) ** 2
    z = special.ellipk(m)  # complete elliptic integral of the first kind
    w = np.pi * c[2] / 2.0 / z

    dt = t[1] - t[0]
    phase = np.cumsum(w) * dt

    error = np.sum((y - c[0] * np.exp(-c[1] * t) * np.cos(phase + c[3]) - c[4]) ** 2)
    return error


def pendulum(g, q, tmax):
    """Simulate and plot the motion of the chaotic pendulum.

    Solves the dimensionless damped, driven pendulum ODE (see
    :func:`pendeq`) and produces a four-panel figure: position/velocity
    vs. time, power spectrum, phase diagram, and Poincare section.

    Parameters
    ----------
    g : float
        Driving force (typically ``0.8 < g < 4``).
    q : float
        Damping term (quality factor).
    tmax : int
        Maximum sample index to plot; must be large enough (>= 5000) to
        show a meaningful Poincare plot.
    """
    tmin = 500  # allow transients to settle
    w = 0.666667  # driving angular frequency (f = 0.106 Hz)

    abserr = 1.0e-8
    relerr = 1.0e-6
    stoptime = 5000.0
    numpoints = 25 * int(stoptime)

    t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
    p = [g, q, w]
    v0 = [0, 0]

    ysol = odeint(pendeq, v0, t, args=(p,), atol=abserr, rtol=relerr)

    ya = ysol[:, 0]
    ya = np.mod(ya, 2 * np.pi)
    ya = ya - (ya > np.pi) * 2 * np.pi
    yb = ysol[:, 1]
    n_samples = ysol[:, 0].size
    n_times = len(t)

    if tmax > n_samples:
        raise ValueError(
            "Time range exceeds matrix dimensions. Try increasing numpoints."
        )
    pos = ya[tmin:tmax]
    vel = yb[tmin:tmax]
    t = np.array(t[tmin:tmax])

    plt.figure(figsize=(7.0, 7.0))
    plt.subplots_adjust(hspace=0.45, wspace=0.3)

    # Position and velocity vs. time (a few cycles, for clarity).
    n_show = int(5e-3 * n_times)
    plt.subplot(221)
    plt.plot(t[1:n_show], pos[1:n_show], "r", t[1:n_show], vel[1:n_show], "g")
    plt.xlabel("time in seconds")
    plt.ylabel("position/velocity")
    plt.title(r"pendulum motion: $g=%s$, $Q=%s$" % (g, q))
    plt.grid()
    plt.legend(("position", "velocity"), loc="best")

    # Power spectrum.
    y_fft = np.fft.fft(pos, 4096)
    power = y_fft * np.conj(y_fft) / 4096.0
    freq = (n_times / tmax) * np.arange(0, 2048) / 4096.0
    plt.subplot(222)
    plt.plot(freq[1:300], power[1:300].real)
    plt.ylabel("Power Spectrum")
    plt.xlabel("frequency (Hz)")
    plt.title(r"Power Spectrum: $g=%s$, $Q=%s$" % (g, q))
    plt.grid()

    # Phase diagram.
    plt.subplot(223)
    plt.plot(pos, vel, "r.")
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.title(r"Phase diagram: $g=%s$, $Q=%s$" % (g, q))
    plt.grid()

    # Poincare section: phase diagram sampled at a fixed drive phase.
    # A tolerance window (cos(w*t) > 0.9999) is used since no sample
    # falls exactly on cos(w*t) == 1.
    i = np.where(np.cos(w * t) > 0.9999)
    plt.subplot(224)
    plt.plot(pos[i], vel[i], "r.")
    plt.axis([-np.pi, np.pi, -np.pi, np.pi])
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.title(r"Poincare plot: $g=%s$, $Q=%s$" % (g, q))
    plt.grid()

    plt.tight_layout()
    plt.show()


class _DataCursor:
    """Small helper that annotates clicked points on a plot with their
    coordinates. Used internally by :func:`ginput`.
    """

    text_template = "x: %0.2f\ny: %0.2f"

    def __init__(self, ax):
        self.ax = ax
        self.x, self.y = 0.0, 0.0
        self.annotation = ax.annotate(
            self.text_template,
            xy=(self.x, self.y),
            xytext=(-20, 20),
            textcoords="offset points",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )
        self.annotation.set_visible(False)

    def __call__(self, event):
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        if self.x is not None:
            self.annotation.xy = (self.x, self.y)
            self.annotation.set_text(self.text_template % (self.x, self.y))
            self.annotation.set_visible(True)
            event.canvas.draw()


def ginput(mu, n_points):
    """Interactively pick points on a logistic-map cobweb plot.

    Plots the logistic map (see :func:`logisticmap`) for a given ``mu``
    and lets the user click ``n_points`` points on it. Clicked points
    are saved to ``coordinates.txt`` in the current directory.

    Parameters
    ----------
    mu : float
        Logistic-map growth parameter, ``mu < 4``.
    n_points : int
        Number of points to click on the plot.

    Returns
    -------
    x : numpy.ndarray
        The X_n coordinates of the clicked points.
    y : numpy.ndarray
        The X_{n+1} coordinates of the clicked points.
    """
    num = 200
    xx = np.arange(0, 1.01, 0.01)
    yy = mu * xx * (1 - xx)

    fig, ax = plt.subplots()
    line1, = ax.plot(xx, xx, lw=1.5)
    line2, = ax.plot(xx, yy, lw=1.5)

    x = [0.1]
    z, y_stair, q = [], [], []
    for n in range(num + 1):
        x.append(mu * x[n] * (1 - x[n]))
        z.append([[x[n]], [x[n]]])
        y_stair.append([[x[n]], [x[n + 1]]])
        q.append([[x[n + 1]], [x[n + 1]]])

    z = np.array(z).flatten()
    y_stair = np.array(y_stair).flatten()
    q = np.array(q).flatten()

    # Devil's-staircase cobweb construction.
    line3, = ax.plot(z, y_stair, "m", lw=1.5)
    line4, = ax.plot(y_stair, q, "m", lw=1.5)

    plt.xlabel(r"$X_n$", fontsize=14)
    plt.ylabel(r"$X_{n+1}$", fontsize=14)
    plt.title(r"Logistic Map: $\mu = %s$" % mu)
    plt.grid()
    plt.tight_layout()

    cursor = _DataCursor(plt.gca())
    fig.canvas.mpl_connect("pick_event", cursor)
    for line in (line1, line2, line3, line4):
        line.set_picker(3)  # picking tolerance, in points

    points = plt.ginput(n=n_points)
    np.savetxt("coordinates.txt", points)
    picked_x, picked_y = np.loadtxt("coordinates.txt", usecols=(0, 1), unpack=True)

    plt.show()
    return picked_x, picked_y


def logistic(mumin, mumax):
    """Plot a bifurcation diagram of the logistic map.

    Parameters
    ----------
    mumin : float
        Minimum ``mu`` value to scan (``mu < 4``), e.g. 2.8.
    mumax : float
        Maximum ``mu`` value to scan (``mu < 4``), e.g. 3.9.
    """
    x = 0.1
    num = 200
    warmup = 100

    mu_range = np.linspace(mumin, mumax, 1001)
    xn = np.zeros([mu_range.size, num])

    for j, mu in enumerate(mu_range):
        print(j)
        for n in range(num):
            xn[j, n], x = x, mu * x * (1 - x)

    plt.plot(mu_range, xn[:, warmup:], "r.", ms=0.2)
    plt.grid()
    plt.xlabel(r"$\mu$ value")
    plt.ylabel("x value")
    plt.title("Bifurcation diagram of logistic map")
    plt.tight_layout()
    plt.show()


def logisticmap(mu):
    """Plot the cobweb diagram showing evolution of the logistic map.

    Parameters
    ----------
    mu : float
        Logistic-map growth parameter, ``mu < 4``.
    """
    num = 200
    xx = np.arange(0, 1.01, 0.01)
    yy = mu * xx * (1 - xx)

    plt.figure()
    plt.plot(xx, xx, lw=1.5)
    plt.plot(xx, yy, lw=1.5)

    x = [0.1]
    z, y_stair, q = [], [], []
    for n in range(num + 1):
        x.append(mu * x[n] * (1 - x[n]))
        z.append([[x[n]], [x[n]]])
        y_stair.append([[x[n]], [x[n + 1]]])
        q.append([[x[n + 1]], [x[n + 1]]])

    z = np.array(z).flatten()
    y_stair = np.array(y_stair).flatten()
    q = np.array(q).flatten()

    plt.plot(z, y_stair, "m", lw=1.5)
    plt.plot(y_stair, q, "m", lw=1.5)

    plt.xlabel(r"$X_n$", fontsize=14)
    plt.ylabel(r"$X_{n+1}$", fontsize=14)
    plt.title(r"Logistic Map: $\mu = %s$" % mu)
    plt.grid()
    plt.tight_layout()
    plt.show()
