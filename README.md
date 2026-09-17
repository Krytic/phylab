# phylab

> [!NOTE]
> This package was developed for the University of Auckland Department of Physics Advanced Lab course to support our teaching. Whilst others are welcome to use it if they find it useful, it assumes a lot about our specific setup, and isn't really designed for general use. Support to non-students will thus necessarily be limited.

Data analysis tools for the University of Auckland undergraduate physics
lab courses (curve fitting, ODE integration, and the e314 chaotic-pendulum
and e315 double-pendulum experiments).

This package is a repackaged, PEP 8-cleaned version of the original
`phylab.py` script. Every function keeps its original name and signature
(with a couple of documented exceptions below), so existing lab handouts
and notebooks keep working after switching to `import phylab`.

## Installation

From the project directory (the one containing `pyproject.toml`):

```bash
pip install .
```

For local development, install in editable mode so changes to the source
take effect immediately:

```bash
pip install -e .
```

## Usage

Everything is available directly from the top-level package:

```python
import phylab

# Curve fitting
p, sp, chi2 = phylab.nonlinft(model, x, y, sy, p0, v)

# Chaotic pendulum
phylab.pendulum(g=1.2, q=2, tmax=6000)

# Double pendulum
t, p1, p2, sampling_period, _ = phylab.acquire("run1.txt", show_fig=False)
```

If you prefer more explicit namespacing, the same functions are also
available from their topic submodules:

```python
from phylab.fitting import nonlinft
from phylab.chaotic_pendulum import pendulum
from phylab.double_pendulum import acquire, period
```

## Module layout

| Submodule                 | Contents                                                                 |
|----------------------------|---------------------------------------------------------------------------|
| `phylab.fitting`           | `nonlinft`, `regress` — non-linear (Levenberg-Marquardt) and polynomial regression |
| `phylab.odesolvers`        | `rk4` — a general-purpose fourth-order Runge-Kutta integrator            |
| `phylab.dataio`            | `load_dampshm`, `load_co57`, `importElvisOsc`, `importElvisBode`, `importElvis2wire` — file loaders |
| `phylab.optics`            | `photoproc`, `photocalib` — diffraction-photo digitisation tools         |
| `phylab.chaotic_pendulum`  | `pendeq`, `bifurc`, `ellipmodel`, `pendulum`, `ginput`, `logistic`, `logisticmap` — e314 |
| `phylab.double_pendulum`   | `acquire`, `findlyap`, `mom`, `pendcut`, `pendlyap`, `period` — e315      |

## Changes from the original script

The numerical algorithms are unchanged. To make the code installable,
readable, and runnable on current NumPy/SciPy/Matplotlib, the following
was cleaned up:

- Split the single ~1450-line file into topic-based submodules, all
  re-exported from the top-level `phylab` package.
- Reformatted to PEP 8 (naming, whitespace, removed stray semicolons,
  added docstrings for every public function).
- Replaced NumPy/SciPy APIs that have since been removed, e.g.
  `np.float_` / `np.float` -> `float`, `np.error(...)` ->
  `raise RuntimeError(...)`, and `scipy.integrate.odepack.odeint` ->
  `scipy.integrate.odeint`.
- Fixed a couple of bugs that would otherwise crash under current
  library versions: `rk4` now actually appends `tfinal` to its output
  times instead of discarding the result of `np.append`; float step
  counts are now cast to `int` before being used as array sizes; and
  `findlyap` now takes its time vector `t` as an explicit parameter
  instead of relying on it existing as a global variable.
- `pendlyap` no longer uses `global` pendulum-parameter variables to
  communicate with its nested ODE function; the nested function is now
  a module-level helper (`phylab._double_pend_ode`... internally,
  `double_pendulum._double_pend_ode`) that takes the parameters
  explicitly. Behaviour is unchanged.

One quirk preserved from the original code: in `pendlyap`'s
Gram-Schmidt re-orthonormalisation loop, the first orthogonalisation
step only starts contributing at the third base vector (index 2) rather
than the second (index 1). This matches the original implementation; if
you rely on `pendlyap` for exponent values you intend to publish, it is
worth independently verifying that loop against your reference (e.g.
Wolf et al., Physica D 16, 285 (1985)).

## Notes on individual functions

- `phylab.ginput(mu, n_points)` opens an interactive matplotlib window
  and lets you click points on a logistic-map plot; it is unrelated to
  (and does not override) `matplotlib.pyplot.ginput`, which it uses
  internally.
- `phylab.photoproc` and `phylab.photocalib` are interactive: they open
  a window and wait for you to click on it, so they only work in an
  environment with a GUI-capable Matplotlib backend.
- `load_dampshm` and `load_co57` look for their `.mat` files in the
  current working directory, falling back to a legacy Windows/MATLAB
  path from the original lab machines.

## Requirements

- Python >= 3.9
- numpy, scipy, matplotlib (installed automatically as dependencies)
