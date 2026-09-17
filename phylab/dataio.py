"""Loaders for the various data file formats used in phylab experiments."""

import numpy as np
from scipy.io import loadmat

__all__ = [
    "load_dampshm",
    "load_co57",
    "importElvisOsc",
    "importElvisBode",
    "importElvis2wire",
]

# Fallback locations used by the original MATLAB-era lab setups, kept for
# backward compatibility with older lab machines.
_DAMPSHM_FALLBACK = "C:/Program Files/MATLAB/R2012b/toolbox/matlab/local/dampshm.mat"
_CO57_FALLBACK = "C:/Program Files/MATLAB/R2012b/toolbox/matlab/local/co57.mat"


def load_dampshm():
    """Load the damped simple-harmonic-motion demonstration dataset.

    Looks for ``dampshm.mat`` in the current directory first, falling back
    to the legacy MATLAB toolbox location.

    Returns
    -------
    t : numpy.ndarray
        Time values.
    x : numpy.ndarray
        Position values.
    """
    try:
        data = loadmat("dampshm.mat")
    except FileNotFoundError:
        data = loadmat(_DAMPSHM_FALLBACK)
    x = data["position"].reshape(-1)
    t = data["time"].reshape(-1)
    return t, x


def load_co57():
    """Load the Co-57 gamma-ray spectrum demonstration dataset.

    Looks for ``co57.mat`` in the current directory first, falling back to
    the legacy MATLAB toolbox location.

    Returns
    -------
    x : numpy.ndarray
        Channel numbers.
    y : numpy.ndarray
        Counts per channel.
    """
    try:
        data = loadmat("co57.mat")
    except FileNotFoundError:
        data = loadmat(_CO57_FALLBACK)
    x = data["chan"].reshape(-1)
    y = data["counts"].reshape(-1)
    return x, y


def importElvisOsc(theFile):
    """Import an ELVIS II oscilloscope log file.

    Parameters
    ----------
    theFile : str
        Path to the oscilloscope log file.

    Returns
    -------
    numpy.ndarray
        The parsed data, with any HH:MM:SS timestamp columns converted to
        seconds.
    """
    try:
        data = np.genfromtxt(
            theFile, skip_header=5, usecols=(1, 2, 4, 5), encoding="utf-8", dtype="|U32"
        )
        for row in data:
            row[0] = _hms_to_seconds(row[0])
            row[2] = _hms_to_seconds(row[2])
    except (ValueError, IndexError):
        data = np.genfromtxt(
            theFile, skip_header=5, usecols=(1, 2), encoding="utf-8", dtype="|U32"
        )
        for row in data:
            row[0] = _hms_to_seconds(row[0])
    return data.astype(float)


def _hms_to_seconds(timestamp):
    """Convert an ``HH:MM:SS`` string to a number of seconds (as a str)."""
    multipliers = (3600, 60, 1)
    parts = (float(value) for value in timestamp.split(":"))
    return sum(m * v for m, v in zip(multipliers, parts))


def importElvisBode(theFile):
    """Import an ELVIS II Bode-plot log file."""
    return np.genfromtxt(theFile, skip_header=3)


def importElvis2wire(theFile):
    """Import an ELVIS II two-wire measurement log file."""
    return np.genfromtxt(theFile, skip_header=2)
