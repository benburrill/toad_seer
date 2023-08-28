import numpy as np
from .tdse import TDSE
from .viewer import do_animation

def energy_diff(n1, n2):
    return abs(n1**2 - n2**2)*np.pi**2 / 2

n0 = 1     # Initial state
amp = 1.5  # Perturbation amplitude
trans = [(1, 2), (2, 3)]
dt = 4e-5
x = np.linspace(0, 1, 150)
freqs = [energy_diff(*t) for t in trans]

do_animation(TDSE(
    x, np.sin(n0*np.pi*x), dt=dt,
    V=lambda t: amp*sum(np.sin(freq*t)*x for freq in freqs)
))
