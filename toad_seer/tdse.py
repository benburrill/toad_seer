import numpy as np

fade_in = np.array([0]) #((1-np.cos(np.linspace(0, np.pi, 5)))/2)
fade_out = fade_in[::-1]

def zero_boundary(f):
    f[0] = 0
    f[-1] = 0
    # f[:len(fade_in)] *= fade_in
    # f[-len(fade_out):] *= fade_out


def d2(f):
    diff = -2 * f
    diff[:-1] += f[1:] # f(x+dx)
    diff[1:] += f[:-1] # f(x-dx)
    diff[0] -= f[1]
    diff[-1] -= f[-2]
    return diff


class TDSE:
    def __init__(self, x, phi0, *, V=0, m=1, dt=1e-5):
        self.x = x
        # self._scratch = np.empty(x.shape)
        self.dx2_2m = 2 * m * np.gradient(x)**2
        self.step = 0
        self.V = V if callable(V) else lambda _: V
        self.dt = dt

        self.set_phi(phi0)

    def H(self, f, t):
        return self.V(t) * f - d2(f)/self.dx2_2m

    def set_phi(self, phi, copy=True):
        if copy:
            phi = phi.copy()

        self.R = phi.real
        # propagate forward and backwards in time by half a time-step
        # is this a good idea?  Not sure, but why not!
        I_step = (self.dt/2) * self.H(self.R, self.step * self.dt)
        self.I = phi.imag - I_step
        self.I_prev = phi.imag + I_step
        # self.I = phi.imag + 0

        self.initial_norm = self.norm()

    def get_phi(self):
        result = np.array(self.R, dtype=complex)
        result.imag = self.I
        result.imag += self.I_prev
        result.imag /= 2

        return result

    def cur_pot(self):
        return np.broadcast_to(self.V(self.dt * self.step), self.x.shape)

    def inner_product(self, bra, f=1, *, renorm=False):
        # <bra|f|self>, where bra is a normalized, spatial complex wf
        # with same shape as self.x.
        norm = self.norm() if renorm else self.initial_norm
        if callable(f):
            raise NotImplementedError
        else:  # f is a function of x
            # should we in some way involve I_prev?
            # TODO: this seems inefficient
            return np.trapz(
                bra.conjugate() * f * self.get_phi(),
                self.x
            ) / np.sqrt(norm)

    def inner_squared(self, *args, **kwargs):
        # |<bra|f|self>|^2
        inner = self.inner_product(*args, **kwargs)
        return (inner.conjugate() * inner).real

    def norm(self):
        # <self|self>
        return np.trapz(self.pdf(), self.x)

    def expected(self, f, *, renorm=False):
        # <self|f|self>
        norm = self.norm() if renorm else self.initial_norm
        if callable(f):
            raise NotImplementedError
        else:  # f is a function of x
            return np.trapz(f * self.pdf(), self.x) / norm

    def pdf(self):
        return self.R * self.R + self.I_prev * self.I

    @staticmethod
    def momentum():
        raise NotImplementedError

    def evolve(self, steps=1):
        for _ in range(steps):
            self.step += 1
            zero_boundary(self.I)
            self.R += self.dt * self.H(self.I, self.dt * (self.step-0.5))

            self.I_prev = self.I.copy()

            zero_boundary(self.R)
            self.I -= self.dt * self.H(self.R, self.dt * self.step)

        # TODO: maybe renormalize?
