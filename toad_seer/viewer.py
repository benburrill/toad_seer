import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def make_animation(tdse):
    fig, ax = plt.subplots()
    Pplot, = ax.plot(tdse.x, tdse.pdf())
    Vinit = tdse.cur_pot()
    Vmax = max(abs(np.max(Vinit)), 1)
    Vplot, = ax.plot(tdse.x, tdse.cur_pot()/Vmax)
    ax.set_yticks([])

    def animate(_):
        tdse.evolve(500)

        Pplot.set_ydata(tdse.pdf())
        Vplot.set_ydata(tdse.cur_pot()/Vmax)
        return Pplot, Vplot

    return FuncAnimation(fig, animate, interval=20, blit=True, save_count=50)


def do_animation(tdse):
    ani = make_animation(tdse)
    plt.show()
