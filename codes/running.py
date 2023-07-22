import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""References: Wilczek arXiv 0812.4946,
for g, gprime and gs see: Planck scale black hole dark matter from Higgs inflation"""

# define the beta functions

Mpl = 10 ** (5)

v_SM = 1900  # SM energy scale
start = -v_SM
chi = np.linspace(start, Mpl * 3, 1000)
h0 = start


def beta_xi(xi, lamb, yt, g, gp, gs):
    # beta function for xi, depends on xi and eta
    beta_xi = (
        (6 * xi + 1)
        / (16 * np.pi**2)
        * (2 * lamb + yt**2 - (3 / 4) * g**2 - (1 / 4) * gp**2)
    )
    return beta_xi


def beta_lamb(xi, lamb, yt, g, gp, gs):
    # beta function for eta, depends on xi and eta
    beta_lamb = (1 / (16 * np.pi**2)) * (
        24 * lamb**2
        + -6 * yt**4
        + (9 / 8) * g**4
        + (3 / 4) * g**2 * gp**2
        + 3 / 8 * gp**4
        + lamb * (12 * yt**2 - 9 * g**2 - 3 * gp**2)
    )
    return beta_lamb


def beta_yt(xi, lamb, yt, g, gp, gs):
    beta_yt = (yt / (16 * np.pi**2)) * (
        (9 / 2) * yt**2 - 8 * gs**2 - 9 / 4 * g**2 - 17 / 12 * gp**2
    )
    return beta_yt


def beta_g(g):
    beta_g = (1 / 16 * np.pi**2) * (-13 / 4 * g**3)
    return beta_g


def beta_gp(gp):
    beta_gp = (1 / 16 * np.pi**2) * (27 / 4 * gp**3)
    return beta_gp


def beta_gs(gs):
    beta_gs = (1 / 16 * np.pi**2) * (-7 * gs**3)
    return beta_gs


# define the potential function
def potential(h, xi, lamb):
    # This function should return the value of the potential given the field value and the current values of xi and eta.
    V = (lamb * (h**2 - v_SM**2) ** 2) / (4 * (1 + xi * h**2 / Mpl**2) ** 2)
    return V


def Upot(chi, xi, lamb):
    U = (
        lamb
        * Mpl**4
        * (1 / (4 * xi**2))
        * (1 + np.exp(-(2 * chi / np.sqrt(6) * Mpl))) ** (-2)
    )
    return U


def dh_dchi(chi, h, xi):
    Omega2 = 1 + xi * h**2 / Mpl**2  # conformal factor
    dhdchi = Omega2 * 1 / (np.sqrt(Omega2 + xi * (6 * xi + 1) * h**2 / Mpl**2))
    # chi is the canonically normalized background field (background? sure?)

    return dhdchi


def running_():
    lamb, xi, g, gp, gs, yt = initial_value_parameters()
    xi_list = []
    for k in range(len(chi)):
        # running coupling on the parameters
        xi += beta_xi(xi, lamb, yt, g, gp, gs)
        lamb += beta_lamb(xi, lamb, yt, g, gp, gs)
        yt += beta_yt(xi, lamb, yt, g, gp, gs)
        g += beta_g(g)
        gp += beta_gp(gp)
        gs += beta_gs(gs)

        xi_list.append(xi)

    xi_list = np.array(xi_list)
    # print(xi_list)
    # solve the differential equation

    return xi_list


def RungeKutta(f, y0):
    xi_ = running_()
    y = np.zeros(len(chi))
    y[0] = y0

    for i in range(0, len(chi) - 1):
        dchi = chi[i + 1] - chi[i]

        k1 = dchi * f(chi[i], y[i], xi_[i])
        k2 = dchi * f(chi[i] + dchi / 2, y[i] + k1 / 2, xi_[i])
        k3 = dchi * f(chi[i] + dchi / 2, y[i] + k2 / 2, xi_[i])
        k4 = dchi * f(chi[i] + dchi, y[i] + k3, xi_[i])

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y


def initial_value_parameters():
    #!!!!!!!!one must set this initial values correctly!!!!!!!
    lamb_initial = 0.001
    xi_initial = 10**4 * np.sqrt(lamb_initial)
    g_initial = 0.01
    gp_initial = 0
    gs_initial = 0
    yt_initial = 0  # 0.01

    return lamb_initial, xi_initial, g_initial, gp_initial, gs_initial, yt_initial


# define a function to simulate the evolution of the field
def simulate_field(num_steps=len(chi)):
    # initialize the field value, xi and eta
    lamb, xi, g, gp, gs, yt = initial_value_parameters()
    h = RungeKutta(dh_dchi, h0)
    # we will store the history of the field, xi and eta values in these lists
    # phi_history = [phi]
    xi_history = [xi]
    lamb_history = [lamb]
    yt_history = [yt]
    gp_history = [gp]
    g_history = [g]
    gs_history = [gs]
    V = []
    for step in range(num_steps):
        # update xi and eta according to their beta functions
        xi += beta_xi(xi, lamb, yt, g, gp, gs)
        lamb += beta_lamb(xi, lamb, yt, g, gp, gs)
        yt += beta_yt(xi, lamb, yt, g, gp, gs)
        g += beta_g(g)
        gp += beta_gp(gp)
        gs += beta_gs(gs)

        # calculate the potential
        V.append(potential(h[step], xi, lamb))
        # V.append(Upot(h[step], xi, lamb))

        # store the new field, xi and eta values
        # phi_history.append(phi)
        xi_history.append(xi)
        lamb_history.append(lamb)
        yt_history.append(yt)
        g_history.append(g)
        gp_history.append(gp)
        gs_history.append(gs)

    return V, xi_history, lamb_history, yt_history, g_history, gp_history, gs_history


if __name__ == "__main__":
    V, xi_, lamb_, yt_ = simulate_field()[0:4]
    h = RungeKutta(dh_dchi, h0)  # + v_SM
    h = h / max(h)
    V = np.array(V)
    V = V / (lamb_[0] * Mpl**4)
    plt.figure()
    plt.plot(h, V, color="blue", label="V")
    plt.xlabel(r"$\chi$")
    plt.ylabel(r"V/$V_0$")
    plt.show()

    # plt.figure()
    # plt.plot(range(len(yt_)), yt_, color="red", label="V")
    # plt.plot(range(len(yt_)), lamb_, color="blue", label="V")
    # plt.show()
