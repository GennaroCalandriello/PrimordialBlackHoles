import numpy as np
import matplotlib.pyplot as plt


def Tgup(xi):
    sign = -1
    a = 5 * 10**5
    GeV = 5.61 * 10**23  # GeV in grams
    Mp = 4.3 * 10 ** (-6)  # Planck mass in grams
    Tlist = []
    xilist = []
    for i in range(len(xi)):
        T = (
            a
            * xi[i]
            / (4 * np.pi)
            * (1 + sign * np.sqrt(1 - ((Mp / (5 * xi[i])) * 10 ** (-5))) ** 2)
        )
        # print(T)
        if T * GeV >= 100:
            # print("baccalaaa")
            Tlist.append(T * GeV)
            xilist.append(xi[i])
    Tlist = np.array(Tlist)
    # T = a * xi / (4 * np.pi) * (1 + sign * np.sqrt(1 - (Mp / (5 * xi)) * 10 ** (-5)))
    return Tlist, xilist


if __name__ == "__main__":
    xi_start = -2  # 10 ** (-5)
    xi_end = 2
    xi = np.linspace(xi_start, xi_end, 3000)
    T, x = Tgup(xi)
    plt.figure()
    plt.title(r"T as a function of $\xi = (\frac{r^2}{\alpha})^{1/2}$")
    plt.xlabel(r"$\xi$")
    plt.ylabel(r"$T_{GUP}$ [GeV]")
    plt.scatter(x, T, s=0.6, color="blue")
    plt.show()
    # plt.plot(xi, T)
    # plt.show()
