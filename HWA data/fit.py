import numpy as np
import matplotlib.pyplot as plt
x = np.array([0,
1.9804134929270942,
4.047878128400436,
6.006528835690968,
8.073993471164309,
10.01088139281828,
12.05658324265506,
14.102285092491838,
16.03917301414581,
18.01958650707291,
20])

y=np.array([ 1.114486921529175,
 1.4054325955734406,
 1.4911468812877264,
 1.5623742454728369,
 1.6251509054325954,
 1.6698189134808852,
 1.7156941649899395,
 1.7507042253521126,
 1.782092555331992,
 1.8171026156941648,
 1.8448692152917503])


def fit(datax, datay):
    mat = np.zeros((datax.shape[0],4))
    for i in range(datax.shape[0]):
        mat[i,0] = 1
        mat[i,1] = datay[i]**2
        mat[i,2] = datay[i]**3
        mat[i,3] = datay[i]**4

    res = np.linalg.lstsq(mat,datax)[0]

    return lambda xi: res[0]+res[1]*xi**2+res[2]*xi**3+res[3]*xi**4

xx = np.linspace(1.1,2)
f = fit(x,y)
plt.plot(f(xx),xx)
plt.grid()
plt.scatter(x,y)
plt.show()

