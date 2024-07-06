import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq

plt.close("all")


def sproll(M, k=1):
    return sps.hstack((M[:, k:], M[:, :k]), format="csr", dtype=float)


def B_mat(N, h):
    I = sps.eye(N, format="csr", dtype=float)
    return (-sproll(I, 2) + 2 * sproll(I, 1) - 2 * sproll(I, -1) + sproll(I, -2)) / (2 * h ** 3)


def D_mat(N, h):
    I = sps.eye(N, format="csr", dtype=float)
    return (sproll(I, -1) - sproll(I, 1)) / (2 * h)


def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2


def approxSquare(V1,V2):
    return (V1**2 + V1*V2 + V2**2)/3

def energy(U):
    return np.sum(-h*L.dot(U)*U/2 + h*U**3/6)

"""Données relative à la discrétisation du Tore"""
length = 2
a = -length * np.pi;
b = length * np.pi
N = 100
X, h = np.linspace(a, b, N, endpoint=False, retstep=True)

"""Données relative a la FFT"""

xfreq = fftfreq(N, 1 / N) / (b - a) * 2 * np.pi
firstfreq_ratio = 10
firstfreq = xfreq[1:N // firstfreq_ratio]

"""Construction des matrices intervenant dans la boucle"""
D = D_mat(N, h)
B = B_mat(N, h)
L = L_mat(N, h)
I = sps.eye(N, format="csr", dtype=float)

"""Construction des données initiales et des paramètres d'évolution en temps"""

# U = np.sin(2*X) + 0.1*np.sin(3*X) + 0.2 *np.sin(10 * X)
# U = np.sin(2 * X)
U = 6 * np.exp(-30 * (X) ** 2) + 6
# U = (np.abs(X)< length/3).astype(float) #signal carré
# U = np.maximum(0, length/3 - np.abs(X)) #signal triangle


E = [energy(U)]
L2 = [np.linalg.norm(U)*np.sqrt(h)]
Mean = [np.sum(h*U)]


waveheight = np.max(np.abs(U))

T = 1;
dt = 1e-3;
t = 0
ipf = 10
Nt = int(T/dt+1)
Tlist = [0]

alpha = 0
theta = 0.5
Mat = I + dt * theta * B

"""Execution de la boucle"""
i = 0
while i < Nt:

    """Calcul du prochain U"""
    # Unew = sps.linalg.spsolve(I + dt * B,  U) # Formulation implicite
    res = 1
    Uk = np.copy(U)
    k = 1
    while res > 1e-14:
        k += 1
        Uknew = sps.linalg.spsolve((alpha + 1) * Mat,
                                   alpha * Mat.dot(Uk) + U - dt * (1 - theta) * B.dot(U) - 0.5 * dt * D.dot(
                                       approxSquare(Uk,U)))  # Formulation Sanz Serna + itération de picard
        res = np.linalg.norm(Uk - Uknew) * np.sqrt(h)
        Uk = np.copy(Uknew)
    Unew = np.copy(Uk)
    i+=1

    """Affichage des données concernant l'evolution de la norme L2 de U"""
    E.append(energy(Unew))
    L2.append(np.linalg.norm(Unew)*np.sqrt(h))
    Mean.append(np.sum(h*U))
    Tlist.append(t)
    if i%ipf == 0:
        print(str(int(i/Nt*100))+"%")
        
    """Actualisation de U et de t"""
    U = np.copy(Unew)
    t += dt
    
plt.figure(0)
plt.clf()
plt.plot(Tlist, Mean)
plt.xlabel("t")
plt.title("$\\int u$")
plt.show()
plt.figure(1)
plt.clf()
plt.plot(Tlist, L2)
plt.xlabel("t")
plt.title("$\\int u^2$")
plt.show()
plt.figure(2)
plt.clf()
plt.plot(Tlist, E)
plt.xlabel("t")
plt.title("$J(u)$")
plt.show()
    


