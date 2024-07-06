import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq
from scipy.interpolate import splev, splrep

plt.close("all")

def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)

def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2

def passeBandeFilter(V, k_1, k_2):
    n = len(V)
    x = np.linspace(0,1,n, endpoint = False)
    xi = fftfreq(n,1/n).astype(int)[:n//2]
    ck = np.abs(fft(V)[:n//2]/(n//2))* ((np.arange(n//2)>k_1) * (np.arange(n//2)<=k_2))
    pk = np.angle(fft(V)[:n//2])
    newV = np.zeros(n, float)
    for k in range(n//2):
        newV +=  ck[k]*np.cos(2*np.pi*xi[k]*x +pk[k])
    return newV

def getMoy(N, k_1, k_2, dt = 1e-7, T = 1e-7, filterbeforematrix = False):
    L = L_mat(N,2*np.pi/N)
    sA = 0
    s = 0
    for i in range(20):
        V = (1-2*(np.random.random(N)<0.5)).astype(float)
        W = passeBandeFilter(V, k_1, k_2)
        if filterbeforematrix:
            sA += W.dot(L.dot(W))
        else:
            sA += W.dot(passeBandeFilter(L.dot(V), k_1, k_2))
        s += W.dot(W)
    print("k_1 = "+str(k_1)+"; k_2 = "+str(k_2)  +" ... done")
    return sA/s


"""Choix des paramètres d'affichage"""
showMeans = True
showInterpotale = False
showcontinu = False
numberofmeans = 10

"""Choix des paramètres de simulation"""
N = 2**10
h = 2*np.pi/N

"""Calculs de l'approximation"""
C = []
F = np.linspace(0,N/2,numberofmeans)[:].astype(int)
for i in range(numberofmeans - 1):
    C.append(getMoy(N, F[i],F[i+1]))
#F = np.logspace(4,10,6,  base = 2, endpoint = True).astype(int)

"""Calcul des symboles théoriques"""
xi = np.arange(N//2)
symbol_continu = xi**2
symbol_discret = 2/h**2 * (1-np.cos(h*xi))

"""Affichage""" 
plt.figure(1)
if showcontinu:
    plt.plot(xi, symbol_continu, "k--", label = "Symbole de l'opérateur continu")
plt.plot(xi, symbol_discret, "r--", label = "Symbole de l'opérateur discret")
plt.ylim(-symbol_discret[-1]*0.1,symbol_discret[-1]*1.1)
for i in range(numberofmeans-1):
    if i ==0:
        if showMeans:
            if showcontinu:
                mc = np.mean(symbol_continu[1:F[i]])
                plt.plot([1,F[i]], [mc, mc], "ko-", label = "Moyennes locales du cas continu")
            ms = np.mean(symbol_discret[1:F[i]])
            plt.plot([1,F[i]], [ms, ms], "ro-", label = "Moyennes locales du cas discret")
        plt.plot([1,F[0]], [C[0], C[0]], "b+-", label = "Moyennes locales approchées")
    if showMeans:
        ms = np.mean(symbol_discret[F[i]+1:F[i+1]])
        plt.plot([F[i]+1,F[i+1]], [ms, ms], "ro-")
        if showcontinu:
            mc = np.mean(symbol_continu[F[i]+1:F[i+1]])
            plt.plot([F[i]+1,F[i+1]], [mc, mc], "ko-")
    plt.plot([F[i]+1,F[i+1]], [C[i], C[i]], "b+-")
if showInterpotale:
    plt.plot(np.hstack((0,F[0]/2,0.5*(F[1:] + F[:-1]), F[-1])), np.hstack((C[0] - F[0] * (C[1]-C[0])/(F[2]-F[0]) ,C,C[-1] - (F[-1]-F[-2]) * (C[-1]-C[-2])/(F[-3]-F[-1]) )), "b-.")        
    plt.plot(np.hstack((F[0]/2,0.5*(F[1:] + F[:-1]))), C, "bo")
plt.xlabel("$k$")
plt.ylabel("$\\sigma(k)$")
plt.legend()
plt.show()

