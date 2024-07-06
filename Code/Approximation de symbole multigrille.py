import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.fft import fft, fftfreq
from scipy.interpolate import splev, splrep
plt.close("all")


"""Definition de la matrice"""
def sproll(M,k=1):
    return sps.hstack((M[:,k:], M[:,:k]), format = "csr", dtype = float)
def L_mat(N,h):
    I = sps.eye(N,format = "csr", dtype = float)
    return (2*sproll(I,0) - sproll(I,1) - sproll(I,-1))/h**2

"""Sous echantillonage avec anti-aliasing"""
AAkernel = np.array([1,2,1])
def Si(V,i):
    if i==0:
        return V
    else:
        res = 0
        for l, k in enumerate(AAkernel):
            res = res + k*np.roll(V,l)
        return Si(res[::2]/AAkernel.sum(),i-1)
def SiINV(V,i):
    if i==0:
        return V
    return SiINV(np.ravel(np.vstack((V,(V+np.roll(V,-1))/2)).T)[:-1], i-1)
"""Récupération des moyennes m_i sur les bandes [[0, k_i]]"""
def getMoy1(imax, i, dt = 1e-7, T = 1e-6):
    L = L_mat(2**imax,2*np.pi/2**imax)
    sA = 0
    s = 0
    for k in range(20):
        V = (1 - 2*(np.random.random(2**imax)<0.5)).astype(float)
        sA += SiINV(Si(L.dot(V), imax-i), imax-i).dot(SiINV(Si(V, imax-i), imax-i))
        s += SiINV(Si(V, imax-i), imax-i).dot(SiINV(Si(V, imax-i), imax-i))
    print("nf = "+str(2**i)+" ... done")
    return sA/s
def getMoy2(imax, i, dt = 1e-7, T = 1e-6):
    L = L_mat(2**imax,2*np.pi/2**imax)
    sA = 0
    s = 0
    for k in range(20):
        V = (1 - 2*(np.random.random(2**imax)<0.5)).astype(float)
        sA +=  Si(L.dot(V), imax-i).dot(Si(V, imax-i))
        s +=  Si(V, imax-i).dot(Si(V, imax-i))
    print("nf = "+str(2**i)+" ... done")
    return sA/s

"""Récupération des moyennes sur les bandes [[k_i, k_{i+1}-1]]"""
def getC(F,M):
    return np.hstack(( M[0],  ( M[1:]*F[1:] - M[:-1]*F[:-1] )/( F[1:]-F[:-1] ) ))

"""Paramètres"""
imax = 11
imin = 3
i_list = (imin + np.arange(imax+1-imin))
N = 2**imax
F = 2**(i_list-1)
h = 2*np.pi/N

"""Execution"""
M = [getMoy2(imax, i) for i in i_list]
C = getC(F, M)

"""Affichage"""
xi = np.arange(N//2)
showcontinu = False
showMeans = True
showInterpolate= False
symbol_continu = xi**2
symbol_discret = 2/h**2 * (1-np.cos(h*xi))
plt.figure(1)
if showcontinu:
    plt.plot(xi, symbol_continu, "k--", label = "Symbol de l'opérateur continu")
plt.plot(xi, symbol_discret, "r--", label = "Symbol de l'opérateur discret")
plt.ylim(-symbol_discret[-1]*0.1,symbol_discret[-1]*1.1)
for i in range(len(F)-1):
    if i ==0:
        if showMeans:
            if showcontinu:
                mc = np.mean(symbol_continu[1:F[i]])
                plt.plot([1,F[i]], [mc, mc], "ko-", label = "Moyennes locales du cas continu")
            ms = np.mean(symbol_discret[1:F[i]])
            plt.plot([1,F[i]], [ms, ms], "ro-", label = "Moyennes locales du cas discret")
            plt.plot([1,F[0]], [C[0], C[0]], "b+-", label = "Moyennes locales approximées")
    if showMeans:
        ms = np.mean(symbol_discret[F[i]+1:F[i+1]])
        plt.plot([F[i]+1,F[i+1]], [ms, ms], "ro-")
        if showcontinu:
            mc = np.mean(symbol_continu[F[i]+1:F[i+1]])
            plt.plot([F[i]+1,F[i+1]], [mc, mc], "ko-")
        plt.plot([F[i]+1,F[i+1]], [C[i+1], C[i+1]], "b+-")
if showInterpolate:
    plt.plot(np.hstack((0,F[0]/2,0.5*(F[1:] + F[:-1]), F[-1])), np.hstack((C[0] - F[0] * (C[1]-C[0])/(F[2]-F[0]) ,C,C[-1] - (F[-1]-F[-2]) * (C[-1]-C[-2])/(F[-3]-F[-1]) )), "b--")        
    plt.plot(np.hstack((F[0]/2,0.5*(F[1:] + F[:-1]))), C, "bo")
plt.xlabel("$k$")
plt.ylabel("$\\sigma(k)$")
plt.legend()
plt.show()
