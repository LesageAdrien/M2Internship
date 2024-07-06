import numpy as np
import scipy.sparse as sps
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
plt.close('all')

def defaultuinit(X):
    return 5*np.exp(-1*(X**2))
def Finv(fu):
    return np.roll(fft(fu)[::-1], 1)
class FourierKDV:
    def __init__(self, N, a, b, u0 = defaultuinit):
        self.N = N
        self.xi =  fftfreq(N, (b-a)/N/2/np.pi)
        self.X, self.h = np.linspace(a, b, N, retstep = True, endpoint = False)
        self.U0 = u0(self.X)
        self.FU = fft(self.U0)/self.N
        self.eps = 1e-6
        self.itermax = 25
    def tick(self, dt):
        FUk = np.copy(self.FU)
        res = 1 + self.eps
        k = 0
        while res > self.eps: 
            k+=1
            FUknew = self.FU *((2-1j*dt*self.xi**3) / (2+1j*dt*self.xi**3)) +  (dt*1j*self.xi)/(8 - 4j*dt*self.xi**3)*fft(Finv(FUk + self.FU)**2)/self.N
            res = np.linalg.norm(FUk - FUknew)/self.N
            FUk = np.copy(FUknew)
            if k > self.itermax:
                break
        self.FU = FUk
    def getU(self):
        return fft(self.FU)
    def getFU(self):
        return np.copy(self.FU)
    def getX(self):
        return self.X

def soliton(c,X):
    return 3*c/(1 + np.sinh(0.5*np.sqrt(c)*X)**2)
def u1(X):
    return soliton(2,X)  #  + soliton(4, X-10)

L = 50
N = 2000
a = -L/2
b = L/2   
FKDV = FourierKDV(N, a, b,u0=u1)
FU0 = FKDV.getFU()[:N//2]
dt = 1e-3
T = 10
spg = []
t = 0
i = 0
while t<T:
    i+=1
    if i%100==0:
        plt.figure(1)
        plt.clf()
        plt.title("KDV by Fourier  t = "+str(round(t,4)))
        plt.plot(FKDV.getX(), FKDV.getU())
        plt.show()  
        plt.pause(0.01)
        spg.append(np.abs(FKDV.getFU()[:N//2]-FU0))
        plt.tight_layout()
    FKDV.tick(dt)
    t+=dt
plt.figure(1)
plt.clf()
plt.title("KDV by Fourier  t = "+str(round(t,4)))
plt.plot(FKDV.getX(), FKDV.getU())
plt.show()  
plt.pause(0.01)
plt.tight_layout()
"""Affichage du spectrogramme"""
xi = np.linspace(0,1, N//2)*N/L/np.pi 
plt.figure(2, figsize=(10,3))
plt.clf()
plt.imshow( np.array(spg).T[::-1], extent=[0,T,0,xi[-1]], aspect='auto', vmin = 0, interpolation= "gaussian")
plt.xlabel("temps")
plt.ylabel("fréquence")
plt.colorbar()
plt.tight_layout()
plt.show()