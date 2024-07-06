import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.close("all")
def divarr(N,D):
    return np.divide(N,D, out = np.zeros_like(N, dtype = float), where = D!=0)
    


'''Définition des fonctions b,U_initial et H_initial'''
def energy(H,U,B, eps=1, beta=1):
    return np.sum(eps*H*U*U + H*H + 2 * beta * H * (beta*B -1))
def defaultu0(X):
    return X*0
def defaultzeta0(X):
    return X*0
def defaultbottom(X):
    return X*0
class WBHLLSW:
    def __init__(self, a, b, N, beta=1, eps=1, speed0=defaultu0, surface0=defaultzeta0, bottom=defaultbottom):
        self.N = N
        self.X, self.dx = np.linspace(a, b, N, endpoint = False, retstep = True)
        
        self.beta = beta
        self.eps = eps
        
        self.B = bottom(self.X)
        self.G = self.beta * self.B - 1
        
        self.e = surface0(self.X)
        self.H = self.eps * self.e - self.G
        
        self.U = speed0(self.X)
        
        
        #self.U = (self.H>0).astype(float)*0
        self.UH = self.U*self.H
        
        self.meanU = []
        self.meanH = []
        self.energyHU = []
    
    '''Definition des fonctions de reconstruction de h'''
    def hg(self):
        return np.maximum(0,self.H+self.beta * (self.B-np.maximum(self.B,np.roll(self.B,-1)))) #le max exterieur permet de s'assurer que les hauteurs sont positive, le max ingterieur permet de s'assurer que l'on vérifie l'inégalité d'entropie IV de Leveque
    def hd(self):
        return np.roll(np.maximum(0,self.H+self.beta * (self.B-np.maximum(self.B,np.roll(self.B,1)))),-1)
    
    """fonction flux (se fait en 3 étape)"""   
    #étape primitive 
    def fh1(self, H, U):
        return self.eps*H*U
    def fp1(self, H, U):
        return self.eps*H*U*U+0.5*H*H
    #étape conservative
    def fh2(self, H1, H2, U1, U2, c1, c2):
        F = self.fh1(H1,U1)
        F[(c1<0)] = (c2*self.fh1(H1,U1) - c1*self.fh1(H2,U2) + c2*c1*(H2-H1))[(c1<0)]/(c2-c1)[(c1<0)]
        F[(c2<=0)] = self.fh1(H2,U2)[(c2<=0)]
        return F
    def fp2(self, H1, H2, U1, U2, c1, c2):
        F = self.fp1(H1,U1)
        F[(c1<0)] = (c2*self.fp1(H1,U1) - c1*self.fp1(H2,U2) + c2*c1*(U2*H2-U1*H1))[(c1<0)]/(c2-c1)[(c1<0)]
        F[(c2<=0)] = self.fp1(H2,U2)[(c2<=0)]
        return F
    #étape bien balancée
    def fh3(self, H1, H2, U1, U2, c1, c2):
        return self.fh2(H1, H2, U1, U2, c1, c2)
    def fp3(self, H1, H2, U1, U2, c1, c2):
        return self.fp2(H1, H2, U1, U2, c1, c2)
    
    """fonctions source"""
    def p(self, H):
        return 0.5 * H*H
    def Sg(self, H,Hg):
        return self.p(H) - self.p(Hg)
    def Sd(self, H,Hd):
        return self.p(np.roll(H,-1)) - self.p(Hd)
    
    """fonctions principales"""
    def DUDT(self):
        H1 = self.hg()
        H2 = self.hd()
        U2 = np.roll(self.U,-1)
        c1 = np.minimum(self.U - np.sqrt(H1), U2 - np.sqrt(H2))
        c2 = np.maximum(self.U + np.sqrt(H1), U2 + np.sqrt(H2))
        Fh = self.fh3(H1, H2, self.U, U2, c1, c2)
        Fp = self.fp3(H1, H2, self.U, U2, c1, c2)
        return (Fh - np.roll(Fh,1))/self.dx, (Fp + self.Sg(self.H,H1) - np.roll(Fp + self.Sd(self.H,H2),1))/self.dx
    
    def tick(self, dt):
        dh, duh = self.DUDT()
        self.H += - dt * dh
        self.UH += - dt * duh
        self.U = divarr(self.UH, self.H)
    def storeData(self):
        self.meanU.append(np.sum(self.U)*self.dx)
        self.meanH.append(np.sum(self.H)*self.dx)
        self.energyHU.append(energy(self.H,self.U,self.B, self.eps, self.beta)*self.dx)
    def getMeans(self):
        return self.meanU, self.meanH
    def getEnergy(self):
        return self.energyHU
    def getSurface(self):
        return self.H + self.G
    def getFloor(self):
        return self.G
    def getSpeed(self):
        return self.U
    def getX(self):
        return self.X
    def setSurface(self, S):
        self.H = S - self.G
        self.UH = self.U*self.H
    def setSpeed(self, U):
        self.U = np.copy(U)
        self.Uh = self.H*self.U
"""Paramètres de la simulation"""
L = 100
a = -L/2; b = L/2
N = 4000
dt = 0.004
T = 20
ipf = 20 #ititérations par frame

def b_func(X):
    return 0.5*(1 + np.sin(2*300*np.pi*X/L))/2

"""Construction d'une solution initiale unidirectionnelle"""
def zeta_init(X):
    return np.exp(-20*(X)**2)*2
wbhllsw0 = WBHLLSW(a, b, N, beta=1, eps=1, surface0=zeta_init)
t = 0
while t<5:
    wbhllsw0.tick(dt)
    t+=dt
S0 = wbhllsw0.getSurface() 
U0 = wbhllsw0.getSpeed()

def sigmoid(X):
    return 1/(1+np.exp(np.minimum(X, 80)))
def smoothcut(V,X,x_min, x_max, strengh = 1):
    return V * (1 - sigmoid((X - x_min)*strengh)) * sigmoid((X - x_max)*strengh)
S0 = np.roll(smoothcut(S0, wbhllsw0.getX(),0, L/4), N//2)
U0 = np.roll(smoothcut(U0, wbhllsw0.getX(),0, L/4), N//2)

plt.figure(11)
plt.plot(sws[0].getX(), sws[0].getSurface())
plt.plot(sws[0].getX(), sws[0].getFloor())
plt.show()
plt.pause(1)

"""Initialisation de la simulation"""
def F(U):
    return np.abs(fft(U)[:N//2])/N/L

freq = fftfreq(N)[:N//2]/L

#Fond plat
wbhllsw0 = WBHLLSW(a, b, N, beta=1, eps=1)
wbhllsw0.setSpeed(U0)
wbhllsw0.setSurface(S0)

"""Simulation"""
t=0
while t<T:
    wbhllsw0.tick(dt)
    t+=dt
    
"""Comparison"""
arg = 500


plt.figure(1, figsize = (10,3))
plt.clf()
plt.plot((-2*(F(wbhllsw0.getSurface()) - F(S0))/(T*(1e-14 + F(wbhllsw0.getSurface())+F(S0))))[:arg], "k")
plt.ylabel("symbole $\\sigma(k)$")
plt.xlabel("fréquence k")
plt.legend()
plt.show()  






