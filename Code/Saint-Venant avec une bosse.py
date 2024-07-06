import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.close("all")
def divarr(N,D):
    return np.divide(N,D, out = np.zeros_like(N, dtype = float), where = D!=0)
    


'''Définition des fonctions b,U_initial et H_initial'''
def u_init(X):
    return 0*np.exp(-3*(X+L/4)**2)
def zeta_init(X):
    return np.exp(-20*(X+L/4)**2)
def b_func(X):
    return np.exp(-10*(X)**2)
def b_func0(X):
    return (np.exp(-10*(X+L/2)**2) + np.exp(-10*(X-L/2)**2))


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
"""Boucle d'éxecution"""
L = 100
a = -L/2; b = L/2
N = 4500
dt = 0.005
T = 36
ipf = 20 #ititérations par frame
bumpsgap = L/30
bumpmiddle = -L/4
def b_func(X, a= 0.6, w = L/90):
    return a * np.exp(- w **(-2) * (X- bumpmiddle)**2)

"""Construction d'une solution initiale unidirectionnelle"""
def zeta_init(X):
    return np.exp(-20*(X)**2)*2
wbhllsw0 = WBHLLSW(a, b, N, beta=1, eps=1, surface0=zeta_init)
t = 0
while t<10:
    wbhllsw0.tick(dt)
    t+=dt
S0 = wbhllsw0.getSurface() 
U0 = wbhllsw0.getSpeed()

def sigmoid(X):
    return 1/(1+np.exp(np.minimum(X, 80)))
def smoothcut(V,X,x_min, x_max, strengh = 0.7):
    return V * (1 - sigmoid((X - x_min)*strengh)) * sigmoid((X - x_max)*strengh)
S0 = np.roll(smoothcut(S0, wbhllsw0.getX(),0, L/4), N//2)
U0 = np.roll(smoothcut(U0, wbhllsw0.getX(),0, L/4), N//2)
wbhllsw = WBHLLSW(a, b, N, beta=1, eps=1, surface0=zeta_init, bottom=b_func)

wbhllsw.setSpeed(U0)
wbhllsw.setSurface(S0)

wbhllsw2 = WBHLLSW(a, b, N, beta=1, eps=1, surface0=zeta_init)
wbhllsw2.setSpeed(U0)
wbhllsw2.setSurface(S0)

t = 0
i = 0
snapshots = [0,5,10,15,20,25,35]
snapstep = 0
spg = []
norm = []
norm2 = []
dispcolor = True
while t<T:
    
    if (i%ipf==0):
        plt.figure(0,figsize=(10,3))
        plt.clf()
        plt.title("t = "+str(np.round(t,4)))
        plt.ylim(-1.1, 0.5)
        if dispcolor:
            plt.fill_between(wbhllsw.getX() , np.roll(wbhllsw.getSurface(), N//4), np.roll(wbhllsw.getFloor(), N//4))
            plt.fill_between(wbhllsw.getX() , np.roll(wbhllsw.getFloor(), N//4), -1, color = (0.4,0.2,0.2))
        plt.plot(wbhllsw.getX(), np.roll(wbhllsw.getSurface(), N//4), "b")
        plt.plot(wbhllsw2.getX(), np.roll(wbhllsw2.getSurface(), N//4), "k--", label = "fond plat")
        plt.plot(wbhllsw.getX(), np.roll(wbhllsw.getFloor(), N//4), "k")
        plt.xlabel("x")
        plt.legend()
        #plt.plot(wbhllsw.getX(), wbhllsw.getSpeed(), "r--")
        plt.tight_layout()
        plt.show()
        
        if (snapstep<len(snapshots)) and (t >= snapshots[snapstep]):
            plt.savefig("big - SW fond gaussienT"+str(snapshots[snapstep])+".pdf")
            plt.ylim(-0.1,0.25)
            plt.savefig("big zoomed - SW fond gaussienT"+str(snapshots[snapstep])+".pdf")
            print("printed")
            snapstep +=1
        plt.pause(0.01)
        
        spg.append(np.abs(fft(wbhllsw.getSurface())[:N//2]))
        norm.append(np.linalg.norm(wbhllsw.getSurface()))
        norm2.append(np.linalg.norm(wbhllsw2.getSurface()))
        
    t += dt
    i += 1
    wbhllsw.tick(dt)
    wbhllsw2.tick(dt)
        
xi = np.linspace(0,1, N//2)*N/L/np.pi 
plt.figure(2, figsize=(10,3))
plt.clf()
plt.imshow( np.array(spg).T[::-1], extent=[0,T,0,xi[-1]], aspect='auto', vmin = 0, interpolation= "gaussian")
plt.xlabel("temps")
plt.ylabel("fréquence")
plt.colorbar()
plt.show()


plt.figure(3)
plt.clf()
plt.title("Norme L2")
plt.plot(np.linspace(0,T,len(norm)), norm2, "k--", label = "sans bosse")
plt.plot(np.linspace(0,T,len(norm)), norm, label = "avec bosse")
plt.xlabel("t")
plt.legend()
plt.show()

