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


def defaultdampingrate(t):
    return 1e-2 + 0*t
class DampedKdV:
    def __init__(self, a, b, N, u0, dt = 1e-2, Tmax = 1, dampingrate = defaultdampingrate, q = 2, bandcovapprox = 0, timestep = 1):
        self.N = N
        self.X, self.dx = np.linspace(a, b, N, endpoint=False, retstep=True)
        self.U = u0(self.X)
        
        self.t = 0
        self.dt = dt
        self.Tmax = Tmax
        N = int(1/timestep)
        
        self.I = sps.eye(self.N, format="csr", dtype=float) 
        self.B = B_mat(self.N, self.dx)
        self.D = D_mat(self.N, self.dx)
        #self.L = L_mat(self.N, self.dx)
        self.L = L_mat(self.N, self.dx) + 0*0.1*sps.eye(self.N, format = "csr", dtype = float)
        
        self.dr = dampingrate
        self.covstep = 1
        self.alpha = 0
        self.ncovapprox = q
        self.ai = np.array([np.roll(self.U,i*self.covstep).dot(self.U) for i in range(self.ncovapprox)]).astype(float)
        self.J = [sproll(self.I,(i+1)*self.covstep) + sproll(self.I,-(i+1)*self.covstep)  for i in range(self.ncovapprox-1)]
        
        self.usqrt = np.array([np.exp(2j*np.pi*i/self.N) for i in np.linspace(0, self.N-1, int(self.N/25), True).astype(int)])
        
        self.TAM1_1 = 0
        self.TAM1_2 = 0
        self.TAM3 = 0
        self.lastcondprocess = 1
        self.n = 0
        
        self.theta = 0
        
        self.dk = int(timestep/self.dt)
        self.k = 0
        self.res = 1e-11
        self.itermax = 30
    
    def tick(self):
        Uk = np.copy(self.U)
        res = self.res + 1
        k = 0
        self.k+=1
        while res > self.res:
            k+= 1
            # Formulation Sanz Serna + itération de picard
            Ukn = sps.linalg.spsolve( self.I + self.dt * 0.5 * self.B + self.dt * self.dr(self.t + self.dt) * self.theta * self.L,
                                        self.U - self.dt * (0.5 * self.B +(1-self.theta) * self.dr(self.t)*self.L).dot(self.U) - 0.5 * self.dt * self.D.dot( approxSquare(np.roll((Uk+self.U)/2,1),np.roll((Uk+self.U)/2,-1))))  
            #Calcul du résidu
            res = np.linalg.norm(Uk - Ukn) * np.sqrt(self.dx)
            Uk = np.copy(Ukn)
            #Critère d'arrêt si non convergence
            if k > self.itermax:
                print("itermax reached")
                break
        #Calcul de l'action d'amortissement
        if self.k%self.dk == 0:
            AU = -((Uk - self.U)/dt + 0.5 * self.B.dot(Uk + self.U) + 0.5 * self.D.dot(approxSquare(np.roll((Uk+self.U)/2,1),np.roll((Uk+self.U)/2,-1))))
            self.storedata(self.U, AU)
        
        #actualisation
        self.U = np.copy(Uk)
        self.t += self.dt
        
    def storedata(self, Y, AY):
        self.TAM1_1 += AY.dot(Y)
        self.TAM1_2 += Y.dot(Y)
        
        self.ai = np.array([np.roll(Y,i*self.covstep).dot(Y) for i in range(self.ncovapprox)], dtype = float)
        valP = self.ai[0] * self.usqrt ** 0
        M = self.ai[0]*self.I/self.N
        for i in range(self.ncovapprox - 1):
            M += self.ai[i+1]/self.N*self.J[i]
            valP += self.ai[i+1] * (self.usqrt ** i + self.usqrt ** (-i) )
        self.TAM3 += sps.linalg.spsolve(M, AY).dot(Y)
        self.lastcondprocess = np.max(np.abs(valP))/np.min(np.abs(valP))
        self.n += 1
    def u(self):
        return self.U
    def x(self):
        return self.X
    def time(self):
        return self.t
    def isDone(self):
        return self.t>self.Tmax
    def getsAYY(self):
        return self.TAM1_1
    def getsYY(self):
        return self.TAM1_2
    def gettrace(self):
        return sps.csr_matrix.trace(self.L)
    def getTAM1(self):
        if self.TAM1_1 > 0:
            return self.N*self.TAM1_1/self.TAM1_2
    def getTAM3(self):
        return self.TAM3/self.n
    def getsamplesize(self):
        return self.n
    def dampingrate(self, t):
        return self.dr(t)
"""Données relative à la discrétisation du Tore"""
length = 10
a = -length * np.pi;
b = length * np.pi
N = 500
X, h = np.linspace(a, b, N, endpoint=False, retstep=True)

T = 2;
dt = 1e-3
timestep = 20*dt

t = 0

q = 3
"""Construction des données initiales et des paramètres d'évolution en temps"""
def uinit(X):
    return 1-2*(np.random.random(X.shape)<0.5).astype(float) 

def sumqterms(V, q):
    n = len(V)
    W = np.zeros(n-q)
    for i in range(q+1):
        W += V[i:n-q+i]
    return W/(q+1)

q1 = 2
q2 = 4
proco = DampedKdV(a, b, N, dt = dt, Tmax = T, u0 = uinit, q = q1, timestep = timestep)
proco2 = DampedKdV(a, b, N, dt = dt, Tmax = T, u0 = uinit, q = q2, timestep = timestep)
TAM1list = []
TAM3q1list = []
TAM3q2list = []
n = 0
while not proco.isDone():
    proco.tick()
    proco2.tick()
    if proco.getsamplesize()>n:
        n+=1
        TAM1list.append(proco.getTAM1())
        TAM3q1list.append(proco.getTAM3())
        TAM3q2list.append(proco2.getTAM3())
    
narr = np.arange(proco.getsamplesize())
plt.figure(0)
plt.clf()

plt.plot(narr, proco.dampingrate(narr*timestep)*proco.gettrace(), "k--", label = "exact value")
plt.plot(narr, TAM1list, "b+-", label = "TAM1")
plt.plot(narr, TAM3q1list, "r+-", label = "TAM3 q = "+str(q1))
plt.plot(narr, TAM3q2list, "g+-", label = "TAM3 q = "+str(q2))
plt.xlabel("N")
plt.legend()
plt.show()    
    
    



