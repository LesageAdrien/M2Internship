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

class Spectrogramm:
    def __init__(self):
        self.spectrogramm = []
        self.n = 0
    def appendice(self, vector):
        self.n += 1
        if self.n%10 == 0:
            print(self.n)
            print(len(vector))
        self.spectrogramm.append(np.abs(fft(vector)[:len(vector)//2]).astype(float))
    def disp(self):
        
        plt.figure(10)
        plt.imshow(np.array(self.spectrogramm)/self.spectrogramm[0])
        plt.show()
        plt.pause(0.01)

def defaultdampingrate(t):
    return 0.01 * (2.2 + (np.sin(t/0.1 * 2*np.pi) + 0.4*np.sin(t/0.2 * 2*np.pi)))
class DampedKdV:
    def __init__(self, a, b, N, u0, dt = 1e-2, Tmax = 1, dampingrate = defaultdampingrate, p = 10, bandcovapprox = 0, timestep = 1):
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
        self.ncovapprox = 5
        self.ai = np.array([np.roll(self.U,i*self.covstep).dot(self.U) for i in range(self.ncovapprox)]).astype(float)
        self.J = [sproll(self.I,(i+1)*self.covstep) + sproll(self.I,-(i+1)*self.covstep)  for i in range(self.ncovapprox-1)]
        self.Tcovlist = [0 for i in range(self.ncovapprox)]
        
        self.p = p
        self.r1 = (0.9)**(0.01*timestep)
        self.r2 = (2)**(timestep)   
        self.circle = np.array([np.exp(2*i/self.p*np.pi*1j) for i in range(self.p)])
        self.circlemap = np.meshgrid(self.circle, self.U)[0]
        self.TsAYY = 0
        self.TsYY = 0
        self.sAYY = []
        self.sYY = []
        
        
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
        #print(AY.dot(Y)*self.r1**(-self.n))
        #print(Y.dot(Y)*self.r2**((-self.n)))
        self.ai = self.alpha*self.ai + (1-self.alpha)*np.array([np.roll(Y,i*self.covstep).dot(Y) for i in range(self.ncovapprox)], dtype = float)
        
        M = self.ai[0]*self.I/self.N
        for i in range(self.ncovapprox - 1):
            M += self.ai[i+1]/self.N*self.J[i]
        for i in range(self.ncovapprox):
            self.Tcovlist[i] = self.Tcovlist[i] + self.ai[i]*(self.circle*self.r2)**(-self.n)
        ayy = sps.linalg.spsolve(M, AY).dot(Y)
        self.sAYY.append(ayy)
        self.sYY.append(Y.dot(Y))
        self.TsAYY += ayy*(self.circle*self.r1)**(-self.n)
        self.TsYY += Y.dot(Y)*(self.circle*self.r2)**(-self.n)
        
        self.n += 1
    def u(self):
        return self.U
    def x(self):
        return self.X
    def time(self):
        return self.t
    def isDone(self):
        return self.t>self.Tmax
    def getcircle(self):
        return self.circle
    def getTsAYY(self):
        return self.TsAYY
    def getTsYY(self):
        return self.TsYY
    def getcurrentiter(self):
        return self.n
    def getr1(self):
        return self.r1
    def getr2(self):
        return self.r2
    def getsAYY(self):
        return self.sAYY
    def getsYY(self):
        return self.sYY
    def gettrace(self):
        return sps.csr_matrix.trace(self.L)
    def getTcov(self):
        return self.Tcovlist
    def getU(self):
        return np.copy(self.U)
    
"""Données relative à la discrétisation du Tore"""
length = 10
a = -length * np.pi;
b = length * np.pi
N = 500
X, h = np.linspace(a, b, N, endpoint=False, retstep=True)

T = 0.5;
q = 14
dt = 1e-3
Ztransformtimestep = dt

t = 0

p = 2000


ntest = 1
"""Construction des données initiales et des paramètres d'évolution en temps"""
def soliton(c,xi):  
    return 3*c/(1 + np.sinh(0.5*np.sqrt(c)*xi)**2)
def uinit(X):
    return 1-2*(np.random.random(X.shape)<0.5).astype(float) 

def sumqterms(V, q):
    n = len(V)
    W = np.zeros(n-q)
    for i in range(q+1):
        W += V[i:n-q+i]
    return W/(q+1)


V1 = 0
V2 = 0
W1 = 0
W2 = 0
C = 0
for j in range(ntest):
    spctgrm = Spectrogramm()
    proco = DampedKdV(a, b, N, dt = dt, Tmax = T, u0 = uinit, p = p, timestep = Ztransformtimestep)
    while not proco.isDone():
        proco.tick()
        spctgrm.appendice(proco.getU())
    spctgrm.disp()
    circle1 = proco.getcircle()*proco.getr1()
    circle2 = proco.getcircle()*proco.getr2()
    tcovlist = proco.getTcov()
    nc = len(tcovlist)
    n = proco.getcurrentiter()
    Vl1 = np.zeros(n, dtype = np.complex128)
    Vl2 = np.zeros(n, dtype = np.complex128)
    Cl = np.zeros((n, nc), dtype = np.complex128)
    for i in range(n):
        Vl1[i] = np.mean(circle1**(i) * (proco.getTsAYY())*(1-circle1**-(q+1))/(1-circle1**-1))/(q+1)
        Vl2[i] = np.mean(circle2**(i) * (proco.getTsYY())*(1-circle2**-(q+1))/(1-circle2**-1))
        for l in range(nc):
            Cl[i,l] = np.mean(circle2**(i) * tcovlist[l])
    V1 = V1 + np.array(Vl1)[q:]/ntest
    V2 = V2 + np.array(Vl2)[q:]/ntest
    W1 = W1 + sumqterms(np.array(proco.getsAYY()), q)/ntest
    W2 = W2 + sumqterms(np.array(proco.getsYY()), q)/ntest
    C = C + Cl/N/proco.getr2()/ntest
    print("step = "+str(j)+" Done")



tarr = np.arange(n)
tarrq = sumqterms(tarr, q)
plt.figure(0)
plt.clf()
#plt.title("tr(A^N) et ses différentes approximations")
plt.plot(tarrq, defaultdampingrate(dt*tarrq) * proco.gettrace(),"k--", label = "exact value")
plt.plot(tarrq[::4], V1[::4], "g+", label = "Z-TD-TAM3 p = 7")
# plt.plot(tarrq[::4], W1[::4], 'r+', label = "Moyenne point par point")

plt.xlabel("N")
plt.legend()
plt.show()

plt.figure(1)
plt.clf()
#plt.title("Coefficients a_i de la matrice circulante Cov(V^k)")
for j in range(nc):
    plt.plot(tarr, C[:,j],color = [0,j/nc,1-j/nc],  label = "$a_{"+str(j)+"}^k$")
plt.xlabel("k")
plt.legend()
plt.show()

nlmbd = 30
circle = np.array([np.exp(2j*np.pi*i/nlmbd) for i in range(nlmbd)])
plt.figure(2)
plt.clf()

plt.title('Approximation de $F(u(.,t))(\\xi = \\delta \\times \\xi_{max})$')
for j in range(nlmbd):
    Ctemp = C[:,0]
    for k in range(C.shape[1]-1):
        Ctemp = Ctemp + C[:,k+1]*np.real(circle[j]**k)
    plt.plot(tarr[:], Ctemp[:]/Ctemp[0], color = [0,j/nlmbd,1-j/nlmbd],  label = "$\\delta = "+str(round(j/(nlmbd-1), 2))+"$")
plt.xlabel("t")
plt.legend()
plt.show()




