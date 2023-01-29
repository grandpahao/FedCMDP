import numpy as np
from IPython import embed
from utils import *

class WindyCliff:
    def __init__(self, M, N, ncost, wind_power=0.3):
        self.M, self.N, self.gamma = M, N, 0.9
        self.ncost = ncost
        self.nS, self.nA = M*N, 4
        self.init = np.zeros((self.nS,))
        self.init[(M-1)*N] = 1
        self.wind_power = wind_power
        
        self.R = np.ones((self.nS, self.nA))*(-0.1)
        for a in range(self.nA):
            self.R[-1][a] = 20
        
        self.C = []
        base = 2
        segL = (N-4)//self.ncost
        for i in range(self.ncost):
            C = np.zeros((self.nS, self.nA))
            for _ in range(segL):
                for a in range(self.nA):
                    C[(M-1)*N+base][a] = 10
                base += 1
            self.C.append(C)
        self.C = np.array(self.C)
        self.D = np.ones((self.ncost,))*2.0

        self.P = np.zeros((self.nS, self.nA, self.nS))
        Dx, Dy = [0, 0, 1, -1], [1, -1, 0, 0]
        for i in range(self.M):
            for j in range(self.N):
                s = i*self.N+j
                downi, downj = i+1, j
                if downi<0 or downi>=M or downj<0 or downj>=N:
                    downi, downj = i, j
                downs = downi*self.N+downj
                for a in [0, 1, 3]:
                    nxti, nxtj = i+Dx[a], j+Dy[a]
                    if nxti<0 or nxti>=M or nxtj<0 or nxtj>=N:
                        nxti, nxtj = i, j
                    nxts = nxti*self.N+nxtj
                    self.P[s][a][nxts] += (1-self.wind_power)
                    self.P[s][a][downs] += self.wind_power
                self.P[s][2][downs] = 1
        self.s = (M-1)*N
    
    def reset(self, S0=None):
        if S0==None:
            s0 = np.random.choice(a=self.nS, size=1, p=self.init)[0]
        else:
            s0 = S0
        self.s = s0
        return s0

    def traj_gen(self, S0, Pi, K, obs_c, A0=None):
        S, R, Cs = [], [], []
        s = self.reset(S0)
        if A0 is not None:
            ns, r, cs = self.step(A0)
            R.append(r)
            Cs.append(cs)
            S.append(s)
            s = ns
            K -= 1
        for _ in range(K):
            S.append(s)
            a = np.random.choice(a=self.nA, size=1, p=Pi[s])[0]
            ns, r, cs = self.step(a)
            R.append(r)
            Cs.append(cs)
            s = ns
        R_sum = sum(R)
        obs_C = [sum([cs[i] for cs in Cs]) for i in obs_c]
        return S, R_sum, obs_C

    def step(self, a):
        trans = self.P[self.s][a]
        # embed()
        nxt_s = np.random.choice(a=self.nS, size=1, p=trans)[0]
        r = self.R[self.s][a]
        Cs = [self.C[j][self.s][a] for j in range(self.ncost)]
        self.s = nxt_s
        return nxt_s, r, Cs
