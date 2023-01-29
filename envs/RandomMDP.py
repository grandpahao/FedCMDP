import numpy as np 
from utils.tools import *

class RandomMDP:
    def __init__(self, nS, nA, seed, args, R=None, C=None, D=None):
        np.random.seed(seed)
        self.nS, self.nA, self.ncost, self.gamma = nS, nA, args.ncost, 0.9
        self.R, self.C, self.D = R, C, D
        self.init = np.ones((self.nS,))/self.nS
        
        P = np.random.uniform(size=(self.nS, self.nA, self.nS))
        mask = (np.random.uniform(size=(self.nS, self.nA, self.nS))<0.5)
        P = P*mask
        self.P = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                if(np.sum(P[s][a])==0):
                    self.P[s][a] = np.ones((self.nS,))/self.nS
                else:
                    self.P[s][a] = P[s][a]/np.sum(P[s][a])
        
        if self.R is None:
            self.R = np.random.uniform(size=(self.nS, self.nA))
        if self.C is None:
            costs = []
            for t in range(self.ncost):
                mask = (np.random.uniform(size=(self.nS, self.nA))<0.5)
                cost = np.random.uniform(size=(self.nS, self.nA))
                costs.append(cost*mask)
            self.C = np.array(costs)
            
            anchor_p = np.random.uniform(low=-1, high=1, size=(self.nS, self.nA))
            # anchor_p = np.zeros((self.nS, self.nA))
            anchor_p = np.exp(anchor_p)
            anchor_p = anchor_p/np.sum(anchor_p, 1, keepdims=True)
            
            self.D = []
            for i in range(self.C.shape[0]):
                V, _ = eval_QV(self.C[i], self.P, anchor_p, self.gamma)
                val = np.sum(V*self.init)
                self.D.append(val*args.hard)
                
            self.s = np.random.choice(a=self.nS, size=1, p=self.init)[0]
                
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
        nxt_s = np.random.choice(a=self.nS, size=1, p=trans)[0]
        r = self.R[self.s][a]
        Cs = [self.C[j][self.s][a] for j in range(self.ncost)]
        self.s = nxt_s
        return nxt_s, r, Cs