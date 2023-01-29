import sys
sys.path.append('..')

import numpy as np
from IPython import embed
from utils.tools import *

class softmax_agent:
    def __init__(self, nS, nA, e, args, obs_c=None, fed=True):
        self.nS, self.nA, self.env, self.obs_c = nS, nA, e, obs_c
        self.args = args
        if self.obs_c is None:
            self.obs_c = [_ for _ in range(self.env.ncost)]
        self.w, self.d = np.zeros((nS, nA)), np.zeros((len(self.obs_c),))
        self.lr_w, self.lr_d = self.args.lr_p, self.args.lr_d
        self.npg_style = True
        self.fed = fed 
        
    def pg_update(self):
        pi = softmax(self.w)
        d_init = eval_d(self.env.P, pi, self.env.gamma, self.env.init)
        
        V_R, Q_R = eval_QV(self.env.R, self.env.P, pi, self.env.gamma)
        A_R = Q_R-V_R[:, np.newaxis]
        V_Cs, Q_Cs = [], []
        for i in self.obs_c:
            V, Q = eval_QV(self.env.C[i], self.env.P, pi, self.env.gamma)
            V_Cs.append(V)
            Q_Cs.append(Q)
        R_grad = d_init[:, np.newaxis]*pi*A_R/(1-self.env.gamma)
        
        Q_C = np.zeros_like(R_grad)
        V_C = np.zeros_like(V_R)
        for i in range(len(self.obs_c)):
            Q_C += (Q_Cs[i]*self.d[i])
            V_C += (V_Cs[i]*self.d[i])
        A_C = Q_C-V_C[:, np.newaxis]
        C_grad = d_init[:, np.newaxis]*pi*A_C/(1-self.env.gamma)
        
        self.w = self.w + self.lr_w*(R_grad/self.args.ncost+C_grad)
        for i in range(len(self.obs_c)):
            violate = np.sum(V_Cs[i]*self.env.init)-self.env.D[self.obs_c[i]]
            if self.args.trunc == 1:
                self.d[i] = min(max(0, self.d[i]-self.lr_d*violate),10)
            else:
                self.d[i] = max(0, self.d[i]-self.lr_d*violate)
        return np.square(R_grad/self.args.ncost+C_grad).sum()**0.5

    def npg_update(self):
        pi = softmax(self.w)
        if self.args.sample:
            V_R, Q_R, V_Cs, Q_Cs = sample_QV_RC(self.env, pi, self.args.sample_K, self.obs_c)
        else:
            V_R, Q_R = eval_QV(self.env.R, self.env.P, pi, self.env.gamma)
            
            V_Cs, Q_Cs = [], []
            for i in self.obs_c:
                V, Q = eval_QV(self.env.C[i], self.env.P, pi, self.env.gamma)
                V_Cs.append(V)
                Q_Cs.append(Q)
        
        Q_C, V_C = np.zeros_like(Q_R), np.zeros_like(V_R)
        for i in range(len(self.obs_c)):
            Q_C += (Q_Cs[i]*self.d[i])
            V_C += (V_Cs[i]*self.d[i])
            
        A_R = Q_R-V_R[:, np.newaxis]
        A_C = Q_C-V_C[:, np.newaxis]

        R_grad = A_R/(1-self.env.gamma)
        C_grad = A_C/(1-self.env.gamma)

        regular = self.args.ncost if self.fed else 1.0
        self.w += self.lr_w*(R_grad/regular-C_grad)

        for i in range(len(self.obs_c)):
            violate = np.sum(V_Cs[i]*self.env.init)-self.env.D[self.obs_c[i]]
            if self.args.trunc==1:
                self.d[i] = min(max(0, self.d[i]+self.lr_d*violate),10)
            else:
                self.d[i] = max(0,self.d[i]+self.lr_d*violate)
        
    def assign_w(self, w):
        self.w = w
        
    def step(self, s):
        w = self.policy[s]
        p = softmax(w)
        a = np.random.choice(a=self.nA, size=1, p=p)[0]
        return a