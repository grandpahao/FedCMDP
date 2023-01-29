import torch
import numpy as np
from IPython import embed

def softmax(w):
    w = np.exp(w)
    w /= np.sum(w,axis=1,keepdims=True)
    return w

def inv_softmax(p):
    w = np.log(p+1e-8)
    w += (np.sum(w,axis=1,keepdims=True)+1)
    return w

def bi_mean(pi, vio):
    vio_p, p = np.zeros_like(pi[0]), np.zeros_like(pi[0])
    for i in range(len(pi)):
        if i in vio:
            vio_p += pi[i]
        else:
            p += pi[i]
    if len(vio)==0:
        return p/len(pi)
    else:
        return 0.5*(vio_p/len(vio))+0.5*(p/(len(pi)-len(vio)))

def arith_mean(Pi):
    pi = np.zeros_like(Pi[0])
    for p in Pi:
        pi += p 
    return pi/len(Pi)

def geo_mean(Pi):
    pi = np.zeros_like(Pi[0])
    for p in Pi:
        pi += np.log(p+1e-8) 
    pi = np.exp(pi/len(Pi))
    return pi

def sample_QV_RC(env, pi, K, obs_c):
    VR, QR = np.zeros((env.nS,)), np.zeros((env.nS, env.nA))
    VC, QC = [np.zeros((env.nS,)) for _ in range(len(obs_c))], [np.zeros((env.nS, env.nA)) for _ in range(len(obs_c))]

    for s in range(env.nS):
        for k in range(K):
            l = np.random.geometric(1-env.gamma, 1)[0]
            _, R, Cs = env.traj_gen(s, pi, l, obs_c)
            VR[s] += R/K
            for i in range(len(obs_c)):
                VC[i][s] += Cs[i]/K
        for a in range(env.nA):
            for k in range(K):
                l = np.random.geometric(1-env.gamma, 1)[0]
                _, R, Cs = env.traj_gen(s, pi, l, obs_c, a)
                QR[s][a] += R/K
                for i in range(len(obs_c)):
                    QC[i][s][a] += Cs[i]/K 
    return VR, QR, VC, QC

def eval_QV(R, P, pi, gamma):
    nS, nA = R.shape[0], R.shape[1]
    R_pi = np.array([np.sum(R[s]*pi[s]) for s in range(nS)])
    P_pi = np.zeros((nS, nS))
    
    for s in range(nS):
        for s_ in range(nS):
            for a in range(nA):
                P_pi[s][s_] = P_pi[s][s_] + pi[s][a]*P[s][a][s_]
                
    V = np.linalg.inv(np.eye(nS)-gamma*P_pi+1e-5*np.ones((nS,nS)))@R_pi
    Q = np.zeros_like(R)
    for s in range(nS):
        for a in range(nA):
            Q[s][a] = R[s][a] + gamma*np.sum(P[s][a]*V)
    
    return V, Q

def evaluate_policy(env, policy):
    R, Ratios, Cs = 0.0, [], []

    VR, _ = eval_QV(env.R, env.P, policy, env.gamma)
    R = np.sum(env.init*VR)
    for i in range(env.ncost):
        VC, _ = eval_QV(env.C[i], env.P, policy, env.gamma)
        C = np.sum(env.init*VC)
        Cs.append(C)
        if env.D[i] == 0.0:
            Ratios.append(C)
        else:
            Ratios.append(C/env.D[i])

    return R, Ratios, Cs

def eval_d(P, pi, gamma, init):
    nS, nA = P.shape[0], P.shape[1]
    P_pi = np.zeros((nS, nS))
    for s in range(nS):
        for s_ in range(nS):
            for a in range(nA):
                P_pi[s_][s] = P_pi[s_][s] + pi[s][a]*P[s][a][s_]
    d_pi = (1-gamma)*np.linalg.inv(np.eye(nS)-gamma*P_pi+1e-5*np.ones((nS, nS)))@init
    return d_pi

def polyak_target_update(net, target_net, polyak):
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), target_net.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1-polyak)*p.data)

class Scale_Continuous_Action:
    def __init__(self, a_mean, a_range, a_low, a_high):
        self.a_mean, self.a_range = a_mean, a_range
        self.a_low, self.a_high = a_low, a_high
    
    def __call__(self, a):
        A = a*self.a_range*0.5 + self.a_mean
        A = np.clip(A, self.a_low, self.a_high)
        return A