import os 
import sys
sys.path.append("..")

import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from IPython import embed

from utils.tools import polyak_target_update
from utils.PIDcontroller import LagrangianPIDController
from nets.fc import FC

from agents.AC import actor_critic_w_constraints

class PolicyBasedAgent:
    def __init__(self, env, args, obs_c, budget_c, device, **kwargs):
        self.gamma = args.gamma
        self.gamma_C = args.gamma_C
        self.continuous = env.continuous
        self.device = device
        self.polyak = args.polyak
        self.obs_c = obs_c
        if self.continuous:
            self.action_low, self.action_high = env.action_low, env.action_high
            self.action_mean, self.action_range = env.action_mean, env.action_range

        self.nC = torch.tensor(len(obs_c), device=self.device, requires_grad=False)
        
        FedN = len(env.constraints)
        if args.single_agent_mode:
            FedN = 1
        self.nagent = torch.tensor(FedN, device=self.device, requires_grad=False)
        self.thres_c = torch.tensor(budget_c, requires_grad=False, device=self.device)

        # implementations of policy networks
        self.policy = actor_critic_w_constraints(env.nS, env.nA, device, self.nC, action_continuous=env.continuous, action_var = args.action_std_init)
        self.target_policy = actor_critic_w_constraints(env.nS, env.nA, device, self.nC, action_continuous=env.continuous)
        self.target_policy.load_params(self.policy.parameters)

        # implementations of loss functions for critics
        self.loss_fn = nn.MSELoss()
        
        # details of dual variables
        self.lamb = torch.zeros(self.nC, requires_grad=True, device=self.device)
        self.lamb_max = args.dual_bound
        self.lr_dual = args.lr_dual

        # control settings for standardization
        self.standard_cost = args.standard_cost
        self.standard_obs = args.standard_obs
        self.standard_reward = args.standard_reward

    def target_update(self):
        self.target_update_critic()
        self.target_update_actor()

    def target_update_critic(self):
        polyak_target_update(self.policy.critic_r, self.target_policy.critic_r, self.polyak)
        for i in range(self.nC):
            polyak_target_update(self.policy.critic_Cs[i], self.target_policy.critic_Cs[i], self.polyak)

    def target_update_actor(self):
        polyak_target_update(self.policy.actor, self.target_policy.actor, self.polyak)

    def select_action(self, state, Eval=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            a, a_logp = self.target_policy.act(state, Eval)
            a = a.detach().cpu().numpy().flatten()
        if not self.continuous:
            a = a.item()
        return a, a_logp

    def aggregate_policy(self, Ps, CriticRs, Ws):
        for p, p_local in zip(self.target_policy.actor.parameters(), Ps[0]):
            p.data.mul_(0.0)
            p.data.add_(p_local.data*Ws[0])

        for p, p_local in zip(self.target_policy.critic_r.parameters(), CriticRs[0]):
            p.data.mul_(0.0)
            p.data.add_(p_local.data*Ws[0])

        for i in range(1, self.nagent):
            for p, p_local in zip(self.target_policy.actor.parameters(), Ps[i]):
                p.data.add_(p_local.data*Ws[i])
            for p, p_local in zip(self.target_policy.critic_r.parameters(), CriticRs[i]):
                p.data.add_(p_local.data*Ws[i])

    def assign_policy(self, central_policy, central_criticR):
        for p, p_aggr in zip(self.policy.actor.parameters(), central_policy):
            p.data.mul_(0.0)
            p.data.add_(p_aggr)
        for p, p_aggr in zip(self.policy.critic_r.parameters(), central_criticR):
            p.data.mul_(0.0)
            p.data.add_(p_aggr)

    def _process_trajs(self, traj):
        epN, epR, epCs = -1, [], [[] for _ in range(self.nC)]
        cumuR, cumuCs = 0, [0 for _ in range(self.nC)]
        stepR, stepCs = [], [[] for _ in range(self.nC)]

        for r, C, d in zip(reversed(traj.rewards), reversed(traj.costs), reversed(traj.dones)):
            if d:
                epN += 1
                if epN > 0:
                    epR.append(cumuR)
                    for i in range(self.nC):
                        epCs[i].append(cumuCs[i])
                cumuR, cumuCs = 0, [0 for _ in range(self.nC)]
            cumuR = r + self.gamma*cumuR
            stepR.insert(0, cumuR)
            for i in range(self.nC):
                cumuCs[i] = C[i] + self.gamma_C*cumuCs[i]
                stepCs[i].insert(0, cumuCs[i])

        S = torch.squeeze(torch.tensor(traj.states, dtype=torch.float32)).detach().to(self.device)
        A = torch.squeeze(torch.tensor(traj.actions, dtype=torch.float32)).detach().to(self.device)
        oldLogp = torch.squeeze(torch.tensor(traj.logps, dtype=torch.float32)).detach().to(self.device)

        stepR = torch.tensor(stepR, dtype=torch.float32).to(self.device)
        stepCs = torch.tensor(stepCs, dtype=torch.float32).to(self.device)
        
        epCs = torch.tensor(epCs, dtype=torch.float32).to(self.device)
        avg_epCs = epCs.mean(dim=-1).squeeze()

        if self.standard_reward:
            stepR = (stepR-stepR.mean())/(stepR.std()+1e-7)
        if self.standard_cost:
            stepCs = (stepCs-stepCs.mean(dim=-1).unsqueeze(dim=-1))/(stepCs.std(dim=-1).unsqueeze(dim=-1)+1e-7)
        if self.standard_obs:
            S = (S-S.mean())/(S.std()+1e-7)
        return S, A, oldLogp, stepR, stepCs, epR, avg_epCs

    def _update_lambda(self, avg_epCs):
        with torch.no_grad():
            violate = avg_epCs-self.thres_c
            self.lamb += self.lr_dual*violate
            self.lamb = torch.clamp(self.lamb, 0, self.lamb_max)

    def save_model(self, model_dir, n, idx):
        ckpt_path = os.path.join(model_dir, "agent:{},step:{}.ckpt".format(idx, n))
        params = {
            'ac': self.policy.parameters,
            'lambda': self.lamb
        }
        torch.save(params, ckpt_path)
    
    def load_model(self, ckpt_path):
        params = torch.load(ckpt_path)
        self.policy.load_params(params['ac'])
        self.target_policy.load_params(params['ac'])
        self.lamb = params['lambda']

    def update(self, trajs):
        pass 

