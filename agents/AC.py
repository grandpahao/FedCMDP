import os 
import sys 
sys.path.append("..")

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from utils.tools import polyak_target_update
from utils.PIDcontroller import LagrangianPIDController
from nets.fc import FC

class actor_critic_w_constraints:
    def __init__(self, nS, nA, device, ncost, net_structure=[64, 64], action_continuous=False, action_var=0.1):
        self.nS, self.nA, self.nC = nS, nA, ncost
        self.continuous = action_continuous
        self.device = device
        if self.continuous:
            self.set_action_var(action_var)
            net_structure = [256, 256]
            self.actor = FC(self.nS, self.nA, net_structure, device, activation=nn.Tanh())
        else:
            self.actor = FC(self.nS, self.nA, net_structure, device, activation = nn.Softmax(dim=-1))

        self.critic_r = FC(self.nS, 1, net_structure, device)
        self.critic_Cs = [FC(self.nS, 1, net_structure, device) for _ in range(self.nC)]
        

    def set_action_var(self, action_var):
        if self.continuous:
            self.action_var = torch.full((self.nA,), action_var*action_var).to(self.device)
        else:
            raise NotImplementedError

    def decay_action_var(self, decay_rate = 0.9999):
        self.action_var *= decay_rate

    def act(self, state, Eval=False):
        if self.continuous:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean,cov_mat)
            # self.decay_action_var()
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action = dist.sample()
        if self.continuous and Eval:
            action = action_mean
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        if self.continuous:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.nA==1:
                action = action.reshape(-1, self.nA)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values_Cs = [self.critic_Cs[i](state) for i in range(self.nC)]
        values_r = self.critic_r(state)       

        return action_logprob, values_r, values_Cs, dist_entropy, dist

    def load_model(self, ckpt_path):
        params = torch.load(ckpt_path)
        self.load_params(params)

    def load_params(self, params):
        self.actor.load_state_dict(params['actor'])
        self.critic_r.load_state_dict(params['critic_r'])
        for i in range(self.nC):
            self.critic_Cs[i].load_state_dict(params['critic_c'][i])
    
    def detach_dist(self, state):
        if self.continuous:
            action_mean = self.actor(state).detach()
            action_var = self.action_var.expand_as(action_mean).detach()
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state).detach()
            dist = Categorical(action_probs)
        return dist

    @property
    def parameters(self):
        params = {
            'actor': self.actor.state_dict(), 
            'critic_r': self.critic_r.state_dict(), 
            'critic_c': [self.critic_Cs[i].state_dict() for i in range(self.nC)],
        }
        return params