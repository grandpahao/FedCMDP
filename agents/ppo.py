import torch
from IPython import embed
from .base import PolicyBasedAgent

class PPO(PolicyBasedAgent):
    def __init__(self, env, args, obs_c, budget_c, device, **kwargs):
        super().__init__(env, args, obs_c, budget_c, device)
        
        # implementations of optimizers
        optim_list = [{'params':self.policy.actor.parameters(), 'lr':args.lr_actor}, 
                      {'params':self.policy.critic_r.parameters(), 'lr':args.lr_critic}]
        for i in range(self.nC):
            optim_list += [{'params':self.policy.critic_Cs[i].parameters(), 'lr':args.lr_critic}]
        self.optimizer_p = torch.optim.Adam(optim_list)

        self.early_stop_w_KL = args.early_stop_w_KL
        self.eps_clip = args.ppo_clip
        self.ThresKL = args.ThresKL
        self.po_K = args.po_K

    def update(self, trajs):
        S, A, oldLogp, stepR, stepCs, epR, avg_epCs = self._process_trajs(trajs)
        stepR /= self.nagent.float()

        # dual update
        self._update_lambda(avg_epCs)
        multipliers = [self.lamb[i].item() for i in range(self.nC)]

        # keep \pi_0 before update
        D0 = self.policy.detach_dist(S)
        for _ in range(self.po_K):
            Logp, VR, VCs, dist_entropy, D = self.policy.evaluate(S, A)
            VR = torch.squeeze(VR)
            for i in range(self.nC):
                VCs[i] = torch.squeeze(VCs[i])

            ratios = torch.exp(Logp-oldLogp.detach())
            clip_ratios = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)
            advR = stepR - VR.detach()
            advCs = [stepCs[i] - VCs[i].detach() for i in range(self.nC)]

            obj_pi = torch.min(ratios*advR, clip_ratios*advR)

            for i in range(self.nC):
                clip_advC = clip_ratios*advCs[i]
                # obj_pi -= multipliers[i]*torch.min(ratios*advCs[i], clip_advC)
                obj_pi -= multipliers[i]*ratios*advCs[i]
            obj_pi /= (1.0/self.nagent.float()+sum(multipliers))

            obj_pi -= 0.01*dist_entropy
            error_actor = -obj_pi
            error_critic = 0.5*self.loss_fn(VR, stepR)
            for i in range(self.nC):
                error_critic += (0.5*self.loss_fn(VCs[i], stepCs[i]))

            error_policy = error_critic+error_actor
            self.optimizer_p.zero_grad()
            error_policy.mean().backward()
            self.optimizer_p.step()
            self.target_update()

            KL_dist = torch.distributions.kl.kl_divergence(D, D0).mean().item()
            if KL_dist > self.ThresKL:
                break

        return True