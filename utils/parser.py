import argparse

class Parser:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # settings for RandomMDP
        parser.add_argument("--nS", default=3, type=int)
        parser.add_argument("--nA", default=5, type=int)
        parser.add_argument("--ncost", default=4, type=int)
        parser.add_argument("--T", default=10000000, type=int)
        parser.add_argument("--E", default=1, type=int)
        parser.add_argument("--npg", default=1, type=int)
        parser.add_argument("--trunc", default=1, type=int)
        parser.add_argument("--lr_p", default=1e-3, type=float)
        parser.add_argument("--lr_d", default=1e-3, type=float)
        parser.add_argument("--seed", default=233, type=int)
        parser.add_argument('--sample', default=0, type=int)
        parser.add_argument('--sample_K', default=10, type=int)
        parser.add_argument('--hard', default=0.7, type=float)

        # additional settings for WindyCliff
        parser.add_argument("--M", default=4, type=int)
        parser.add_argument("--N", default=10, type=int)
        parser.add_argument("--wp",default=0.4, type=float)

        # environment settings for DeepExperiment
        parser.add_argument("--env_name", default="CartPole-v0", type=str, choices=["CartPole-v0", "Acrobot-v1", "InvertedPendulum-v2"])
        parser.add_argument("--config_name", default="Pos2", type=str)
        parser.add_argument("--gamma", default=0.99, type=float)
        parser.add_argument("--gamma_C", default=1.0, type=float)

        # agent settings for PolicyBasedAgents
        parser.add_argument("--agent", default="ppo", type=str, choices=["ppo", "trpo", "ddpg"])
        parser.add_argument("--traj_len", default=10000, type=int)
        parser.add_argument("--traj_num", default=9, type=int)
        parser.add_argument("--lr_actor", default=1e-4, type=float)
        parser.add_argument("--lr_critic", default=1e-4, type=float)
        parser.add_argument("--lamb0", default=0.0, type=float)
        parser.add_argument("--dual_bound", default=1.0, type=float)
        parser.add_argument("--lr_dual", default=1e-3, type=float)
        parser.add_argument("--polyak", default=0.995, type=float)
        parser.add_argument("--action_std_init", default=0.1, type=float)
        parser.add_argument("--standard_reward", default=1, type=int)
        parser.add_argument("--standard_cost", default=1, type=int)
        parser.add_argument("--standard_obs", default=0, type=int)

        # additional settings for PPO
        parser.add_argument("--ppo_clip", default=0.2, type=float)
        parser.add_argument("--po_K", default=20, type=int)
        parser.add_argument("--ThresKL", default=0.1, type=float)
        parser.add_argument("--early_stop_w_KL", default=1, type=int)
        
        # additional settings for TRPO
        parser.add_argument("--max_kl", default=1e-2, type=float)
        parser.add_argument("--damping", default=1e-1, type=float)
        parser.add_argument("--trpo_multiplier", default=1, type=int)
        
        # settings for logging files
        parser.add_argument("--log_ep", default=50, type=int)
        parser.add_argument("--exp", default=1, type=int)
        parser.add_argument("--single_agent_mode", default=0, type=int)
        parser.add_argument("--obs_c", nargs='+', type=int)
        parser.add_argument("--fed_reweight", default="uniform", type=str, choices=["softmax","uniform"])
        
        self.args = parser.parse_args()
    
    def parse_args(self):
        return self.args
