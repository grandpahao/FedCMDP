from .ppo import PPO
from .trpo import TRPO


def make_agent(env, args, obs_c, budget_c, device):
    if args.agent == "ppo":
        return PPO(env, args, obs_c, budget_c, device)
    elif args.agent == "trpo":
        return TRPO(env, args, obs_c, budget_c, device)
    elif args.agent == "ddpg":
        # return DDPG()
        pass
    else:
        raise NotImplementedError("{} is not implemented!".format(args.agent))