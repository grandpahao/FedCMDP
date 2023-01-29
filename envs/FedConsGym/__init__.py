from .base import *

def make_env(args):
    ename = args.env_name.split('-')[0]
    if ename == "CartPole":
        return CartPole(args)
    elif ename == "Acrobot":
        return Acrobot(args)
    elif ename == "InvertedPendulum":
        return InvertedPendulum(args)
    elif ename == "InvertedDoublePendulum":
        return InvertedDoublePendulum(args)
    else:
        raise NotImplementedError("{} is not implemented in FedConsGym!".format(ename))