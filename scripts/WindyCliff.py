import os
import sys
sys.path.append("..")

import argparse
import math
import pickle as pkl
import numpy as np
from tqdm import tqdm

from envs.WindyCliff import WindyCliff
from utils.parser import Parser
from utils.tools import *
from agents.softmax_agent import softmax_agent
from tensorboardX import SummaryWriter

parser = Parser()
args = parser.parse_args()

file_name = "../record/WindyCliff/M:{},N:{},ncost:{},E:{},wind_power:{},2".format(
    args.M, args.N, args.ncost, args.E, args.wp)
train_setting = "lr_w:{},lr_d:{},exp:{}".format(args.lr_p, args.lr_d, args.exp)
rcd_path = os.path.join(file_name, train_setting)

writer = SummaryWriter(rcd_path)

if __name__ == "__main__":
    env = WindyCliff(args.M, args.N, args.ncost, args.wp)
    # FedAs represents agents in the federated setting
    FedAs = [softmax_agent(env.nS, env.nA, env, args, [i]) for i in range(args.ncost)]
    # As represents agents without communication
    As = [softmax_agent(env.nS, env.nA, env, args, [i], False) for i in range(args.ncost)]
    # Ao represents the agent knowing all constraints
    Ao = softmax_agent(env.nS, env.nA, env, args)
    W = np.zeros((env.nS, env.nA))
    
    for t in tqdm(range(args.T)):
        for i in range(len(FedAs)):
            FedAs[i].assign_w(W)
        
        # The evaluation of central policy in FedNPG
        policy = softmax(W)
        R, Ratios, Cs = evaluate_policy(env, policy)
        writer.add_scalar("Reward/0", R, t)
        for i in range(len(Cs)):
            writer.add_scalar("Cost{}/0".format(i+1), Cs[i], t)
            writer.add_scalar("Ratio{}/0".format(i+1), Ratios[i], t)
        
        for j in range(env.ncost):
            policy = softmax(As[j].w)
            R, Ratios, Cs = evaluate_policy(env, policy)
            writer.add_scalar("Reward/{}".format(j+1), R, t)
            for i in range(len(Cs)):
                writer.add_scalar("Cost{}/{}".format(i+1, j+1), Cs[i], t)
                writer.add_scalar("Ratio{}/{}".format(i+1, j+1), Ratios[i], t)
        
        policy = softmax(Ao.w)
        R, Ratios, Cs = evaluate_policy(env, policy)
        writer.add_scalar("Reward/{}".format(env.ncost+1), R, t)
        for i in range(len(Cs)):
            writer.add_scalar("Cost{}/{}".format(i+1, env.ncost+1), Cs[i], t)
            writer.add_scalar("Ratio{}/{}".format(i+1, env.ncost+1), Ratios[i], t)

        for _ in range(args.E):
            cs, rs, vs, gs = [], [], [], []
            for i in range(env.ncost):
                FedAs[i].npg_update()
                As[i].npg_update()
            Ao.npg_update()
        
        P_W = arith_mean([softmax(a.w) for a in FedAs])
        W = inv_softmax(P_W)


