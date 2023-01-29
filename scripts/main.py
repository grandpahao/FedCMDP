import os
import sys
import time
import pickle as pkl
sys.path.append("..")

import math
import torch
import argparse
from tensorboardX import SummaryWriter

from utils.tools import *
from utils.Buffer import *
from utils.parser import Parser
from envs.FedConsGym import make_env

from agents import make_agent

parser = Parser()
args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading an environment copy for evaluation
Etest = make_env(args)
if Etest.continuous:
    ScaleA = Scale_Continuous_Action(Etest.action_mean, Etest.action_range, Etest.action_low, Etest.action_high)
ncost = len(Etest.constraints)
FedN = ncost
Gamma = [[i] for i in range(ncost)]

# setting of log files
env_log_dir = "../record/{}".format(Etest.ename)
config_log_dir = os.path.join(env_log_dir, args.config_name)

agent_setting = ","
if args.single_agent_mode:
    FedN = 1
    Gamma = [args.obs_c]
    agent_setting += "N:1"
    agent_setting += ",{}".format(args.obs_c)
else:
    agent_setting += "N:{}".format(ncost)
config_log_dir += agent_setting

if not os.path.exists(config_log_dir):
    os.makedirs(config_log_dir)

train_setting = ""
train_setting += "#{}".format(args.exp)
train_setting += "T:{},E:{},lr_actor:{},lr_critic:{},".format(args.T, args.E, args.lr_actor, args.lr_critic)
train_setting += "lr_dual:{},dual_bound:{},".format(args.lr_dual, args.dual_bound)
train_setting += "agent:{}".format(args.agent)
if args.agent == "ppo":
    train_setting += "po_K:{}".format(args.po_K)
elif args.agent == "trpo":
    train_setting += "trpo_m:{}".format(args.trpo_multiplier)
if args.single_agent_mode == 0:
    train_setting += ",fed_reweight:{}".format(args.fed_reweight)
train_setting += ",TS:{}".format(time.time())

rcd_path = os.path.join(config_log_dir, train_setting)
writer = SummaryWriter(rcd_path)

model_dir = os.path.join(rcd_path, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if __name__ == "__main__":
    Es = [make_env(args) for _ in range(FedN)]
    Bs = [Traj_Buffer() for _ in range(FedN)]
    As = [make_agent(Etest, args, Gamma[i], [Etest.config["budget"][j] for j in Gamma[i]], device) for i in range(FedN)]    
    central_A = make_agent(Etest, args, [], [], device)

    t, train_t, test_t = 0, 0, 0

    S = [Es[i].reset() for i in range(FedN)]
    Traj_num = [0 for i in range(FedN)]
    while t < args.T:
        # Action contains [(a, a_logp)_i]
        Actions = [As[i].select_action(S[i]) for i in range(FedN)]
        # Info contains [(ns, r, c, done, others)_i]

        if Etest.continuous:
            Infos = [Es[i].step(ScaleA(Actions[i][0])) for i in range(FedN)]
        else:
            Infos = [Es[i].step(Actions[i][0]) for i in range(FedN)]
        
        for i in range(FedN):
            # Buffer stores (s, a, logp, r, c, done)
            Bs[i].insert(S[i], Actions[i][0], Actions[i][1], Infos[i][1], [Infos[i][2][j] for j in As[i].obs_c], Infos[i][3])
            if Infos[i][3]:
                S[i] = Es[i].reset()
                Traj_num[i] += 1
            else:
                S[i] = Infos[i][0]
        t += 1

        all_local_success = True
        if t%args.traj_len == 0 and t > 0:
            if min(Traj_num)<args.traj_num:
                continue
            else:
                Traj_num = [0 for i in range(FedN)]

            for i in range(FedN):
                if not As[i].update(Bs[i]):
                    all_local_success = False
                Bs[i].clear()
            
            Ps = [As[i].policy.actor.parameters() for i in range(FedN)]
            CriticRs = [As[i].policy.critic_r.parameters() for i in range(FedN)]
            
            lamb_sum = np.array([torch.sum(As[i].lamb).item() for i in range(FedN)])
            if args.fed_reweight == "uniform":
                Ws = np.ones(FedN)/FedN
            elif args.fed_reweight == "softmax":
                Ws = np.exp(lamb_sum)/np.exp(lamb_sum).sum()
            
            central_A.aggregate_policy(Ps, CriticRs, Ws)
            # evaluate aggregate_policy
            eval_R, eval_Cs = 0.0, [0.0 for i in range(ncost)]
            for _ in range(args.sample_K):
                s = Etest.reset()
                while True:
                    a, _ = As[0].select_action(s, True)
                    if Etest.continuous:
                        a = ScaleA(a)
                    ns, r, c, eval_done, _ = Etest.step(a)
                    eval_R += r 
                    eval_Cs = [eval_Cs[i]+c[i] for i in range(ncost)]
                    s = ns
                    if eval_done:
                        break
            writer.add_scalar("Reward", eval_R/args.sample_K, t)
            for i in range(ncost):
                writer.add_scalar("Cost-{}".format(i), eval_Cs[i]/args.sample_K, t)
            if train_t % 50 == 0:
                print("Train Step {}: Cost {}, Reward {}, Lambda {}.".format(train_t, [eval_Cs[i]/args.sample_K for i in range(ncost)],
                                eval_R/args.sample_K, [[As[i].lamb[j].item() for j in range(As[i].nC)] for i in range(FedN)]))
            train_t += 1
            Traj_num = [0 for i in range(FedN)]

            if train_t%args.log_ep == 0:
                for i in range(FedN):
                    As[i].save_model(model_dir, train_t, i)

            if (not args.agent=="trpo") or all_local_success:
                # print("Communication Success!")
                for i in range(FedN):
                    As[i].assign_policy(central_A.target_policy.actor.parameters(), central_A.target_policy.critic_r.parameters())
