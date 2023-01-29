import gym
import numpy as np
from IPython import embed
from .config import load_config, STR_CONSTRAINT

def extend_state_fn(ctype):
    def theta1(s):
        theta = np.arctan2(s[1], s[0])
        return theta
    def theta2(s):
        theta = np.arctan2(s[3], s[2])
        return theta
    if ctype == "theta1":
        return theta1
    elif ctype == "theta2":
        return theta2
    else:
        raise NotImplementedError("{} is not a proper extended state!".format(ctype))


class FedConsGym:
    def __init__(self, args):
        self.ename = args.env_name.split('-')[0]
        self.gamma = args.gamma

        self.e = gym.make(args.env_name)
        self.nS = self.e.observation_space.shape[0]

        if self.ename in ["Acrobot", "CartPole"]:
            self.continuous = False
            self.nA = self.e.action_space.n

        elif self.ename in ["InvertedPendulum", "InvertedDoublePendulum"]:
            self.continuous = True
            self.nA = self.e.action_space.shape[0]

        else:
            raise NotImplementedError

        self.config = load_config(self.ename, args.config_name)
        self.state_dict, self.extend_state_list, self.constraints = None, None, None

    def step(self, a):
        Cs = [C(self.s,a) for C in self.constraints]
        ns, r, done, _ = self.e.step(a)
        self.s = ns
        self.l += 1
        return ns, r, Cs, done, _

    def reset(self):
        self.s = self.e.reset()
        self.l = 0
        return self.s

    def extend_state(self):
        pass

    def _make_constraint(self, c):
        class constraint:
            def __init__(self, c, ctypes, extend_ctypes):
                self.state_dict, self.extend_state, self.activate_a = ctypes, extend_ctypes, None
                self.idx, self.extend_state_fn = None, None
                if c[0] in self.state_dict.keys():
                    self.idx = self.state_dict[c[0]]
                elif c[0] in self.extend_state:
                    self.extend_state_fn = extend_state_fn(c[0])
                else:
                    raise NotImplementedError("{} is not a proper constraint!".format(c[0]))

                
                if type(c[1]) == str:
                    self.ctype = c[1]
                    if c[1] in STR_CONSTRAINT.keys():
                        self.cost_range = STR_CONSTRAINT[c[1]]
                        if len(c)>=3:
                            self.activate_a = c[2]
                    elif c[1]=="mod":
                        # for certain state vector, penalize modular hazard zone
                        self.mod_l, self.mod_n, self.mod_m = c[2], c[3], c[4]
                    else:
                        raise NotImplementedError("Improper constraint type!")
                elif type(c[1]=="list"):
                    self.ctype = "hazard"
                    self.cost_range = c[1]
                else:
                    raise NotImplementedError("Improper constraint type!")
            
            def __call__(self, s, a):
                if self.ctype == "mod":
                    mod_s = abs(s[self.idx])//self.mod_l
                    if mod_s % self.mod_n == self.mod_m:
                        return 1
                    return 0
                else:
                    if self.idx is None:
                        s = self.extend_state_fn(s)
                    else:
                        s = s[self.idx]
                    for r in self.cost_range:
                        if s>=r[0] and s<=r[1]:
                            if self.activate_a==None or self.activate_a==a:
                                return 1
                    return 0
        
        C = constraint(c, self.state_dict, self.extend_state_list)
        return C

class CartPole(FedConsGym):
    def __init__(self, args):
        super().__init__(args)

        self.state_dict = {
            "pos": 0, "vel": 1,
            "angle": 2, "vel_angle": 3
        }
        self.constraints = [self._make_constraint(c) for c in self.config["constraints"]]

class Acrobot(FedConsGym):
    def __init__(self, args):
        super().__init__(args)

        self.state_dict = {
            "cos_theta1": 0, "sin_theta1": 1,
            "cos_theta2": 2, "sin_theta2": 3,
            "vel_theta1": 4, "vel_theta2": 5
        }
        self.extend_state_list = ["theta1", "theta2"]
        self.constraints = [self._make_constraint(c) for c in self.config["constraints"]]
        self.H4R = 0.7

    def extend_state_fn(self, state_name):
        def theta1(s):
            return np.arctan2(s[1], s[0])
        def theta2(s):
            return np.arctan2(s[3], s[2])

        if state_name == "theta1":
            return theta1
        else:
            return theta2

    def step(self, a):
        Cs = [c(self.s, a) for c in self.constraints]
        ns, r, _, _ = self.e.step(a)
        done = False
        if self.l > 500:
            done = True
        if -ns[0]-(ns[0]*ns[2]-ns[1]*ns[3])>self.H4R:
            r = 1.0
        else:
            r = (-ns[0]-(ns[0]*ns[2]-ns[1]*ns[3])-self.H4R)*0.001
        self.s = ns
        self.l += 1
        return ns, r, Cs, done, _



class InvertedPendulum(FedConsGym):
    def __init__(self, args):
        super().__init__(args)

        self.state_dict = {
            "pos": 0, "angle": 1,
            "vel": 2, "vel_angle": 3
        }
        self.constraints = [self._make_constraint(c) for c in self.config["constraints"]]
        self.action_low = self.e.action_space.low
        self.action_high = self.e.action_space.high 
        self.action_mean = (self.action_low+self.action_high)*0.5
        self.action_range = self.action_high-self.action_low
        self.pos_max = 2.4
        self.L_max = 500

    def step(self, a):
        Cs = [C(self.s,a) for C in self.constraints]
        ns, r, done, _ = self.e.step(a)
        self.s = ns
        self.l += 1
        if abs(self.s[0])>self.pos_max:
            done = True
            r = -1.0
        if self.l > self.L_max:
            done = True
        return ns, r, Cs, done, _

class InvertedDoublePendulum(FedConsGym):
    def __init__(self, args):
        super().__init__(args)

        self.state_dict = {
            "pos": 0, "sin_theta1": 1, "sin_theta2": 2,
            "cos_theta1": 3, "cos_theta2": 4, "vel": 5,
            "vel_theta1": 6, "vel_theta2": 7, "cforce1": 8,
            "cforce2": 9, "cforce3": 10
        }
        self.constraints = [self._make_constraint(c) for c in self.config["constraints"]]
        self.action_low = self.e.action_space.low
        self.action_high = self.e.action_space.high 
        self.action_mean = (self.action_low+self.action_high)*0.5
        self.action_range = self.action_high-self.action_low

