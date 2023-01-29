import random
from collections import deque

class Traj_Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.costs = []
        self.dones = []

    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.costs = []
        self.dones = []

    def insert(self, s, a, logp, r, c, d):
        self.states.append(s)
        self.actions.append(a)
        self.logps.append(logp)
        self.rewards.append(r)
        self.costs.append(c)
        self.dones.append(d)

class Trans_Buffer:
    def __init__(self, size):
        self.buffer = deque()
        self.size = 0
        self.max_size=size

    def insert(self, s, a, r, c, ns, done, info):
        if self.size<self.max_size:
            self.buffer.append((s, a, r, c, ns, done, info))
            self.size += 1
        else:
            self.buffer.popleft()
            self.buffer.append((s, a, r, c, ns, done, info))

    def sample(self, bs):
        batch = random.sample(self.buffer, bs)
        return batch