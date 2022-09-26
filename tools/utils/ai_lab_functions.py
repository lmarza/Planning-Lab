from collections import deque
import heapq
import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, state, parent=None, pathcost=0, value=0):
        self.state = state
        self.pathcost = pathcost
        self.value = value
        self.parent = parent
        self.removed = False

    def __hash__(self):
        return int(self.state)

    def __lt__(self, other):
        return self.value < other.value


class NodeQueue():
    
    def __init__(self):
        self.queue = deque()
        self.node_dict = {}
        self.que_len = 0

    def is_empty(self):
        return (self.que_len == 0)

    def add(self, node):
        self.node_dict[node.state] = node
        self.queue.append(node)
        self.que_len += 1

    def remove(self):
        while True:
            n = self.queue.popleft()
            if not n.removed:
                if n.state in self.node_dict:
                    del self.node_dict[n.state]
                self.que_len -= 1
                return n

    def __len__(self):
        return self.que_len

    def __contains__(self, item):
        return item in self.node_dict

    def __getitem__(self, i):
        return self.node_dict[i]


class PriorityQueue():
    def __init__(self):
        self.fringe = []
        self.frdict = {} 
        self.frlen = 0

    def is_empty(self):
        return self.frlen == 0

    def add(self, n):
        heapq.heappush(self.fringe, n)
        self.frdict[n.state] = n
        self.frlen += 1

    def remove(self):
        while True:
            n = heapq.heappop(self.fringe)
            if not n.removed:
                if n.state in self.frdict:
                    del self.frdict[n.state]
                self.frlen -= 1
                return n

    def replace(self, n):
        self.frdict[n.state].removed = True
        self.frdict[n.state] = n
        self.frlen -= 1
        self.add(n)

    def __len__(self):
        return self.frlen

    def __contains__(self, item):
        return item in self.frdict

    def __getitem__(self, i):
        return self.frdict[i]


class Heu():
    @staticmethod
    def l1_norm(p1, p2):
        return np.sum(np.abs(np.asarray(p1) - np.asarray(p2)))

    @staticmethod
    def l2_norm(p1, p2):
        return np.linalg.norm((np.asarray(p1) - np.asarray(p2)))

    @staticmethod
    def chebyshev(p1, p2):
        return np.max(np.abs(np.asarray(p1) - np.asarray(p2)))


def build_path(node):
    path = []
    while node.parent is not None:
        path.append(node.state)
        node = node.parent
    return tuple(reversed(path))


def solution_2_string(sol, env):
    if( not isinstance(sol, tuple) ):
        return sol

    if sol is not None:
        solution = [env.state_to_pos(s) for s in sol]
    return solution


def zero_to_infinity():
    i = 0
    while True:
        yield i
        i += 1

def run_episode(environment, policy, limit):
    obs = environment.reset()
    done = False
    reward = 0
    s = 0
    while not done and s < limit:
        obs, r, done, _ = environment.step(policy[obs])
        reward += r
        s += 1
    return reward

def plot(series, title, xlabel, ylabel):
        plt.figure(figsize=(13, 6))
        for s in series:
            plt.plot(s["x"], s["y"], label=s["label"])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()

        
def values_to_policy(U, env):
    p = [0 for _ in range(env.observation_space.n)]
    
    for state in range(env.observation_space.n):
        max_array = [0 for _ in range(env.action_space.n)]
        for action in range(env.action_space.n):
            for next_state in range(env.observation_space.n):
                max_array[action] += env.T[state, action, next_state] * U[next_state]
                
        max_array = np.round(max_array, 6)
        winners = np.argwhere(max_array == np.amax(max_array)).flatten()
        win_action = winners[0]#np.random.choice(winners)
        p[state] = win_action
                
    return np.asarray(p)


def rolling(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.mean(np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides), -1)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class CheckResult_L1A1():

    def __init__(self, student_ts_sol, student_gs_sol, env):
        # student_ts_sol is a list where student_ts_sol[0] = solution_ts, student_ts_sol[1] = time_ts, student_ts_sol[2] = memory_ts
        # same fore the graph search solutions
        self.student_ts_sol = student_ts_sol
        self.student_gs_sol = student_gs_sol
        self.env = env

    def check_sol_ts(self):
        print(bcolors.OKCYAN + '##########################################' + bcolors.ENDC)
        print(bcolors.OKCYAN + '#######  BFS TREE SEARCH PROBLEM  ########' + bcolors.ENDC)
        print(bcolors.OKCYAN + '##########################################'+ bcolors.ENDC)


        print("Your solution: {}".format(solution_2_string(self.student_ts_sol[0], self.env)))
        print("N° of nodes explored: {}".format(self.student_ts_sol[1]))
        print("Max n° of nodes in memory: {}\n".format(self.student_ts_sol[2]))

        if solution_2_string(self.student_ts_sol[0], self.env) != [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]:
            print(bcolors.FAIL + "> Your solution is not correct, should be: \n[(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]\n" + bcolors.ENDC)
        elif self.student_ts_sol[1] != 103721:
            print(bcolors.FAIL + "> The number of node explored is not correct, should be: 103721\n" + bcolors.ENDC)
        elif self.student_ts_sol[2] != 77791:
            print(bcolors.FAIL + "> The max number of nodes in memory is not correct, should be: 77791\n" + bcolors.ENDC)
        else:
            print(bcolors.BOLD + bcolors.OKGREEN + '===> Your solution is correct!\n'+ bcolors.ENDC)


    def check_sol_gs(self):
        print(bcolors.OKCYAN + '##########################################' + bcolors.ENDC)
        print(bcolors.OKCYAN + '#######  BFS Graph SEARCH PROBLEM  #######' + bcolors.ENDC)
        print(bcolors.OKCYAN + '##########################################'+ bcolors.ENDC)

        print("Solution: {}".format(solution_2_string(self.student_gs_sol[0], self.env)))
        print("N° of nodes explored: {}".format(self.student_gs_sol[1]))
        print("Max n° of nodes in memory: {}\n".format(self.student_gs_sol[2]))

        if solution_2_string(self.student_gs_sol[0], self.env) != [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]:
            print(bcolors.FAIL + "> Your solution is not correct, should be: \n[(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]\n" + bcolors.ENDC)
        elif self.student_gs_sol[1] != 57:
            print(bcolors.FAIL + "> The number of node explored is not correct, should be: 57\n" + bcolors.ENDC)
        elif self.student_gs_sol[2] != 15:
            print(bcolors.FAIL + "> The max number of nodes in memory is not correct, should be: 15\n" + bcolors.ENDC)
        else:
            print(bcolors.BOLD + bcolors.OKGREEN + '===> Your solution is correct!\n'+ bcolors.ENDC)


class CheckResult_L1A2():
    def __init__(self, student_ts_sol, student_gs_sol, env):
       
        # student_ts_sol is a list where student_ts_sol[0] = solution_ts, student_ts_sol[1] = time_ts, student_ts_sol[2] = memory_ts, student_ts_sol[3] = iterations_ts
        # same fore the graph search solutions
        self.student_ts_sol = student_ts_sol
        self.student_gs_sol = student_gs_sol
        self.env = env

    def check_sol_ts(self):
        print(bcolors.OKCYAN + '##########################################' + bcolors.ENDC)
        print(bcolors.OKCYAN + '#######  IDS TREE SEARCH PROBLEM  ########' + bcolors.ENDC)
        print(bcolors.OKCYAN + '##########################################'+ bcolors.ENDC)

        print("Necessary Iterations: {}".format(self.student_ts_sol[3]))
        print("Your solution: {}".format(solution_2_string(self.student_ts_sol[0], self.env)))
        print("N° of nodes explored: {}".format(self.student_ts_sol[1]))
        print("Max n° of nodes in memory: {}\n".format(self.student_ts_sol[2]))

        if self.student_ts_sol[3] != 9:
            print(bcolors.FAIL + "> Your necessary iterations are not correct, should be: 9\n" + bcolors.ENDC)
        elif solution_2_string(self.student_ts_sol[0], self.env) != [(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]:
            print(bcolors.FAIL + "> Your solution is not correct, should be: \n[(0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]\n" + bcolors.ENDC)
        elif self.student_ts_sol[1] != 138298:
            print(bcolors.FAIL + "> The number of node explored is not correct, should be: 138298\n" + bcolors.ENDC)
        elif self.student_ts_sol[2] != 9:
            print(bcolors.FAIL + "> The max number of nodes in memory is not correct, should be: 9\n" + bcolors.ENDC)
        else:
            print(bcolors.BOLD + bcolors.OKGREEN + '===> Your solution is correct!\n'+ bcolors.ENDC)


    def check_sol_gs(self):
        print(bcolors.OKCYAN + '##########################################' + bcolors.ENDC)
        print(bcolors.OKCYAN + '#######  IDS GRAPH SEARCH PROBLEM  #######' + bcolors.ENDC)
        print(bcolors.OKCYAN + '##########################################'+ bcolors.ENDC)

        print("Necessary Iterations: {}".format(self.student_gs_sol[3]))
        print("Solution: {}".format(solution_2_string(self.student_gs_sol[0], self.env)))
        print("N° of nodes explored: {}".format(self.student_gs_sol[1]))
        print("Max n° of nodes in memory: {}\n".format(self.student_gs_sol[2]))

        if self.student_gs_sol[3] != 11:
            print(bcolors.FAIL + "> Your necessary iterations are not correct, should be: 11\n" + bcolors.ENDC)
        elif solution_2_string(self.student_gs_sol[0], self.env) != [(0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]:
            print(bcolors.FAIL + "> Your solution is not correct, should be: \n[(0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3)]\n" + bcolors.ENDC)
        elif self.student_gs_sol[1] != 132:
            print(bcolors.FAIL + "> The number of node explored is not correct, should be: 132\n" + bcolors.ENDC)
        elif self.student_gs_sol[2] != 11:
            print(bcolors.FAIL + "> The max number of nodes in memory is not correct, should be: 11\n" + bcolors.ENDC)
        else:
            print(bcolors.BOLD + bcolors.OKGREEN + '===> Your solution is correct!\n'+ bcolors.ENDC)