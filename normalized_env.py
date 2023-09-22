from copy import deepcopy
import gym
from envs_numpy import gen_random_directions
import networkx as nx
from networkx.algorithms import bipartite
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import dgl
import sys
import numpy as np
import random
# from model import GNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MyEnv():
    seed: int = 42

    def __init__(self, users, players, sigma, reward_function, utility_function):
        self.observation_space = players
        self.action_space = users
        self.users = users
        self.players = players
        self.n_user = len(users)
        self.n_player = len(players)
        self.G = nx.DiGraph()
        self.dim = users.shape[1]
        self.sigma = sigma
        self.R = reward_function
        # self.gnn = GNN()

        for index in range(self.n_user):
            self.G.add_node(index, position=users[index], bipartite=0)
        for index in range(self.n_player):
            self.G.add_node(index + self.n_user, position=players[index], bipartite=1)

        adj_mat = np.zeros((self.n_user, self.n_player))
        for i in range(self.n_user):
            for j in range(self.n_player):
                self.G.add_edge(i, j + self.n_user, weight=self.sigma(players[j], users[i]))
                self.G.add_edge(j + self.n_user, i, weight=self.sigma(players[j], users[i]))
        self.G = dgl.from_networkx(self.G, node_attrs=["position"], edge_attrs=["weight"])
        self.G = self.G
        self.initial_G = deepcopy(self.G)
        self.u = utility_function

    def step(self, action, lr=0.03, tau=0.1, top_k=1, return_W=False, by_user=False):
        users = self.G.ndata["position"][0:self.n_user].numpy() # positions of users
        S = self.G.ndata["position"][self.n_user:].numpy() # position of players
        # seems like dgl automatically transform the node and edge info into tensors
        if action is None:
            action = np.ones(self.n_user)
        
        utilities, allocation_matrix = self.u(S, users, tau=tau, top_k=top_k, user_weight=action)

        ''' two actions + stay '''
        g1 = gen_random_directions(self.n_player, self.dim)
        g2 = gen_random_directions(self.n_player, self.dim)
        # find improvement direction
        utilities_new1, _ = self.u(S + g1 * lr, users, tau=tau, top_k=top_k, user_weight=action)
        utilities_new2, _ = self.u(S + g2 * lr, users, tau=tau, top_k=top_k, user_weight=action)
        
        # select two actions, calculate the utility difference
        # if the utility decreases, stay
        diff1 = utilities_new1 - utilities
        diff2 = utilities_new2 - utilities
        g = np.where(diff1 > diff2, g1.T, g2.T).T
        diff = np.where(diff1 > diff2, diff1, diff2)
        diff = np.expand_dims(diff, 1)
        g = np.where(diff > 0, g, np.zeros(g.shape))

        # print("====================================")
        # np.set_printoptions(precision=3, sci_mode=False)
        # print(S + g * lr)
        if by_user:
            utilities_new, _, utilities_by_user = self.u(S + g * lr, users, tau=tau, top_k=top_k, user_weight=np.array([1,1,1,1,1]), by_user=by_user)
        else:
            utilities_new, _ = self.u(S + g * lr, users, tau=tau, top_k=top_k, user_weight=np.array([1,1,1,1,1]), by_user=False)
        print("utilities by player: ")
        print(utilities_new)
        old_S = deepcopy(S)
        S += g * lr
        print("strategy: ")
        print(S)
        # utility_for_training = self.u(S + g * lr, users, tau=tau, top_k=top_k, user_weight=action)
        old_S_matrix = np.zeros((self.n_player, self.n_user))
        for i_strategy in range(self.n_player):
            for i_user in range(self.n_user):
                old_S_matrix[i_strategy][i_user] = self.sigma(users[i_user], old_S[i_strategy])
        old_reward = self.R(old_S_matrix)
        new_S_matrix = np.zeros((self.n_player, self.n_user))
        for i_strategy in range(self.n_player):
            for i_user in range(self.n_user):
                new_S_matrix[i_strategy][i_user] = self.sigma(users[i_user], S[i_strategy])
        # print(new_S_matrix)
        new_reward = self.R(new_S_matrix)
        reward = new_reward - old_reward

        # A very naive way to early end, 
        # if the reward is larger than 49 (out of 50) we say it is good
        if new_reward >= 49:
            done = 1
        else:
            done = 0
        # return self.G, sum(utilities_new).cpu().detach(), done, None # state, reward, done, info
        if by_user:
            return self.G, new_reward, done, None, utilities_by_user
        if return_W:
            return self.G, reward, done, None, new_reward # state, reward, done, info, welfare
        else:
            return self.G, new_reward, done, None # state, reward, done, info

    def reset(self):
        # One Initial Strategy
        # self.G = deepcopy(self.initial_G)
        # return self.G
        
        # Multiple Initial Strategies

        ''' This gives a set of pre-defined cluster '''
        # initial_players = [
        #     np.array([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]])*1.5,
        #     np.array([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]])*-1.5,
        #     np.array([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]])*1.5,
        #     np.array([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]])*-1.5,
        # ]
        # initial_players = [
        #     np.array([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]])*0.5,
        #     np.array([[1.1,0.1], [0.9,0.1], [1.0,0.2], [0.8,0.1], [1.1,-0.1]])*-0.5,
        #     np.array([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]])*0.5,
        #     np.array([[0.1, 1.1], [0.1, 0.9], [0.2, 1.0], [0.1, 0.8], [-0.1, 1.1]])*-0.5,
        # ]
        # players = random.choice(initial_players)
        

        ''' This gives a randomized cluster within the entire plane'''
        initial_centroid = 3.0*np.random.rand(2)-1.5
        # print(initial_centroid)
        initial_players = np.stack([
            np.add(initial_centroid, np.array([0.1, 0.1])), 
            np.add(initial_centroid, np.array([0.1, -0.1])),
            np.add(initial_centroid, np.array([-0.1, 0.1])),
            np.add(initial_centroid, np.array([-0.1, -0.1])),
            np.add(initial_centroid, np.array([0.0, 0.0]))])
        # print("Reset initial players: ")
        # print(initial_players)
        players = initial_players

        self.new_G = nx.DiGraph()
        for index in range(self.n_user):
            self.new_G.add_node(index, position=self.users[index], bipartite=0)
        for index in range(self.n_player):
            self.new_G.add_node(index + self.n_user, position=players[index], bipartite=1)
        
        adj_mat = np.zeros((self.n_user, self.n_player))
        for i in range(self.n_user):
            for j in range(self.n_player):
                self.new_G.add_edge(i, j + self.n_user, weight=self.sigma(players[j], self.users[i]))
                self.new_G.add_edge(j + self.n_user, i, weight=self.sigma(players[j], self.users[i]))
        self.new_G = dgl.from_networkx(self.new_G, node_attrs=["position"], edge_attrs=["weight"])

        self.initial_G = deepcopy(self.new_G)
        self.G = deepcopy(self.new_G)
        return self.new_G

    def seed(self, seed):
        self.seed = seed