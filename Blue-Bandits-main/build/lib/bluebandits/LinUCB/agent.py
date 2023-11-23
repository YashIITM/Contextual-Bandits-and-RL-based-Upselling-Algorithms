import numpy as np

import torch
import random
from copy import deepcopy

class BestAgent:
    def __init__(self, K, T, d, A):
        # K is Total number of actions,
        # T is Total number of periods
        # d is the dimension of context
        # A is the context
        self.K = K
        self.T = T
        self.d = d
        self.t = 0  # marks the index of period
        self.A = A
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.history_context = np.zeros((d, T))

    def Action(self, context_list):
        # context_list is a d*K matrix, each column represent a context
        # the return value is the action we choose, represent the index of action, is a scalar

        expected_reward = np.zeros(self.K)
        for kk in range(0, self.K):
            context = context_list[kk, :]
            expected_reward[kk] = context.transpose().dot(self.A.transpose().dot(self.A)).dot(context)
        ind = np.argmax(expected_reward, axis=None)
        self.history_context[:, self.t] = context_list[ind, :]
        self.history_action[self.t] = ind
        return ind

    def Update(self, reward):
        # reward is the realized reward after we adopt policy, a scalar
        self.history_reward[self.t] = reward
        self.t = self.t + 1

    def GetHistoryReward(self):
        return self.history_reward

    def GetHistoryAction(self):
        return self.history_action

    def GetHistoryContext(self):
        return self.history_context


class UniformAgent:
    def __init__(self, K, T, d):
        # K is Total number of actions,
        # T is Total number of periods
        # d is the dimension of context
        self.K = K
        self.T = T
        self.d = d
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.history_context = np.zeros((d, T))

    def Action(self, context_list: np.array) -> int:
        # context_list is a d*K matrix, each column represent a context
        # the return value is the action we choose, represent the index of action, is a scalar

        ind = np.random.randint(0, high=self.K)  # we just uniformly choose an action
        self.history_context[:, self.t] = context_list[ind, :]
        return ind

    def Update(self, reward):
        # reward is the realized reward after we adopt policy, a scalar
        self.history_reward[self.t] = reward
        self.t = self.t + 1

    def GetHistoryReward(self):
        return self.history_reward

    def GetHistoryAction(self):
        return self.history_action

    def GetHistoryContext(self):
        return self.history_context

class Agent:
    def __init__(
        self,
        K: int,
        T: int,
        d: int,
        A,
        theta,
        X,
        p,
        alpha
    ):
        """The proposed Neural UCB algorithm for solving contextual bandits

        Args:
            K (int): Number of arms
            T (int): Number of rounds
            d (int): Dimension of context
            L (int, optional): Number of Layers. Defaults to 2.
            m (int, optional): Width of each layer. Defaults to 20.
            gamma_t (float, optional): Exploration parameter. Defaults to 0.01.
            v (float, optional): Exploration parameter. Defaults to 0.1.
            lambda_ (float, optional): Regularization parameter. Defaults to 0.01.
            delta (float, optional): Confidence parameter. Defaults to 0.01.
            S (float, optional): Norm parameter. Defaults to 0.01.
            eta (float, optional): Step size. Defaults to 0.001.
            frequency (int, optional): The interval between two training rounds. Defaults to 50.
            batchsize (int, optional): The batchsize of applying SGD on the neural network. Defaults to None.
        """
        self.K = K
        self.T = T
        self.d = d
        self.A = A
        self.theta = theta
        self.p = p
        self.X = X
        self.alpha = alpha
        self.t = 0  # marks the index of period
        self.history_reward = np.zeros(T)
        self.history_action = np.zeros(T)
        self.predicted_reward = np.zeros(T)
        self.predicted_reward_upperbound = np.zeros(T)
        self.history_context = np.zeros((T, d))

        

    def Action(self, context_list: np.array) -> int:
        """Given the observed context of each arm, return the predicted arm

        Args:
            context_list (np.array): The observed context of each arm. context_list.shape = (K, d)

        Returns:
            int: the index of predicted arm, take value from 0, 1, ..., K-1
        """
        

        
        for arm in range(self.K):
            inv_A = np.linalg.inv(self.A[arm])
            self.theta[self.t,arm] = inv_A.dot(context_list[arm])
            self.p[self.t,arm] = self.theta[self.t,arm].dot(self.X[self.t,arm])+self.alpha*np.sqrt(self.X[self.t,arm].dot(inv_A).dot(self.X[self.t,arm]))


        # calculate the upper confidence bound
        chosen_arm = np.argmax(self.p[self.t])
        ind = chosen_arm


        # save the history
        self.history_action[self.t] = ind
        self.history_context[self.t, :] = context_list[ind, :]
        #self.predicted_reward[self.t] = predict_reward[ind]
        self.predicted_reward_upperbound = self.p[self.t][ind]
        return ind

    def Update(self, reward):
        self.history_reward[self.t] = reward
        ind = int(self.history_action[self.t])
        
        context = self.history_context[self.t, :]

        temp_vec = self.A[ind]
        self.A[ind] = temp_vec+ np.outer(self.X[self.t,ind],self.X[self.t,ind].T)
        context+=reward*self.X[self.t,ind]
        self.t += 1