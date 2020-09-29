# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 02:44:27 2020

@author: prodi
"""
import numpy as np
import pandas as pd

from tictactoe.board import Board

class QMatrix():
    
    def __init__(self, actions, states=[]):
        self._actions = {actions[i]:i for i in range(len(actions))}
        self._states = {states[i]:i for i in range(len(states))}
        self._Q = np.zeros((len(states),len(actions)))
        
    def __str__(self):
        return str(self.to_pandas())
        
    def __repr__(self):
        return self.__str__()
        
    def __getitem__(self, move):
        if not isinstance(move, tuple):
            raise ValueError(f'not a tuple {move}.')
        
        state = move[0]
        action = move[1]
        
        if isinstance(state, slice):
            state_idx = state
        elif state in self._states:
            state_idx = self._states[state]
        else:
            state_idx = None
            
        if isinstance(action, slice):
            action_idx = action
            if state_idx is None:
                return np.zeros([len(self._states)])
        elif action in self._actions:
            if state_idx is None:
                return 0
            action_idx = self._actions[action]
        else:
            raise KeyError(f'action {action} is not defined.')
        
        return self._Q[state_idx, action_idx]
    
    def __setitem__(self, move, value):
        if not isinstance(move, tuple):
            raise ValueError(f'not a tuple {move}.')
        
        state = move[0]
        action = move[1]
        
        if action not in self._actions:
            raise ValueError(f'illegal action {action}.')
        
        if state not in self._states:
            self._states[state] = len(self._states)
            row = np.zeros((1,len(self._actions)))
            if self._Q.size==0:
                self._Q = row
            else:
                self._Q = np.vstack([self._Q,row])
        
        state_idx = self._states[state]
        action_idx = self._actions[action]
        self._Q[state_idx,action_idx] = value
        
    @property
    def Q(self):
        return self._Q
        
    @property
    def states(self):
        return list(self._states.keys())
    
    @property
    def actions(self):
        return list(self._actions.keys())
    
    def to_pandas(self):
        if self._Q.size == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(self._Q, columns=self.actions)

class Agent():
    
    def __init__(self, Q=None, gamma=0.1, eps=0.5, nu=0.6, 
                 eps_decay=0):
        self._gamma = gamma
        self._nu = nu
        self._eps = eps
        self._eps_decay = eps_decay
        self._actions = [(i,j) for i in range(3) for j in range(3)]
        self.reset(Q)
        
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def nu(self):
        return self._nu
    
    @property
    def eps(self):
        return self._eps
    
    @property
    def eps_decay(self):
        return self._eps_decay
    
    @property
    def Q(self):
        return self._Q
    
    @property
    def actions(self):
        return self._actions
    
    def random_action(self, board):
        return board.random_move()
    
    def best_action(self, state, board):
        if state not in self.Q.states:
            return self.random_action(board)
        
        moves = board.moves
        choice = self.random_action(board)
        bestq = self.Q[state, choice]
        for action in moves:
            if bestq < self.Q[state, action]:
                choice = action
                bestq = self.Q[state, action]
        return choice
        
    def max_q(self, state):
        try:
            return np.max(self.Q[state,:])
        except:
            return 0
    
    def reset(self, Q=None):
        if Q is None:
            self._Q = QMatrix(self._actions)
        else:
            self._Q = Q
    
    def train(self, player='X', n=100000):
        self.reset()
        
        for episode in range(n):
            board = Board('X','O', ' ')
            state = board.state
            reward = 0
            
            if player != 'X':
                board.make_random_move()
            
            while not board.game_over:
                if np.random.uniform() < self.eps:
                    action = self.random_action(board)
                else:
                    action = self.best_action(state, board)
                
                next_state, reward, terminated = \
                                board.step(player, action)
                q = self.Q[state, action]
                max_q = self.max_q(next_state)
                next_q = (1-self.nu)*q + self.nu*(reward + self.gamma*max_q)
                self.Q[state, action] = next_q
                state = next_state
                
    def score(self, player='X', n=100):
        wins = draws = losses = 0
        
        for i in range(n):
            board = Board('X','O', ' ')
            if player != 'X':
                board.make_random_move()
                
            while not board.game_over:
                state = board.state
                action = self.best_action(state, board)
                next_state, reward, terminated = board.step(player, action)
                if terminated:
                    if reward == 0:
                        draws = draws + 1
                    elif reward > 0:
                        wins = wins + 1
                    else:
                        losses = losses + 1
                        
        return round(wins/n,2), round(draws/n,2), round(losses/n,2)


        
        
        