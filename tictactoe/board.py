# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 02:07:45 2020

@author: prodi
"""

""" board class. """
import numpy as np
import hashlib

class Board():
    
    def __init__(self, player1=0, player2=1, blank=None):
        self._player1 = player1
        self._player2 = player2
        
        if blank=='':
            blank = ' '
        self._blank = blank
        
        self.reset()
    
    def __str__(self):
        value = ''
        for idx, line in enumerate(self._board):
            value = value + '{:^5}|{:^5}|{:^5}'.format(str(line[0]),str(line[1]),str(line[2])) + '\n'
            if idx != 2:value = value + '-----+-----+-----\n'
        return value
    
    def __repr__(self):
        return self.__str__()
    
    @property
    def board(self):
        return self._board
    
    @property
    def state(self):
        return hashlib.md5(str(self._board).encode()).hexdigest()
    
    @property
    def players(self):
        return self._player1, self.player2
    
    @property
    def winner(self):
        winner = self._check_winner()
        if winner != -1:
            self._winner = self._winner
        return self._winner
    
    @property
    def next_player(self):
        return self._next_player
    
    @property
    def moves(self):
        rows, cols = np.where(self._board==self._blank)
        return [e for e in zip(rows, cols)]
    
    @property
    def history(self):
        return self._history
    
    @property
    def game_over(self):
        winner = self._check_winner()
        
        if winner == -1:
            return False
        
        self._winner = winner
        return True
    
    def _lines(self):
        for row in self._board:yield row  
        for colum in self._board.T:yield colum
        yield self._board.diagonal()
        yield np.fliplr(self._board).diagonal()

    def _check_winner(self):
        for line in self._lines():
            if np.all(line==self._player1):
                self._winner = self._player1
                return self._player1
            elif np.all(line==self._player2):
                self._winner = self._player2
                return self._player2
            elif np.count_nonzero(line==self._blank) > 1:
                return -1
            elif len(self.moves) > 1:
                return -1
        return None
    
    def _process_move(self, player, move):
        self._board[move[0],move[1]] = player
        winner = self._check_winner()
        self._history.append((player, move, winner))
        self._next_player = self._player1 \
                if self._player2==player else self._player2

    def make_move(self, player, move):
        if self._next_player != player:
            raise ValueError('player {} is playing out of turn'.format(player))
        if self._board[move[0],move[1]] != self._blank:
            raise ValueError('illegal move, cell already marked.')
        
        self._process_move(player, move)
        
    def random_move(self):
        moves = self.moves
        if moves:
            idx = np.random.choice(range(len(moves)))
            return moves[idx]
            
    def make_random_move(self, player=None):
        moves = self.moves
        if player and self._next_player != player:
            raise ValueError('player {} is playing out of turn'.format(player))
        if moves:
            move = self.random_move()
            self.make_move(self.next_player, move)
            
    def reset(self):
        self._board = np.full((3,3),self._blank)
        self._next_player = self._player1
        self._winner = None
        self._history = []
        
    def step(self, player, move):
        reward = 0
        terminated = True
        
        if move is None:
            self.make_random_move(player)
        else:
            self.make_move(player, move)
    
        other_player = self._next_player
        self.make_random_move()
        winner = self._check_winner()
        
        if winner == player:
            reward = 100
        elif winner is None:
            reward = 0
        elif winner == other_player:
            reward = -100
        else:
            terminated = False
            
        return self.state, reward, terminated
        