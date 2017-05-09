"""
Ningmu Zou
nzou3@gatech.edu
"""

import numpy as np
import random as rand


class QLearner(object):
    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.q_table = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        self.t_diction = {}
        self.r = -np.ones((num_states, num_actions))

    def author(self):
        return 'nzou3'  # replace tb34 with your Georgia Tech username.

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """

        if np.random.random_sample() < self.rar:
            action = int(np.random.random()*self.num_actions) # Do random action
        else:
            action = np.argmax(self.q_table[s, :])  # Choose action from Qtable

        self.s = s
        self.a = action

        if self.verbose: print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        # Q_table update according to learning step
        self.q_table[self.s, self.a] = ((1 - self.alpha) * self.q_table[self.s, self.a]) + \
                                       self.alpha * (r + self.gamma * self.q_table[s_prime, :].max())

        # If dyna is required
        if self.dyna > 0:
            #generate a r table
            alpha = self.alpha
            gamma = self.gamma
            s = self.s
            a = self.a
            # reward update according to a and s
            self.r[s][a] = (1 - alpha) * (self.r[s][a]) + alpha * r
            # put s_prime into dictionaty with the index of a and s
            if self.t_diction.get((s, a)) == None:
                self.t_diction[(s, a)] = [(s_prime)]
            else:
                self.t_diction[(s, a)].append(s_prime)

        if self.dyna > 0:
            k = 0
            while k < self.dyna:
                rand_s = rand.randint(0, self.num_states - 1)
                rand_a = rand.randint(0, self.num_actions - 1)
                # if there is no such s_prime, generate a random number. Or, random choice a number
                if self.t_diction.get((rand_s, rand_a)) == None:
                    s_prime_temp = rand.randint(0, self.num_states - 1)
                else:
                    p=len(self.t_diction[(rand_s, rand_a)])
                    s_prime_temp_ind = rand.randint(0, p-1)
                    s_prime_temp=self.t_diction[(rand_s, rand_a)][s_prime_temp_ind]

                # get the q_table max at first
                q_table_max = self.q_table[s_prime_temp, :].max()

                #update q_table
                self.q_table[rand_s, rand_a] = ((1 - self.alpha) * self.q_table[rand_s, rand_a]) + self.alpha * (
                    self.r[rand_s][rand_a] + self.gamma * q_table_max)
                k += 1


        self.rar *= self.radr
        self.s = s_prime
        if np.random.random_sample() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[self.s, :])

        self.a = action

        if self.verbose: print "s =", s_prime, "a =", action, "r =", r
        return action


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
