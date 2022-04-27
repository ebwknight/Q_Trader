import numpy as np
import random as rand

class TabularQLearner:
    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9, epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):
        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.actions = actions
        self.states = states
        self.epsilson = epsilon
        self.ed = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna
        self.ptsd = []
        self.prev_state = None
        self.prev_action = None
        self.Qtab = np.ones([states, actions]) * 0.00001

    def train (self, s, r):
        # Receive new state s and new reward r.  Update Q-table and return selected action.
        Qprev = self.Qtab[self.prev_state, self.prev_action]
        Qmax = np.max(self.Qtab[s])
        Qnew = ((1 - self.alpha) * Qprev) + (self.alpha * (r + (self.gamma * Qmax)))
        self.Qtab[self.prev_state, self.prev_action] = Qnew

        next_action = np.argmax(self.Qtab[s]) #optimal by default
        if (rand.random() < self.epsilson):
            next_action = rand.randint(0,self.actions-1) #explore

        self.ptsd.append([self.prev_state, self.prev_action, s, r]) #append to memories
        #update instance variables
        self.prev_action = next_action
        self.prev_state = s
        self.epsilson *= self.ed

        #Preform Dyna
        for i in range(self.dyna):
            sample = self.ptsd[rand.randint(0, len(self.ptsd)-1)] #sample memories
            #update q table
            Qprev = self.Qtab[sample[0], sample[1]]
            Qmax = np.max(self.Qtab[sample[2]])
            Qnew = ((1 - self.alpha) * Qprev) + (self.alpha * (sample[3] + (self.gamma * Qmax)))
            self.Qtab[sample[0], sample[1]] = Qnew

        return next_action

    def test (self, s):
        # Receive new state s.  Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # When testing, you probably do not want to take random actions... (What good would it do?)
        #need to return location of max q value, not the value itself
        a = np.argmax(self.Qtab[s])
        self.prev_state = s
        self.prev_action = a
        return a