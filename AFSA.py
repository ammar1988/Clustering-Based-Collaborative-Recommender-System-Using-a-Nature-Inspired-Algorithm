import numpy as np
from scipy import spatial
import pandas as pd 
from sklearn.cluster import KMeans

##delete 2, -1 from r 

class AFSA:
    def __init__(self, func, n_dim, size_pop, cluster_size, max_iter,
                 max_try_num, step, visual, q, delta):
        self.func = func
        self.n_dim = n_dim
        self.size_pop = size_pop
        self.cluster_size = cluster_size
        self.max_iter = max_iter
        self.max_try_num = max_try_num  # Maximum number of prey attempts
        self.step = step  # Maximum displacement ratio for each step
        self.visual = visual  # maximum perception range of fish
        self.q = q  # fish perception range attenuation factor
        self.delta = delta  # The crowding degree threshold, the larger the easier it is to cluster and rear-end

        self.X = np.random.rand(self.size_pop, self.cluster_size, self.n_dim) * (5.0-1.0) + 1.0
        self.X = np.round_(self.X, decimals = 3)

        l = []
        for idx, x in enumerate(self.X):
           l.append(self.func(x))
        self.Y = np.array(l)
        best_idx = self.Y.argmin() #Return indices of the minimum values along the given axis
        self.best_x, self.best_y = self.X[best_idx, :], self.Y[best_idx]
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase

    def move_to_target(self, idx_individual, x_target):
        '''
        move to target
        called by prey(), swarm(), follow()

        :param idx_individual:
        :param x_target:
        :return:
        '''
        x = self.X[idx_individual, :]
        r = np.random.rand()
        # r = np.round_(r1, decimals = 3)
        x_new = x + self.step * r * (x_target - x)
        # x_new = x_target
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_Y:
            self.best_x = self.X[idx_individual, :].copy()
            self.best_y = self.Y[idx_individual].copy()

    def move(self, idx_individual):
        '''
        randomly move to a point

        :param idx_individual:
        :return:
        '''
        r = np.random.rand(self.n_dim)
        # r = np.round_(r1, decimals = 3)
        x_new = self.X[idx_individual, :] + self.visual * r
        self.X[idx_individual, :] = x_new
        self.Y[idx_individual] = self.func(x_new)
        if self.Y[idx_individual] < self.best_Y:
            self.best_X = self.X[idx_individual, :].copy()
            self.best_Y = self.Y[idx_individual].copy()

    def prey(self, idx_individual):
        '''
        prey
        :param idx_individual:
        :return:
        '''
        for try_num in range(self.max_try_num):
            r = np.random.rand(self.n_dim)
            # r = np.round_(r1, decimals = 3)
            x_target = self.X[idx_individual, :] + self.visual * r
            if self.func(x_target) < self.Y[idx_individual]:  # successful predation
                self.move_to_target(idx_individual, x_target)
                return None
        # If the prey is still unsuccessful after max_try_num times, the move operator is called
        self.move(idx_individual)

    def find_individual_in_vision(self, idx_individual):
        # Find all fish in line of sight of the fish idx_individual
        m = self.n_dim * self.cluster_size
        n = self.size_pop
        arr = self.X[[idx_individual], :].reshape(1, m)  #10(clusters)*18(features)=180
        arr2 = self.X.reshape(n, m)                     #(10, 180) (pup_size, )
        distances = spatial.distance.cdist(arr, arr2, metric='euclidean').reshape(-1)
        # print(distances)
        idx_individual_in_vision = np.argwhere((distances > 0) & (distances < self.visual))[:, 0]  #distances range between 24-29
        # idx_individual_in_vision = np.argwhere((distances > 0))[:, 0]
        return idx_individual_in_vision

    def swarm(self, idx_individual):
        # flocking behavior
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        # print(idx_individual_in_vision)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            center_individual_in_vision = individual_in_vision.mean(axis=0)
            center_y_in_vision = self.func(center_individual_in_vision)
            if center_y_in_vision * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
            # if center_y_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, center_individual_in_vision)
                return None
        self.prey(idx_individual)

    def follow(self, idx_individual):
        # rear-end behavior
        idx_individual_in_vision = self.find_individual_in_vision(idx_individual)
        num_idx_individual_in_vision = len(idx_individual_in_vision)
        if num_idx_individual_in_vision > 0:
            individual_in_vision = self.X[idx_individual_in_vision, :]
            y_in_vision = np.array([self.func(x) for x in individual_in_vision])
            idx_target = y_in_vision.argmin()
            x_target = individual_in_vision[idx_target]
            y_target = y_in_vision[idx_target]
            if y_target * num_idx_individual_in_vision < self.delta * self.Y[idx_individual]:
                self.move_to_target(idx_individual, x_target)
                return None
        self.prey(idx_individual)

    def run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for epoch in range(self.max_iter):
            for idx_individual in range(self.size_pop):
                self.swarm(idx_individual)
                self.follow(idx_individual)
            self.visual *= self.q
            # self.step *= self.q
        self.best_X, self.best_Y = self.best_x, self.best_y  # will be deprecated, use lowercase
        # return self.best_x, self.best_y
        return self.best_x