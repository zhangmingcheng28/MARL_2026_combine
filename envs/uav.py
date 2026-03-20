# -*- coding: utf-8 -*-
"""
@Time    : 19/3/2026 2:05 pm
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np


class UAV:
    def __init__(self, n_actions, agent_idx, gamma, tau, max_nei_num, maxSPD):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions  # this n_actions is the dimension of the action space
        self.agent_name = 'agent_%s' % agent_idx
        self.max_nei = max_nei_num
        #self.agent_size = 1.5  # meter in radius
        self.agent_grid_obs = None
        # self.max_grid_obs_dim = actor_obs[1]  # The 2nd element is the maximum grid observation dimension

        # state information
        self.pos = None
        self.ini_pos = None
        self.pre_pos = None
        self.vel = None
        self.pre_vel = None
        self.acc = np.zeros(2)
        self.pre_acc = np.zeros(2)
        self.maxSpeed = maxSPD
        self.goal = None
        self.waypoints = None
        self.ref_line = None
        self.ref_line_segments = None
        self.heading = None
        # self.detectionRange = 30  # in meters, this is the in diameter
        # self.detectionRange = 40  # in meters, this is the in diameter
        self.detectionRange = 30  # in meters, this is the in diameter
        # self.detectionRange = 100  # in meters, this is the in diameter, 100m, no convergence
        self.protectiveBound = 2.5  # diameter is 2.5*2, this is radius
        # self.protectiveBound = 1.5  # diameter is 2.5*2, this is radius
        # a dictionary, key is the agent idx, value is the array of 1x6,
        # which correspond to the observation vector of that neighbor
        self.pre_surroundingNeighbor = {}
        self.surroundingNeighbor = {}
        self.probe_line = {}
        self.observableSpace = []
        self.target_update_step = None
        self.removed_goal = None
        self.update_count = 0
        self.reach_target = False
        self.bound_collision = False
        self.building_collision = False
        self.drone_collision = False
        self.collide_wall_count = 0