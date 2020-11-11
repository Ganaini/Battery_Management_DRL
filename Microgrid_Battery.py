# -*- coding: utf-8 -*-
"""
Created on Sun Nov  11 22:10:24 2020

v_125 notes: 1- Also calculate net_power after action--> after time_step incremental,
                but take care to keep loadpower,PVoutput the same.
             2- net_load shouuld depend on +ve delta_battery NOT -ve battery_charge
             3- After many tries, I think I should try different algorithm
             
             
@author: mahmoud
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces

# import random




# Default case (No control)
# all load energy bought from grid and all PV generation sold to grid 
#day_energy_cost = Lambda_n.dot(load) - lambda_p.dot(pv)
#print('Energy cost for a day = ',day_energy_cost,'$')

# Import half-hourly time series data for a whole year (2007)
DF = pd.read_csv(r'D:\presentations\Nov_5th\spot_2020.csv')
DAY_TIME_INTERVAL = np.array(DF['Time code'])  # 1:48
PV_OUTPUT = np.array(DF['PV_Output (kw) '])
LOAD_POWER = np.array(DF['Load (kw) '])
BUY_PRICE = np.array(DF['System price (yen / kWhalfhour)'])
SELL_PRICE = np.array(DF['random PV price(yen / kWhalfhour)'])


TIME_INTERVALS_A_DAY = 48
K = 11  # number of actions
T = 48   # Time horizon of one episode
SOC_MAX = 1.0
SOC_Min = 0.0
BATTERY_CAPACITY= 400 # kwh
MAX_CHARGE = 200 #kw
MAX_DISCHARGE = 200 #kw
SOC_INITIAL = 1
TOLERANCE = 1.05
RESCALE_REWARD = 0.001


#SOC_Flag = False

class Microgrid(gym.Env):
    """
    Description:
        A bettery, PV, dynamic load, connected to maingrid through PCC.
        We need to control battery charge/discharge scheduling to reduce cost.
        
    Source:
        mahmoud.abdallah09@gmail.com
        mahmoud-mohamed-tr@ynu.jp
        
    Observation(state):
        Type: Box(6)
        Num     Observation                  Min                     Max
        0       time code                     0              TIME_INTERVALS_A_DAY
        1       Selling Price(t)             -Inf                    Inf
        2       Buying Price (t)             -Inf                    Inf
        3       PV Generation Power(t)         0                     Inf
        4       Load Power(t)                  0                     Inf
        5       State of Charge (t)           0.0                    1.0 


    Actions:
        Type: Discrete(k), the range is [-max discharge, max charge]
        Num   Action
        0     battery power flow changes by (- dis_max) 
        1     battery power flow changes by (A[0]+1/(k-1)*(cha_max+dis_max))
        2     battery power flow changes by (A[1]+1/(k-1)*(cha_max+dis_max))
        3     battery power flow changes by (A[2]+1/(k-1)*(cha_max+dis_max))
        .
        .
        .
        k-1   battery power flow changes by (+ cha_max)  
        Note: we should have k output from out NN representing each action
        
    Reward:
        net_power: load_power - pv_output + delta_battery

        Cost_function(t): max(0, buy_price*net_power) - 
                          max(0, -sell_price*net_power)

        Objective_function: min⁡∑ Cost_function(t)
        
        Reward_function(t): - Cost_function(t) :
                            max(0, -sell_price*net_power) - 
                            max(0, buy_price*net_power)

    Starting State:
        State_of_charge starts with 0.0 value
        All other observations take their value from the CSV file data
        
    Episode Termination:
        when t == T

    """


    def __init__(self):
        
        # calculating total cost for a T period (T half-hours)
        self.time_code_vector = DAY_TIME_INTERVAL[:T]
        self.sell_price_vector = SELL_PRICE[:T]
        self.buy_price_vector = BUY_PRICE[:T]
        self.pv_output_vector = PV_OUTPUT[:T]
        self.load_power_vector = LOAD_POWER[:T]
              

        self.soc = SOC_INITIAL
        self.delta_battery = 0.0
        self.battery_charge = self.soc * BATTERY_CAPACITY
        self.net_power = (self.load_power_vector[0] - self.pv_output_vector[0] 
                          + self.delta_battery)
        self.time_step = 0 # time step  
        # self.random_hour = None
        
        
        high = np.array([TIME_INTERVALS_A_DAY,      # time_code range
                         np.finfo(np.float32).max,  # sell_price range
                         np.finfo(np.float32).max,  # buy_price range
                         np.finfo(np.float32).max,  # pv_output range
                         np.finfo(np.float32).max,  # load_power range
                         SOC_MAX],                  # soc range                 
                         dtype=np.float32)
        

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Discrete(K)

        self.state = (self.time_code_vector[0],self.sell_price_vector[0],
                      self.buy_price_vector[0], self.pv_output_vector[0],
                      self.load_power_vector[0], self.soc)
        
        self.state = np.array(self.state)

        # self.blocked_actions = []       
        
        

    def step(self, action):
        #SOC_Flag = False
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        #will only use self.soc
        time_code, sell_price, buy_price, pv_output, load_power, self.soc = self.state
     
        self.time_step += 1

        # action belongs to action space {0,1,2,....,K-1}
        # dis/charged power per step
        self.delta_battery =  0.5 * (- MAX_DISCHARGE + 
                                (action/(K-1) * (MAX_CHARGE+MAX_DISCHARGE))) # kw/half hour
        
        # soc change per t (half hour here). that's why multiplied by half
        self.battery_charge += self.delta_battery
        self.soc = self.battery_charge/BATTERY_CAPACITY
        self.net_power = (load_power - pv_output + self.delta_battery)       
        
        done = False
        if self.time_step >= T:
            done = True
         
        # self.random_hour = random.choice(range(48))    
        reward = 0
        if not done:
            self.state = (self.time_code_vector[self.time_step],
                          self.sell_price_vector[self.time_step], 
                          self.buy_price_vector[self.time_step], 
                          self.pv_output_vector[self.time_step],
                          self.load_power_vector[self.time_step], self.soc)
            
            # make sure action doesn't violate soc limits, two approaches
            # 1st : flag to main file, but too late after action is taken ==> wrong soln
            # if soc > SOC_MAX or soc < SOC_Min:
            #     SOC_Flag = True
            
            # 2nd: put huge penalty on those actions
            # if self.soc > SOC_MAX*TOLERANCE or self.soc < SOC_Min/TOLERANCE:
            #     reward = -5.0
            # else:
            
            # 3rd: make a list of actions that may lead to break soc limitations
            #      and send it back to main file before taking the action to 
            #      block them, is in new function "blocked_actions(action)"
            
            
        reward = (np.max([0,-sell_price*self.net_power]) - 
                  np.max([0,buy_price*self.net_power])) * RESCALE_REWARD 


        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = (self.time_code_vector[0], self.sell_price_vector[0],
                      self.buy_price_vector[0], self.pv_output_vector[0],
                      self.load_power_vector[0], SOC_INITIAL)
        self.time_step = 0
        self.soc = SOC_INITIAL
        self.delta_battery = 0.0
        self.battery_charge = self.soc * BATTERY_CAPACITY
        self.net_power = (self.load_power_vector[0] - self.pv_output_vector[0] 
                          + self.delta_battery)

        
        return np.array(self.state)

    # 3rd: make a list of actions that may lead to break soc limitations
    #      and send it back to main file before taking the action to 
    #      block them:
    def action_accepted(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        dummy_delta_battery = 0.5 * (- MAX_DISCHARGE + 
                                     (action/(K-1) * (MAX_CHARGE+MAX_DISCHARGE)))

        dummy_soc = self.soc + dummy_delta_battery/BATTERY_CAPACITY
        if dummy_soc > SOC_MAX*TOLERANCE or dummy_soc < SOC_Min/TOLERANCE:
            return False  # action rejected 
        
        return True # action is within soc limits



        
        