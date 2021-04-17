# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 01:07:46 2020

baseline_100 notes: 1- 10 days
                    
@author: mahmoud
"""


import numpy as np
import pandas as pd


DF = pd.read_csv(r'\April_Forecasted_CSV.csv')
DAY_TIME_INTERVAL = DF['Time code']  # 1:48
PV_OUTPUT = DF['PV_Output (kw) ']
LOAD_POWER = DF['Load (kw) ']
UNIT_PRICE = DF['System price (yen / kWhalfhour)']
# SELL_PRICE = DF['random PV price(yen / kWhalfhour)']


# T = 2928  # Time horizon of 2 months
t0 = 0
T = 1440    #  682468.60391453        April 2019
            # -132188.972661305       May 2019
            # -31592.989632205004     June 2019
            #  592710.39805628        July 2019
            #  727437.053807135       August 2019
            #  872730.044977415       September 2019
            #  481213.30972411        October 2019
            #  1395370.983756695      November 2019
            #  2343764.1974678        December 2019
            #  1796441.09197884       January 2020
            #  1173508.292695315      February 2020
            #  925597.82690339        March 2020

RESCALE_REWARD = 1

time_code_vector = DAY_TIME_INTERVAL[t0:T]
unit_price_vector = UNIT_PRICE[t0:T]
pv_output_vector = PV_OUTPUT[t0:T]
load_power_vector = LOAD_POWER[t0:T]

negative_costs = []
net_power = 0

for time_step in range(t0,T):
    net_power =  load_power_vector[time_step] - pv_output_vector[time_step]
    negative_cost = -unit_price_vector[time_step]*net_power * RESCALE_REWARD 
    
    negative_costs.append(negative_cost)    

negative_costs_sum = np.sum(negative_costs)

print(-negative_costs_sum)

def compare_with_theoritical_optimum():
    return -negative_costs_sum




