# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:54:37 2020

1- use pandas to save delta batteries
2- if deltabattery > 0:     --> 2
   elif deltabattery == 0:  --> 1
   else:                    --> 0    

@author: mahmo
"""

from baseline_103 import compare_with_theoritical_optimum
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint as nlc
from functools import partial
import matplotlib.pyplot as plt

import pandas as pd

DF = pd.read_csv(r'D:\presentations\March_25_Forecast\spot_2019.csv')
PV_OUTPUT = np.array(DF['PV_Output (kw) '])
LOAD_POWER = np.array(DF['Load (kw) '])
UNIT_PRICE = np.array(DF['System price (yen / kWhalfhour)'])

# BATTERY_LOCE = 200  # $/KWH
BATTERY_LOCE = 300  # $/KWH ADB report Table IV
Battery_Capacity = 2800 #kwh
MAX_CHARGE = 200 #kh/half
MAX_DISCHARGE = -200 # kw/ho
# T = 1440
# T = 2928    # 48*61   2months
t0 = 0
T = 48
SOC_MIN = 0.0
SOC_MAX = 1.0
SOC_INIT = 0.00


class Theoritical_opt(object):
    def __init__(self, T=T, t0=t0, battery_capacity=Battery_Capacity,lcoe=BATTERY_LOCE,  
                 max_charge=MAX_CHARGE, max_discharge=MAX_DISCHARGE, soc_min=SOC_MIN,
                 soc_max=SOC_MAX, soc_init=SOC_INIT, optimization_type='linear', 
                 linear_solver='interior-point', non_linear_solver='SLSQP'):
        
        self.T = T
        self.t0= t0
        self.battery_capacity = battery_capacity
        self.lcoe = lcoe
        self.max_charge = max_charge
        self.max_discharge = max_discharge
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc_init = soc_init
        self.optimization_type = optimization_type
        self.soc=[]
        self.cost = 0
        self.battery_cost_per_month = 0
        self.total_cost = 0
        self.saving = 0
        if optimization_type == 'linear':
            self.soc = self.linear_optimizer(solver=linear_solver)
        elif optimization_type == 'non-linear':
            self.soc = self.non_linear_optimizer(solver=non_linear_solver)
        self.delta_battery() 



    # linear optimization
    def linear_optimizer(self, solver):
        # obj coefficients    
        c = []

        # appending all coeffectionts except for last soc variable coefficient
        for t in range(self.t0, self.T-1):
            c.append(self.battery_capacity*(UNIT_PRICE[t] - UNIT_PRICE[t+1]))

        # append last coefficient    
        c.append(self.battery_capacity*(UNIT_PRICE[self.T-1]))

        # constraints coefficients 
        A = np.zeros((self.T - self.t0 ,self.T - self.t0 ))  

        # append first coefficient 
        A[0][0] = 1

        # appending all coeffectionts except for first soc variable coefficient
        for i in range(1,self.T - self.t0):
            A[i][i] = 1
            A[i][i-1] = -1

        # constraints upper boundaries  
        bu = [] 

        # append first upper bound 
        bu.append((self.max_charge + self.battery_capacity * self.soc_init) / self.battery_capacity)

        # appending all  upper bounds
        for i in range(1,self.T - self.t0 ):
            bu.append(self.max_charge / self.battery_capacity)
 
        # double A to add both <= and >= constraints   but make the 2nd one -ve      
        A = np.concatenate((A, -A), axis=0)                                   

        # constraints lower boundaries  
        bl = [] 

        # append first lower bound 
        bl.append((-self.max_discharge -self.battery_capacity * self.soc_init) / self.battery_capacity)

        # appending all  lower bounds
        for i in range(1,self.T - self.t0 ):
            bl.append(-self.max_discharge / self.battery_capacity)
    
        # add bu and bl together (cause we treated bl as bu here to follow the optimizer)
        b = np.concatenate((bu, bl), axis=0) 

        # bounds
        b1 = (self.soc_min, self.soc_max)
        bnds = []
        for t in range(self.T - self.t0):
            bnds.append(b1)

        # methods: 'interior-point', 'revised simplex', 'simplex' 
        res = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method =solver)   

        print('\n Capacity:\n',self.battery_capacity)    
        print('\n fun:\n',res.fun)
        print('\n Success?:\n',res.success)

        # add the constants of cost functions that were not included in obj fun
        self.cost = res.fun + UNIT_PRICE[0] * (LOAD_POWER[0] 
                                          - self.battery_capacity * self.soc_init - PV_OUTPUT[0])
        for t in range(1,self.T - self.t0):
            self.cost += UNIT_PRICE[t] * (LOAD_POWER[t] - PV_OUTPUT[t])

        print('\n Cost without considering battery cost:\n',self.cost) 
        
        # adding battery LoCE/T for 2 months (30/15)
        self.battery_cost_per_month = self.lcoe * self.battery_capacity / 30
        print('\n Battery cost per month:\n', self.battery_cost_per_month)
        self.total_cost = self.cost + self.battery_cost_per_month
        print('\n Total cost:\n',self.total_cost)
        soc = res.x.round(3)
        
        print('\n')

        print('soc = ' + str(soc))
        print('\n')
        self.saving = (100-self.total_cost / compare_with_theoritical_optimum()*100)

        # compare with baseline
        print('Theoretical optimum saves ',self.saving,'% of baseline cost'  )        
        
        return soc
    
    def delta_battery(self):
        # calculate delta_battery from soc
        delta_battery = []

        delta_battery.append((self.soc[0] - self.soc_init) * self.battery_capacity)
        for t in range(1, self.T - self.t0):
            delta_battery.append((self.soc[t]-self.soc[t-1]) * self.battery_capacity)
        print('\n Delta Battery', delta_battery)
        
        # add actions to train on its labels
        actions = []  
        for delta in delta_battery:
            if delta > 0:
                actions.append(2)
            elif delta == 0:
                actions.append(1)                
            else:
                actions.append(0)
                
        # save delta_battery to csv file
        delta_actions_df = pd.DataFrame({'Delta_Battery': delta_battery,
                                         'Discreate_Actions': actions})
        delta_actions_df.to_csv(index=True)
        delta_actions_df.to_csv('D:\presentations\March_25_Forecast\delta_actions_A_day.csv')
  

    def nl_objective(self, soc):
        total_cost = 0
        total_cost =+ (UNIT_PRICE[0] * (LOAD_POWER[0] + 
                    (soc[0]-self.soc_init)*self.battery_capacity - PV_OUTPUT[0]))
        for t in range(1, self.T - self.t0):
            total_cost += (UNIT_PRICE[t] * (LOAD_POWER[t] + 
                            (soc[t]-soc[t-1])*self.battery_capacity - PV_OUTPUT[t]))
        return total_cost
    
    def nl_constraint(self, soc, index):
        return (soc[index] - soc[index-1]) * self.battery_capacity 
    
    
    def non_linear_optimizer(self, solver):
        soc0 = []    
        for t in range(self.T - self.t0):
            soc0.append(self.soc_init)

        print('\n')
        # show initial objective
        print('Initial Objective: ' + str(self.nl_objective(soc0)))
        print('\n')

        # constraints
        cons = []
        cons.append(nlc(fun= lambda x:(x[0]-self.soc_init)*self.battery_capacity,
                    lb=self.max_discharge,ub=self.max_charge))

        for t in range(1, self.T - self.t0):
            cons.append(nlc(fun=partial(self.nl_constraint, index=t),
                    lb=self.max_discharge,ub=self.max_charge))

        # bounds
        b1 = (self.soc_min, self.soc_max)
        bnds = []
        for t in range(self.T - self.t0):
            bnds.append(b1)

        # optimize
        # only 'trust-constr', 'SLSQP' methods works with the current set-up    
        solution = minimize(self.nl_objective,soc0,method=solver,
                            bounds=bnds, constraints=cons)

        # print('solution: \n', solution)
        # print('\n')
        soc = solution.x.round(2) 
        print('Final Objective: ' + str(self.nl_objective(self.soc)))
        print('\n')
        # print solution
        print('Solution')
        print('\n')
        print('soc = ' + str(self.soc))
        print('\n')        
        return soc
    

if __name__ == "__main__":
    
    # compare with different capacities
    n_capacities = []
    n_c_rate = []    
    n_cost_reduction = []
    n_lcoe = []
    best_tubles = []
    for lcoe in range(300, 301, 150):
        # print("\nLCoE: ", lcoe)
        capacities = []
        cost_reduction = []
        c_rate = []
        for c in range(2800,2801,400):
            print("\nLCoE: ", lcoe)
            optim = Theoritical_opt(T=T, t0=0, battery_capacity=c, lcoe=lcoe,  
                 max_charge=MAX_CHARGE, max_discharge=MAX_DISCHARGE, soc_min=SOC_MIN,
                 soc_max=SOC_MAX, soc_init=SOC_INIT, optimization_type='linear', 
                 linear_solver='interior-point')
        
            capacities.append(c)
            c_rate.append(c/(2*optim.max_charge))
            cost_reduction.append(optim.saving)

        # print best c_rate and its savings for all lcoe
        dic = {}
        for i in range(len(c_rate)):
            dic[c_rate[i]] = cost_reduction[i]

        for k,v in dic.items():
            if v == max(dic.values()):
                # print('Best C rate for ' + str(lcoe) + ' lcoe is ' + str(k) + 
                #       ' with savings ' + str(v) + ' %') 
                best_tubles.append((lcoe, k, v))
                
        n_capacities.append(capacities)
        n_c_rate.append(c_rate)
        n_cost_reduction.append(cost_reduction)
        n_lcoe.append(lcoe)

    plt.plot(n_c_rate[0], n_cost_reduction[0], label='lcoe:' + str(n_lcoe[0]))
    # plt.plot(n_c_rate[1], n_cost_reduction[1], label='lcoe:' + str(n_lcoe[1]))
    # plt.plot(n_c_rate[2], n_cost_reduction[2], label='lcoe:' + str(n_lcoe[2]))
    # plt.plot(n_c_rate[3], n_cost_reduction[3], label='lcoe:' + str(n_lcoe[3]))
    # plt.plot(n_c_rate[4], n_cost_reduction[4], label='lcoe:' + str(n_lcoe[4]))
    # plt.plot(n_c_rate[5], n_cost_reduction[5], label='lcoe:' + str(n_lcoe[5]))
    # plt.plot(n_c_rate[6], n_cost_reduction[6], label='lcoe:' + str(n_lcoe[6]))
    # plt.plot(n_c_rate[7], n_cost_reduction[7], label='lcoe:' + str(n_lcoe[7]))
    # plt.plot(n_c_rate[8], n_cost_reduction[8], label='lcoe:' + str(n_lcoe[8]))
    # plt.plot(n_c_rate[9], n_cost_reduction[9], label='lcoe:' + str(n_lcoe[9]))    
    plt.ylabel('cost saving per month(%)')
    plt.xlabel('C rate')
    plt.legend()
    plt.title('Theoretical optimum Cost using linear solver for 2 months')
    plt.show()
    
    print("best C rate for all LCoE: ", best_tubles)

    # # plotting, comment it for high T
    # time_slots = [ s for s in range(optim.T)]
    # plt.plot(time_slots, PV_OUTPUT[:T], label='PV Output')
    # plt.plot(time_slots, LOAD_POWER[:T], label='Load')
    # plt.plot(time_slots, UNIT_PRICE[:T] * 100, label='Price * 100')
    # plt.plot(time_slots, delta_battery, label='Delta Battery')
    # plt.plot(time_slots, optim.soc * optim.battery_capacity, label='soc * battery_capacity')
    # plt.ylabel('Power (KW)')
    # plt.xlabel('Time Slot (half hourly)')
    # plt.legend()
    # plt.title('Theoretical optimum Cost using SLSQP solver for a day')
    # plt.show()


