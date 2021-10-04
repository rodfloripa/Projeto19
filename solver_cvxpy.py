#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import cvxpy as cp
import numpy as np
from scipy.spatial import distance

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def solve_it(input_data):
    

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    facility_count = int(parts[0])
    customer_count = int(parts[1])
    
    facilities = []
    for i in range(1, facility_count+1):
        parts = lines[i].split()
        facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

    customers = []
    for i in range(facility_count+1, facility_count+1+customer_count):
        parts = lines[i].split()
        customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

    # Solution       
    capacity = [i.capacity for i in facilities]
    demand = [i.demand for i in customers]
    m = facility_count 
    n = customer_count 
    StateMatrix = cp.Variable((m,n),boolean = True)
    

    # calculate the setup cost of the solution
    # 'used' is a vector of 'm' facilities: if 0 not used,if 1 used
    used = cp.sum(StateMatrix,axis=1)
    obj = 0
    for i in range(0,m):
        obj = obj + facilities[i].setup_cost*used[i]  

    # distance matrix
    fac_pts = np.array([i.location for i in facilities])
    cust_pts = np.array([i.location for i in customers])
    print("Customers and Facilities:", n,m)
    # each 'm' column of dist_matrix has the distances from n customers to facility 'm'   
    dist_matrix = distance.cdist(cust_pts,fac_pts)
    # example: print distances to facility 0 : print(dist_matrix[:,0]) : dist_matrix = 50, 16
            
    
    # sum all the distances from facilities to customers 
    dist_sum = [] 
    for i in range(0,m):
        dist_sum.append(cp.sum(StateMatrix[i,:]*dist_matrix[:,i]))
    # sum the total distance from each facility
    dist_sum = cp.sum(dist_sum)

    # sum setup cost and distances
    objective = cp.Minimize( dist_sum + obj )
    # sum of columns==1 to attend all customers,sum of line demand of customers attached to facility 'm' <= 'm' facility capacity
    constraints = [  StateMatrix.T @ np.ones(m) == 1, StateMatrix @ demand  <= capacity*used ]
                 
    # solve the problem            
    prob = cp.Problem(objective, constraints)
    result = prob.solve(cp.XPRESS,verbose = False,parallel=True)
    print("Capacities " ,used.value*capacity)
    print("Used Capacity ",StateMatrix.value @ demand)
    print("State Matrix ",StateMatrix.value)
    
    # build  solution,print the vector of customers and facility index
    # Example [2,4,4] means that customers 0,1,2 are attached to facilities 2,4,4 
    state_mat = np.array(StateMatrix.value)
    arr = np.argwhere(state_mat==1)
    arr = list(arr)
    arr.sort(key=lambda x: x[1])
    solution = [0]*state_mat.shape[1]
 
    for i in range(0,len(solution)):
        solution[i] = arr[i][0]
    print(solution)
    obj = objective.value
    print(obj)


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_1)')

