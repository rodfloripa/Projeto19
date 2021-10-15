# Maximum running time is 40 minutes for big problems
# Install first cvxpy,Cbc( https://github.com/coin-or/Cbc ) and cbcpy

from collections import namedtuple
import math
import cvxpy as cp
import numpy as np
from numpy.random import default_rng
from scipy.spatial import distance
import warnings


print(cp.installed_solvers())
warnings.filterwarnings("ignore")
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

    # my solution
    cost =  [i.setup_cost for i in facilities]
    capacity = [i.capacity for i in facilities]
    demand = [i.demand for i in customers]
    m = facility_count 
    n = customer_count 
    

    fac_pts = np.array([i.location for i in facilities])
    cust_pts = np.array([i.location for i in customers])

    # divide problem into 4 quadrants
    minx = np.min([i[0] for i in cust_pts])
    maxx = np.max([i[0] for i in cust_pts])
    miny = np.min([i[1] for i in cust_pts])
    maxy = np.max([i[1] for i in cust_pts])

    # [xmin,xmax],[ymin,ymax]
    q1 = [[minx,minx+0.5*(maxx-minx)],[miny,miny+0.5*(maxy-miny)]]
    q2 = [[minx,minx+0.5*(maxx-minx)],[miny+0.5*(maxy-miny),maxy]]
    q3 = [[minx+0.5*(maxx-minx),maxx],[miny+0.5*(maxy-miny),maxy]]
    q4 = [[minx+0.5*(maxx-minx),maxx],[miny,miny+0.5*(maxy-miny)]]

    # divide customer points into 4 quadrants
    cust_pts_i = []
    for i in range(0,len(cust_pts)):
        cust_pts_i.append(np.array([i,cust_pts[i]]))
    cust_pts_q1 = []
    cust_pts_q2 = []
    cust_pts_q3 = []
    cust_pts_q4 = []
    cx1,cx2,cx3,cx4 = [],[],[],[]
    for i in range(0,len(cust_pts_i)):
        if cust_pts_i[i][1][0] >= q1[0][0] and cust_pts_i[i][1][0] <= q1[0][1] and \
        cust_pts_i[i][1][1] >= q1[1][0] and cust_pts_i[i][1][1] <= q1[1][1]:
            cust_pts_q1.append(cust_pts_i[i][1])
            cx1.append(cust_pts_i[i][0])
        if cust_pts_i[i][1][0] >= q2[0][0] and cust_pts_i[i][1][0] <= q2[0][1] and \
        cust_pts_i[i][1][1] > q2[1][0] and cust_pts_i[i][1][1] <= q2[1][1]:
            cust_pts_q2.append(cust_pts_i[i][1])
            cx2.append(cust_pts_i[i][0])
        if cust_pts_i[i][1][0] > q3[0][0] and cust_pts_i[i][1][0] <= q3[0][1] and \
        cust_pts_i[i][1][1] > q3[1][0] and cust_pts_i[i][1][1] <= q3[1][1]:
            cust_pts_q3.append(cust_pts_i[i][1])
            cx3.append(cust_pts_i[i][0])
        if cust_pts_i[i][1][0] > q4[0][0] and cust_pts_i[i][1][0] <= q4[0][1] and \
        cust_pts_i[i][1][1] >= q4[1][0] and cust_pts_i[i][1][1] <= q4[1][1]:
            cust_pts_q4.append(cust_pts_i[i][1])
            cx4.append(cust_pts_i[i][0])

    # Customers-Conversion from new index to original
    #cx[new ] =  original
    cx = cx1+cx2+cx3+cx4

    # divide facility points into 4 quadrants
    fac_pts_i = []
    for i in range(0,len(fac_pts)):
        fac_pts_i.append(np.array([i,fac_pts[i]]))
    fac_pts_q1 = []
    fac_pts_q2 = []
    fac_pts_q3 = []
    fac_pts_q4 = []
    fx1,fx2,fx3,fx4 = [],[],[],[]
    for i in range(0,len(fac_pts_i)):
        if fac_pts_i[i][1][0] >= q1[0][0] and fac_pts_i[i][1][0] <= q1[0][1] and \
        fac_pts_i[i][1][1] >= q1[1][0] and fac_pts_i[i][1][1] <= q1[1][1]:
            fac_pts_q1.append(fac_pts_i[i][1])
            fx1.append(fac_pts_i[i][0])
        if fac_pts_i[i][1][0] >= q2[0][0] and fac_pts_i[i][1][0] <= q2[0][1] and \
        fac_pts_i[i][1][1] > q2[1][0] and fac_pts_i[i][1][1] <= q2[1][1]:
            fac_pts_q2.append(fac_pts_i[i][1])
            fx2.append(fac_pts_i[i][0])
        if fac_pts_i[i][1][0] > q3[0][0] and fac_pts_i[i][1][0] <= q3[0][1] and \
        fac_pts_i[i][1][1] > q3[1][0] and fac_pts_i[i][1][1] <= q3[1][1]:
            fac_pts_q3.append(fac_pts_i[i][1])
            fx3.append(fac_pts_i[i][0])
        if fac_pts_i[i][1][0] > q4[0][0] and fac_pts_i[i][1][0] <= q4[0][1] and \
            fac_pts_i[i][1][1] >= q4[1][0] and fac_pts_i[i][1][1] <= q4[1][1]:
            fac_pts_q4.append(fac_pts_i[i][1])
            fx4.append(fac_pts_i[i][0])

    # Facilities-Conversion from new index to original
    #fx[new ] =  original
    fx = fx1+fx2+fx3+fx4


    # Return total cost and dict[customer]:facility for each quadrant
    def calc(cust_pts_q1,fac_pts_q1,cx1,fx1):
        
        capacity,cost = [],[]
        for i in fx1:
            capacity.append(facilities[i].capacity)
            cost.append(facilities[i].setup_cost)
        demand = []
        for i in cx1:
            demand.append(customers[i].demand)

        m = len(fac_pts_q1)
        n = len(cust_pts_q1)
        # StateMatrix gives the solution       
        StateMatrix = cp.Variable((m,n),boolean = True)
        # 'used' is a vector of 'm' facilities: if 0 not used,if 1 used
        used = cp.Variable(m,boolean = True)


        # calculate the setup cost of the solution
        # 'z' is a vector that constrained with 'used' will give me the list of used facilities
        z = cp.sum(StateMatrix,axis=1)
        obj = 0
        for i in range(0,m):
            obj = obj + facilities[fx1[i]].setup_cost*used[i]  
        # distance matrix
        print("Customers and Facilities:", n,m)
        # each 'm' column of dist_matrix has the distances from n customers to facility 'm'   
        dist_matrix = distance.cdist(cust_pts_q1,fac_pts_q1)
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
        constraints = [  StateMatrix.T @ np.ones(m) == 1, StateMatrix @ demand  <= capacity,
                    z <= 1000*used , z >= -1000*used ]

        # solve the problem            
        prob = cp.Problem(objective, constraints)
        result = prob.solve(cp.CBC,verbose = False,maximumSeconds=600)
        #print("Capacities " ,used.value*capacity)
        #print("Used Capacity ",StateMatrix.value @ demand)
        
        # build  solution
        # For example, solution = [2,4,4] means that customers 0,1,2 are attached to facilities 2,4,4 
        state_mat = np.array(StateMatrix.value)
        arr = np.argwhere(state_mat>0.5)
        arr = list(arr)
        arr.sort(key=lambda x: x[1])
        solution = [0]*state_mat.shape[1]

        for i in range(0,len(solution)):
            solution[i] = arr[i][0]
        obj1 = objective.value
        f_sol = {} 
        # transform from new index to original
        for i in range(0,len(solution)):
            f_sol[cx1[i]] = fx1[solution[i]]    

        return(obj1,f_sol)

    # if one quadrant is empty don't optimize for each quadrant
    if len(cust_pts_q1) > 0 and len(cust_pts_q2) > 0 and len(cust_pts_q3) > 0 and \
       len(cust_pts_q4) > 0 and len(fac_pts_q1) > 0 and len(fac_pts_q2) > 0 and \
       len(fac_pts_q3) > 0 and len(fac_pts_q4) > 0:
        obj1,dic1 = calc(cust_pts_q1,fac_pts_q1,cx1,fx1)
        obj2,dic2 = calc(cust_pts_q2,fac_pts_q2,cx2,fx2)
        obj3,dic3 = calc(cust_pts_q3,fac_pts_q3,cx3,fx3)
        obj4,dic4 = calc(cust_pts_q4,fac_pts_q4,cx4,fx4)
        obj = obj1+obj2+obj3+obj4
        dic = {**dic1, **dic2, **dic3, **dic4}
    else:
        cx1 = [i for i in range(0,n)]
        fx1 = [i for i in range(0,m)]
        obj,dic = calc(cust_pts,fac_pts,cx1,fx1)
    solution = []
    for i in range(0,len(dic)):
        solution.append(dic[i])


    #prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))
    return output_data

import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')
