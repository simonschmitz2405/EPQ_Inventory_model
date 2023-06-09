import numpy as np
from scipy.optimize import minimize

'''
This is the implementation of:
"An EPQ inventory model
considering an imperfect
production system with
probabilistic demand and
collaborative approach" 
from Katherinne Salas-Navarro

This is the improved version
Changed: HP, HPD, AMT, CL, restriccion 1 y restriccion 2
Added: Times

author: Simon Schmitz
created: 26.06.2023
''' 


''' 
Parameter of manufacturer
'''
WM  = 27000 # selling price per unit
HM = 650.70 # inventory holding cost per unit per time
Hmi = 976.05 # inventory holding cost of raw materials per unit per time
AM = 6072575.64 # setup cost per lot per time
CM = 5072.36 # material cost per unit
LM = 158399760 # labor cost per time
aM = 1837.62 # tool cost per unit
p = 56880 # production rate units per time
beta = 0.05 # expected percentage of defective items
rM = 2844 # screening rate units per time
sM = 156.12 # quality inspection cost per unit
KM = 1130.80 # cost of sales team per unit


''' 
Parameter of retailer
'''
WR = 31500# selling price per unit
HR = 207.12 # inventory holding cost per unit per time
AR = 1666.67 # setup cost per setup
KR = 366.95# cost of sales teams per unit per time
X = 20712 # expected value of market demand per unit per time
tau = 6213 # Scale parameter
m = 1 # elasticy parameter

# Number of Manufacturer and retailer
number_manufacturer = 1
number_retailer = 1

# Define the times
def TR(q,rho):
    return (demand_retailer(rho)*q)/(X*demand_retailer(rho))

def TM(q,rho):
    return q/demand_retailer(rho)

def TPM(q):
    return q/((1-beta)*p)

'''
Define the functions for the manufacturer profit
'''

# Demand of the retailer with exponential distribution (DR)
def demand_retailer(rho):
    DR = X+tau*(1-(1/(1+rho)))
    return DR

# Inventory holding cost for raw material (HM)
def holding_raw_manu(q,rho):
    demand_retailer_value = demand_retailer(rho)
    HM = (Hmi/2)*(((q**2)/((1-beta)*p))-((demand_retailer_value*(q**2))/((1-beta)**2*p**2)))
    return HM


# Inventory holding cost for good items (HP)
def holding_good_manu(q,rho):
    HP = (HM/2)*((q**2)/demand_retailer(rho)-(q**2)/((1-beta)*p)+2*(q**2)*(1/demand_retailer(rho)-(1/((1-beta)*p))))
    return HP

# Inventory holding cost for defective times (HPD)
def holding_defec_manu(q):
    HPD = (HM/2)*(beta*(1/rM)*(q**2)/(((1-beta)**2)*p))
    return HPD

# Cost for qualitiy inspections (SM)
def quality_inspec_manu(q):
    return (sM/(1-beta))*(q)

# Setup cost (AMT)
def setup_manu(q):
    AMT = AM*(q)/((1-beta)*p)
    return AMT

# Production cost for manufacturer (cp) 
def prod_cost_manu(q):
    cp = q/((1-beta)*p)*(CM*p+LM+aM*p)
    return cp

# Cost of initiative of sales team (KMT)
def init_sales_manu(rho):
    KMT = KM*psi*(rho**m)
    return KMT

# Income from sales (WMT)
def income_sales_manu(q):
    return WM*q



# Total expected profit of the manufacturer (EAPM)
def EAPM(q,rho):
    return income_sales_manu(q)-prod_cost_manu(q)-holding_raw_manu(q,rho)-holding_good_manu(q,rho)- holding_defec_manu(q) -quality_inspec_manu(q)-setup_manu(q)-init_sales_manu(rho)

'''
Define the functions for the retailers profit
'''

# Inventory holding cost for product at retailers (HPR)
def holding_good_retailer(q,rho):
    HPR = (HR/2)*((q**2)/demand_retailer(rho)+2*(q**2)*(demand_retailer(rho)-X)-((q**2)/X))
    return HPR

# Marketing effort for retailer (EM)
def marketing_retailer(rho):
    EM = KR*gamma*(rho**m)
    return EM

# Setup cost at the retailer (CL)
def setup_retailer(q,rho):
    CL = AR * (q/X)
    return CL

# Purchasing cost (CC)
def purch_cost_retailer(q):
    CC = WM*q
    return CC

# Income from sold units
def income_sales_retailer(rho):
    Income = WR*demand_retailer(rho)
    return Income


# Benefit of the retailer r
def benefit_retailer(q,rho):
    Benefit = income_sales_retailer(rho) - purch_cost_retailer(q)-holding_good_retailer(q,rho)-marketing_retailer(rho)-setup_retailer(q,rho)
    return Benefit

# Expected average profit of retailers
def EAPR(q,rho):
    return benefit_retailer(q,rho)

# Collaborating profit function
def EAPC(q,rho):
    return EAPM(q,rho)+EAPR(q,rho)


'''
Objective and constraints 
'''

# Objective with an input vector instead of two vectors
def objective_coll(x):
    q = x[0]
    rho = x[-1]
    return -EAPC(q,rho)

def objective_non_collaborative(x):
    q = x[0]
    rho = x[-1]
    return -EAPM(q, rho)

# Restricction for optimization
cons = [{'type': 'ineq', 'fun': lambda x:  p*TPM(x[0]) - x[0]},
        {'type':'ineq','fun':lambda x: x[0] - demand_retailer(x[-1])*TM(x[0],x[-1])},
        {'type': 'ineq', 'fun': lambda x:  EAPM(x[0],x[-1])},
        {'type': 'ineq', 'fun': lambda x:  EAPR(x[0],x[-1])}]


initial_guess = np.array([1, 1])
bounds = ((0,None),(0,None))
options = {'maxiter':10000, 'disp':True,'ftol':10e-2}

# Parameter for non-collaborative approach
gamma = 0
psi = 1

'''
Give out the results
'''

# Give out the result of optimization
result1 = minimize(objective_non_collaborative,initial_guess,method='slsqp',constraints=cons,options=options,bounds=bounds)

np.set_printoptions(suppress= True)
print('!!!The result of the objective_non_collaborative is: !!!\n',result1.x)

print(result1)

print('EAPM: ',EAPM(result1.x[0],result1.x[-1]))
print('EAPR: ',EAPR(result1.x[0],result1.x[-1]))
print('EAPC: ',EAPC(result1.x[0],result1.x[-1]))

print('income_sales_manu',income_sales_manu(result1.x[0]))
print('prod_cost_manu',prod_cost_manu(result1.x[0]))
print('holding_raw_manu',holding_raw_manu(result1.x[0],result1.x[-1]))
print('holding_good_manu',holding_good_manu(result1.x[0],result1.x[-1]))
print('holding_defec_manu',holding_defec_manu(result1.x[0]))
print('quality_inspec_manu',quality_inspec_manu(result1.x[0]))
print('setup_manu',setup_manu(result1.x[0]))
print('init_sales_manu',init_sales_manu(result1.x[-1]))
print('demand_retailer',demand_retailer(result1.x[-1]))

# Parameter for collaborative approach
gamma = 0.6
psi = 0.4 

result2 = minimize(objective_coll, initial_guess,method = 'slsqp',constraints=cons,bounds=bounds,options=options)

np.set_printoptions(suppress= True)
print('!!!The result of the objective_coll is: !!!\n',result2.x)
print('EAPM: ',EAPM(result2.x[0],result2.x[-1]))
print('EAPR: ',EAPR(result2.x[0],result2.x[-1]))
print('EAPC: ',EAPC(result2.x[0],result2.x[-1]))

print(result2)

print('income_sales_manu',income_sales_manu(result2.x[0]))
print('prod_cost_manu',prod_cost_manu(result2.x[0]))
print('holding_raw_manu',holding_raw_manu(result2.x[0],result2.x[-1]))
print('holding_good_manu',holding_good_manu(result2.x[0],result2.x[-1]))
print('holding_defec_manu',holding_defec_manu(result2.x[0]))
print('quality_inspec_manu',quality_inspec_manu(result2.x[0]))
print('setup_manu',setup_manu(result2.x[0]))
print('init_sales_manu',init_sales_manu(result2.x[-1]))

print('income_sales_retailer',income_sales_retailer(result2.x[-1]))
print('purch_cost_retailer',purch_cost_retailer(result2.x[0]))
print('holding_good_retailer',holding_good_retailer(result2.x[0],result2.x[-1]))
print('marketing_retailer',marketing_retailer(result2.x[-1]))
print('setup_retailer',setup_retailer(result2.x[0],result2.x[-1]))
