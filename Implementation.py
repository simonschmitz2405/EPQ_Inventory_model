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

author: Simon Schmitz
created: 17.05.2023
''' 


''' 
Parameter of manufacturer
'''
WM  = 27000 # selling price per unit
HM = 650.70 # inventory holding cost per unit per time
Hmi = 976.05 # inventory holding cost of raw materials per unit per time
AM = 6072575.64 # setup cost per lot per setup
CM = 5072.36 # material cost per unit
LM = 158399760 # labor cost per cycle
aM = 1837.62 # tool cost per unit
p = 56880 # production rate units per time
beta = 0.05 # expected percentage of defective items
rM = 2844 # screening rate units per time
sM = 156.12 # quality inspection cost per unit
KM = 1130.80 # cost of sales team per unit


''' 
Parameter of retailer
'''
WR = np.array([31500,31000,32000]) # selling price per unit
HR = np.array([207.12,187.26,172.27]) # inventory holding cost per unit per time
AR = np.array([1666.67,2000,2666.67]) # setup cost per setup
KR = np.array([366.95,507.32,464.39]) # cost of sales teams per unit per time
X = np.array([20712,18726,17227]) # expected value of market demand per unit per time
tau = np.array([6213,6554,6029]) # Scale parameter
m = 1 # elasticy parameter

# Number of Manufacturer and retailer
number_manufacturer = 1
number_retailer = 3



'''
Define the functions for the manufacturer profit
'''

# Demand of the retailer with exponential distribution
def demand_retailer(rho):
    DR = np.ones(number_retailer)
    for i in range(number_retailer):
        DR[i] = (X[i])+tau[i]*(1-(1/(1+rho)))
    return DR

# Income from sales (WMT)
def income_sales_manu(q):
    return WM*q.sum()

# Production cost for manufacturer (cp) 
def prod_cost_manu(q):
    return (((q.sum())/(1-beta))*(CM+(LM/p)+aM))

# Inventory holding cost for raw material (HM)
def holding_raw_manu(q,rho):
    return (((Hmi*(q.sum()**2))/(2*(1-beta)*p))-((Hmi*(demand_retailer(rho).sum())*(q.sum()**2))/(2*((1-beta)**2)*(p**2))))

#print('hold',holding_raw_manu(np.array([26889.17,6769.61,23221.23]),172.38))

# Inventory holding cost for good items (HP)
def holding_good_manu(q,rho):
    return HM/2*((q.sum()**2)/(demand_retailer(rho).sum())-(q.sum()**2)/((1-beta)*p))

# Inventory holding cost for defective times (HPD)
def holding_defec_manu(q):
    return HM*((beta*(q.sum()**2))/(2*((1-beta)**2)*p)+(beta*p*(q.sum()))/((1-beta)*rM))
    
# Cost for qualitiy inspections (SM)
def quality_inspec_manu(q):
    return (sM/(1-beta))*(q.sum())

# Setup cost (AMT)
def setup_manu(q):
    return AM*(q.sum())/((1-beta)*p)

# Cost of initiative of sales team (KMT)
def init_sales_manu(rho):
    return KM*psi*(rho**m)

# Total expected profit of the manufacturer (EAPM)
def EAPM(q,rho):
    return income_sales_manu(q)-prod_cost_manu(q)-holding_raw_manu(q,rho)-holding_good_manu(q,rho)- holding_defec_manu(q) -quality_inspec_manu(q)-setup_manu(q)-init_sales_manu(rho)


'''
Define the functions for the retailers profit
'''

# Income from sold units
def income_sales_retailer(rho):
    Income = np.ones(number_retailer)
    for i in range(number_retailer):
        Income[i] = WR[i]*demand_retailer(rho)[i]
    return Income

# Purchasing cost
def purch_cost_retailer(q):
    purch = np.ones(number_retailer)
    for i in range(number_retailer):
        purch[i] = WM*q[i]
    return purch

# Inventory holding cost for product at retailers
def holding_good_retailer(q,rho):
    HPR = np.ones(number_retailer)
    for i in range(number_retailer):
        HPR[i] = (HR[i]/2)*(((((demand_retailer(rho)[i])**2)*(q.sum()**2))/((X[i])*((demand_retailer(rho).sum())**2)))-((demand_retailer(rho)[i]*(q.sum()**2))/((demand_retailer(rho).sum())**2)))
    return HPR

# Marketing effort for retailer (Gamma can not be dollar unit) (Why with elasticy parameter)
def marketing_retailer(rho):
    Marke = np.ones(number_retailer)
    for i in range(number_retailer):
        Marke[i] = KR[i]*gamma[i]*(rho**m)
    return Marke

# Setup cost at the retailer
def setup_retailer(q,rho):
    Setup = np.ones(number_retailer)
    for i in range(number_retailer):
        Setup[i] = AR[i]*((demand_retailer(rho)[i]*(q.sum()))/((X[i])*demand_retailer(rho).sum()))
    return Setup

# Benefit of the retailer r
def benefit_retailer(q,rho):
    Benefit = np.ones(number_retailer)
    for i in range(number_retailer):
        Benefit[i] = income_sales_retailer(rho)[i] - purch_cost_retailer(q)[i]-holding_good_retailer(q,rho)[i]-marketing_retailer(rho)[i]-setup_retailer(q,rho)[i]
    return Benefit

# Expected average profit of retailers
def EAPR(q,rho):
    return benefit_retailer(q,rho).sum()

# Collaborating profit function
def EAPC(q,rho):
    return EAPM(q,rho)+EAPR(q,rho)



'''
Objective and constraints 
'''

# Objective with an input vector instead of two vectors
def objective_coll(x):
    q = x[:3]
    rho = x[3]
    return -EAPC(q,rho)

def objective_non_collaborative(x):
    q = x[:3]
    rho = x[3]
    return -EAPM(q, rho)

# Constraint function
def constraint(x):
    q = x[:3]
    return q.sum() - p

# Define the constraint
constraints = [{'type':'ineq','fun':constraint}]
initial_guess = np.array([10, 10, 10, 10])
bounds = [(0,None),(0,None),(0,None),(0,None)]
#options = {'maxiter':1000, 'maxfev':2000}

# Parameter for non-collaborative approach
gamma = np.array([1.,1.,1.])
psi = 1

'''
Give out the results
'''

# Give out the result of optimization
# result1 = minimize(objective_non_collaborative,initial_guess,constraints=constraints,bounds=bounds, tol=10e-2)

# np.set_printoptions(suppress= True)
# print('!!!The result of the objective_non_collaborative is: !!!\n',result1.x)

# print('EAPM: ',EAPM(result1.x[:3],result1.x[3]))
# print('EAPR: ',EAPR(result1.x[:3],result1.x[3]))
# print('EAPC: ',EAPC(result1.x[:3],result1.x[3]))

# Parameter for collaborative approach
gamma = np.array([0.17,0.23,0.2])
psi = 0.4 

result2 = minimize(objective_coll,initial_guess,method = 'slsqp',constraints=constraints, bounds=bounds, tol = 10e-3)

np.set_printoptions(suppress= True)
print('!!!The result of the objective_coll is: !!!\n',result2.x)
print('EAPM: ',EAPM(result2.x[:3],result2.x[3]))
print('EAPR: ',EAPR(result2.x[:3],result2.x[3]))
print('EAPC: ',EAPC(result2.x[:3],result2.x[3]))

print('EAPC according to Paper: ',EAPC(np.array([26889.17,6769.61,23221.23]),172.38))
