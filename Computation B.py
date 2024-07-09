#!/usr/bin/env python
# coding: utf-8

# # Q2
# 
# ## Part 2

# In[242]:


import numpy as np

# Define the floating point constants
N_single = (2**24 - 1) * 2**128 / 2**24
N_single_test = N_single + 1.0

N_double = (2**53 - 1) * 2**1024 / 2**53
N_double_test = N_double + 1.0

# Print the results in a formatted manner
print("Floating Point Representation Analysis")
print("-------------------------------------")
print(f"N_single: {N_single}")
print(f"N_single_test N_single + 1.0: {N_single_test}")
print()
print(f"N_double: {N_double}")
print(f"N_double_test: N_double + 1.0: {N_double_test}")
print()
print("Observations:")
print("1. The value of N_single_test (N_single + 1.0) is the same as N_single, we expected otherwise.")
print("2. The value of N_double_test (N_double + 1.0) is the same as N_double, due to precision limitations")


# In[47]:


N_double-N_single**7 # The same!


# In[220]:


N_double -N_double/2 # Different!


# # Q4
# 
# ## Part 1

# In[214]:


def variance_1(sample):
    n = len(sample)
    mu = np.mean(sample)
    return sum((sample-mu)**2)/n


# In[225]:


def variance_2(sample):
    n = len(sample)
    mu = np.mean(sample)
    return np.mean(sample**2) - mu**2



# The python function:

# In[ ]:


np.var(np.array([10**9,10**9 + 1,10**9 + 2]))


# In[215]:


variance_1(np.array([10**9,10**9 + 1,10**9 + 2]))


# In[226]:


variance_2(np.array([10**9,10**9 + 1,10**9 + 2]))


# This works for 10^4!

# In[235]:


variance_2(np.array([10**4,10**4+ 1,10**4 + 2]))


# # Q5
# 
# ## Part 3

# In[21]:


import numpy as np
import random as random
import math
from scipy.stats import norm
import pandas as pd
import concurrent.futures


# In[29]:


def f1(x):
    return (x - 2)**9

def f2(x):
    return x**9 - 18*x**8 + 144*x**7 - 672*x**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512

def f2_pow(x):
    
    return (pow(x, 9) - 18 * pow(x, 8) + 144 * pow(x, 7) - 672 * pow(x, 6) +
            2016 * pow(x, 5) - 4032 * pow(x, 4) + 5376 * pow(x, 3) -
            4608 * pow(x, 2) + 2304 * x - 512)

def f2_numpy(x):
    return (np.power(x, 9) - 18 * np.power(x, 8) + 144 * np.power(x, 7) - 672 * np.power(x, 6) +
            2016 * np.power(x, 5) - 4032 * np.power(x, 4) + 5376 * np.power(x, 3) -
            4608 * np.power(x, 2) + 2304 * x - 512)

def f3(x):
    x9 = x * x * x * x * x * x * x * x * x
    x8 = x * x * x * x * x * x * x * x
    x7 = x * x * x * x * x * x * x
    x6 = x * x * x * x * x * x
    x5 = x * x * x * x * x
    x4 = x * x * x * x
    x3 = x * x * x
    x2 = x * x
  
    return(x9 - 18 * x8 + 144 * x7 - 672 * x6 + 2016 * x5 
            - 4032 * x4 +5376 * x3 - 4608 * x2 
           + 2304 * x - 512)


# In[171]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", "use_inf_as_na")
# Define the functions


# Generate the vector
vector = np.arange(1.92, 2.08, 0.001,)

# Create the plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=vector, y=f1(vector), label='f1(x)', color='blue')
sns.lineplot(x=vector, y=f2(vector), label='f2(x)', color='red')
sns.lineplot(x=vector, y=f3(vector), label='f3(x)', color='darkgreen')


# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of f1(x) and f2(x)')
plt.legend()

# Show the plot
plt.show()



# # Q6
# 
# 

# ## Part 2

# #### Python default:

# In[236]:


10**100 % np.pi

np.tan(10**100 % np.pi)


# #### Using the most popular high precision package:

# In[179]:


#Lets compute it accurately:

import mpmath

# Compute the remainder of 10^100 divided by pi
remainder = mpmath.fmod(mpmath.power(10, 100), mpmath.pi)

# Compute the tangent of the remainder
tan_value = mpmath.tan(remainder)

# Print the result
print(tan_value)


# #### Using Decimal, we get a different result:

# In[187]:


from decimal import Decimal, getcontext
import math

# Set precision high enough to handle large numbers
getcontext().prec = 110

# Define the large number and pi
large_number = Decimal(10**100)
pi_value = Decimal(math.pi)

# Compute the remainder of the large number divided by pi
remainder = large_number % pi_value

# Compute the tangent of the remainder using math.tan
tan_value = math.tan(float(remainder))

# Print the result
print(tan_value)


# #### Using sympy we get this result:

# In[188]:


import sympy as sp

# Define the large number and pi
large_number = 10**100
pi_value = sp.pi

# Compute the remainder of the large number divided by pi
remainder = large_number % pi_value

# Compute the tangent of the remainder
tan_value = sp.tan(remainder)

# Print the result
print(tan_value.evalf())


# #### Using gmpy2 we got:

# In[198]:


import gmpy2

# Set precision high enough to handle large numbers
gmpy2.get_context().precision = 113

# Define the large number and pi
large_number = gmpy2.mpz(10**100)
pi_value = gmpy2.const_pi()

# Compute the remainder of the large number divided by pi
remainder = gmpy2.fmod(large_number, pi_value)

# Compute the tangent of the remainder
tan_value = gmpy2.tan(remainder)

# Print the result
print(tan_value)


# ##### SO, which one is accurate? generally, mpmath is used way more frequently so we'll go with 0.929036363690337

# # Q7
# 
# ## Part 2

# Let's first plot the deltas:

# In[237]:


data = np.random.normal(loc=0, scale=0.001, size=100)  # 100 values with mean 100 and standard deviation 10

# Create an index for the x-axis
index = np.arange(len(data))

# Create a scatter plot
sns.scatterplot(x=index, y=data)


# In[167]:


def fun1(x):
    return np.sin(x)

def fun2(x):
    return np.cos(x)

def fun3(x):
    return -x

def fun4(x):
    return 1

def fun5(x):
    return (x-2)**9

def fun6(x):
    return 9 * (x-2)**8

def sim_minus_x(x, n):
    vec = []
    delta = np.random.normal(0, 2**(-23), n)
    for i in range(n):
        est = abs(fun3(x + delta[i]) - fun3(x)) / abs(delta[i])
        vec.append(est)
    return max(vec)

def true_cond_minus_x(x):
    return fun4(x)

def sim_x_power_9(x, n):
    vec = []
    delta = np.random.normal(0, 2**(-23), n)
    for i in range(n):
        est = abs(fun5(x + delta[i]) - fun5(x)) / abs(delta[i])
        vec.append(est)
    return max(vec)

def true_cond_x_power_9(x):
    return fun6(x)
# Example usage




# In[139]:


x1 = 0.5 
x2 = 0.0085
x3 = 50
n = 10000 


# In[168]:


# -x:

sim_cond_1 = sim_minus_x(x1, n)
true_condi_1 = true_cond_minus_x(x1)
sim_cond_2 = sim_minus_x(x2, n)
true_condi_2 = true_cond_minus_x(x2)
sim_cond_3 = sim_minus_x(x3, n)
true_condi_3 = true_cond_minus_x(x3)

#(x-2)^9


sim_cond_x1 = sim_x_power_9(x1, n)
true_cond_x1 = true_cond_x_power_9(x1)
sim_cond_x2 = sim_x_power_9(x2, n)
true_cond_x2 = true_cond_x_power_9(x2)
sim_cond_x3 = sim_x_power_9(x3, n)
true_cond_x3 = true_cond_x_power_9(x3)


# ## -x

# In[169]:


{'sim_cond1': sim_cond_1, 'true_condi1':true_condi_1 ,
 'sim_cond_2': sim_cond_2, 'true_condi_2': true_condi_2,
'sim_cond_3': sim_cond_3, 'true_condi_3': true_condi_3 }


# ## (X-2)^9

# In[170]:


{'sim_cond1': sim_cond_x1, 'true_condi1':true_cond_x1 ,'sim_cond_2': sim_cond_x2,
 'true_condi_2': true_cond_x2,
'sim_cond_3': sim_cond_x3, 'true_condi_3': true_cond_x3 }

