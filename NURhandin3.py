import numpy as np
import matplotlib.pyplot as plt

#Problem 1A

def bracketing(f,a,b):
    w = (1+np.sqrt(5))*0.5 #compute golden ratio phi and set w = phi
    if f(b) > f(a):
        a,b = b,a
    c = b + (b-a)*w
    while f(c) > f(b): #use > sign to find a maximum of f, < to find a minimum
        #first, fit a parabola and set d as the abscissa of the minimum
        r = (b-a)*(f(b)-f(c))
        q = (b-c)*(f(b)-f(c))
        d = b - (0.5*(b-c)*q-(b-a)*r)/((q-r))
        if d > b and d < c:
            if f(d) > f(c):
                return [b,d,c]
            elif f(d) < f(b):
                return [a,b,d]
            else:
                d = c + (c-b)*w
        elif np.abs(d-b) > 100*np.abs(c-b):
            d = c + (c-b)*w
        a = b
        b = c 
        c = d
    return [a,b,c]

def golden_ratio(f,a,b,err):
    a,b,c = bracketing(f,a,b)
    phi = (1+np.sqrt(5))*0.5
    w = 2-phi
    iterations = 0
    while np.abs(c-a) > err:
        if np.abs(c-b) > np.abs(b-a):
            d = b + (c-b)*w
        else:
            d = b + (a-b)*w
        if f(d) > f(b): #use > sign to find a maximum of f, < to find a minimum
            if d > b and d < c:
                a = b
                b = d
            else:
                c = b
                b = d 
        else:
            if d > b and d < c:
                c = d
            else:
                a = d 
        iterations += 1 
    print(f'the algorithm took {iterations} iterations!')
    if f(d) < f(b):
        return d
    else:
        return b

A = 0.2*256/(np.pi**(1.5))
Nsat=100
a=2.4
b=0.25
c=1.6

def n(x,A,Nsat,a,b,c): #first use the function with A=1
    return A*Nsat*((x/b)**(a-3))*np.exp(-(x/b)**c)

def n_integrand(x):
    return 4*np.pi*(x**2)*n(x,A,Nsat,a,b,c)

x_maximum = golden_ratio(n_integrand,0.01,5,1e-3)
N_maximum = n_integrand(x_maximum)
print('Problem 1a:')
print(f'The maximum value of N, {N_maximum}, is found at x = {x_maximum}!')
print('Problem 1b:')

#Problem 1b

"""
First, introduce the integration algorithms we will need. I simply use
trapezoids as they are very fast and the accuracy does not have to be amazing 
"""

def trapezoid(f,a,b,N): #choose a simple trapezoid because its very fast 
    x_values = np.linspace(a,b,N+1)
    y_values = f(x_values)
    h = (b-a)/N #step size
    return 0.5*h*(y_values[0]+y_values[-1]+2*np.sum(y_values[1:N]))

def trap_loweropen(f,a,b,N): #eval at semi open interval (a,b] using lin exterp
    x_values = np.linspace(a,b,N+1)[1:]
    y_values = f(x_values)
    h = (b-a)/N #step size 
    if N > 1: #linearly exterpolate for x = a to find an include a y-value
        y0_exterp = (f(x_values[1])-f(x_values[0]))/(x_values[1]-x_values[0])*\
            x_values[0] + f(x_values[0])
        return 0.5*h*(y0_exterp + y_values[-1]+2*np.sum(y_values[0:N-1]))
    return 0.5*h*(y_values[-1]+2*np.sum(y_values[0:N-1]))

"""
We use a sorting algorithm that we use in step 1 of the downhill simplex.
I chose selection sort because it is relatively easy to expand to N dimensions.
"""

def selection_sort(array): #1 dimensional case of selection sort
    N = len(array)
    for i in range(N-1):
        i_min = i
        for j in range(i+1,N):
            if array[j] < array[i_min]:
                i_min = j
        if i_min != i:
            array[i_min],array[i] = array[i],array[i_min]
    return array

def selection_sort_Ndim(array,x_vector): #selection sort for N dimensions
    N = len(array)
    for i in range(N-1):
        i_min = i
        for j in range(i+1,N):
            if array[j] < array[i_min]:
                i_min = j
        if i_min != i:
            #sort the y_values
            array[i_min],array[i] = array[i],array[i_min]
            #sort the N-1 dimensional x vector
            x_vector[i_min],x_vector[i] = \
                np.copy(x_vector[i]),np.copy(x_vector[i_min])
    return array,x_vector

"""
To minimize a function, we use the N-dimensional downhill simplex algorithm
"""

def downhill_simplex(f,err,*args): #N-dimensional downhill simplex method   
    f_values = f(*args)
    print('f',f_values)
    x_vector = np.dstack((args))[0]
    N = len(f_values)-1
    print('the x vector is',*x_vector[-1])
    print('f(x_vector[-1]) is', f(*x_vector[-1]))
    
    fractional_old = 0 
    iterations = 0
    while iterations < 50: #choose some arbitrary # of iters as a maximum
        
        f_values,x_vector = selection_sort_Ndim(f_values,x_vector)
        centroid = 1/N * np.sum(x_vector[:-1],axis=0)
        
        fractional = 2*(f(*x_vector[-1]) - f(*x_vector[0])) / \
            (f(*x_vector[-1]) + f(*x_vector[0]))
        if np.abs(fractional-fractional_old) < err:
            print(f'the algorithm took {iterations} iterations!')
            return x_vector[0]
        
        x_try = 2*centroid - x_vector[-1]
        
        if f(*x_vector[0]) <= f(*x_try) < f(*x_vector[-1]): #case 1
            x_vector[-1] = np.copy(x_try)
            
        elif f(*x_try) < f(*x_vector[0]): #case 2 
            x_exp = 2*x_try - centroid
            if f(*x_exp) < f(*x_try):
                x_vector[-1] = np.copy(x_exp)
            else: 
                x_vector[-1] = np.copy(x_try)
                
        elif f(*x_try) >= f(*x_vector[-1]): #case 3
            x_try = 0.5*(centroid+x_vector[-1])
            if f(*x_try) < f(*x_vector[-1]):
                x_vector[-1] = np.copy(x_try)
        else: #case 4
            x_vector = 0.5*(x_vector[0] + x_vector[:0])
        
        fractional_old = fractional
        iterations += 1
    print(f'the algorithm exceeded {iterations} iterations!')
    return x_vector[0] #terminate incase do not reach target acc within iters!

"""
Now, we read in the five different files 
"""

def readfile(filename):
    f = open(filename, 'r')
    data = f.readlines()[3:] #Skip first 3 lines 
    nhalo = int(data[0]) #number of halos
    radius = []
    
    for line in data[1:]:
        if line[:-1]!='#':
            radius.append(float(line.split()[0]))
    
    radius = np.array(radius, dtype=float)    
    f.close()
    return radius, nhalo #Return the virial radius for all the satellites in
                            #the file, and the number of halos
                            
#radius_m11, nhalo_m11 = readfile('satgals_m11.txt')
#radius_m12, nhalo_m12 = readfile('satgals_m12.txt')
#radius_m13, nhalo_m13 = readfile('satgals_m13.txt')
#radius_m14, nhalo_m14 = readfile('satgals_m14.txt')
radius_m15, nhalo_m15 = readfile('satgals_m15.txt')

bins = np.logspace(np.log10(0.01),np.log10(5),10)

a_values = np.linspace(2.4,2.9,100) #try these a,b,c values to find best fit!
b_values = np.linspace(0.2,0.3,100)
c_values = np.linspace(1,2,100)

Nsat_m15 = len(radius_m15)/nhalo_m15
weight = np.full(len(radius_m15),Nsat_m15)

Nsat_m15_weighted, bin_edges_weighted = np.histogram(radius_m15,bins=bins,\
                                                     weights=weight)

def Nmodel(radius_min,radius_max,Nsat,a,b,c):
    """
    Proposed model for N(x) using various input parameters. 
    
    """
    
    Nsat_int = Nsat
    a_int = a
    b_int = b 
    c_int = c
    lower_bound = radius_min
    upper_bound = radius_max
    
    def n_findA(x):
        return n(x,1,Nsat_int,a_int,b_int,c_int)
    
    A_int = Nsat_int/trap_loweropen(n_findA,0,5,100)
    
    def integrand(x):
        return n(x,A_int,Nsat_int,a_int,b_int,c_int)
    N_model = trap_loweropen(integrand,lower_bound,upper_bound,100)
    
    return N_model

def Nmodel_minimize(a,b,c):
    return Nmodel(min(radius_m15),max(radius_m15),Nsat_m15,a,b,c)

#print('Nmodel simplex',downhill_simplex(Nmodel_minimize,1e-5,a_values,b_values\
#                                        ,c_values)) 

def chi2(x,Nsat,a,b,c,bin_edges,Nhist):
    
    N_model = []
    for i in range(len(bin_edges)-1):
        N_model.append(Nmodel(bin_edges[i],bin_edges[i+1],Nsat,a,b,c))
         
    chi2 = ((Nhist[:,None] - N_model)**2)/N_model
    
    if type(a) == np.float64:
        #print('a is number',np.sum(chi2))
        return np.sum(chi2)
    else:
        #print('a is list',np.sum(chi2,axis=1))
        return np.sum(chi2,axis=1)
    
    #return np.sum(chi2,axis=0)
    
#print('chi2',chi2(radius_m15,Nsat,a,b,c,bin_edges_weighted,Nsat_m15_weighted))

def chi2_minimize(a,b,c):
    return chi2(radius_m15,Nsat,a,b,c,bin_edges_weighted,Nsat_m15_weighted)

minimized_params = downhill_simplex(chi2_minimize,1e-6,a_values,b_values,\
                                    c_values)
print(minimized_params)

N_model = []
for i in range(len(bin_edges_weighted)-1):
    N_model.append(Nmodel(bin_edges_weighted[i],bin_edges_weighted[i+1],\
                          Nsat_m15,*minimized_params))
        
print('model',Nmodel(min(radius_m15),max(radius_m15),Nsat_m15,\
                              *minimized_params))

print('data binned',Nsat_m15_weighted)

def n_function(x,a,b,c):
    Nsat = Nmodel(min(x),max(x),Nsat_m15,a,b,c)
    return 4*np.pi*(x**2)*n(x,A,Nsat,a,b,c)

fitted_m15 = n_function(radius_m15,*minimized_params)

init_m15 = n_function(radius_m15,a,b,c)

plt.hist(radius_m15,bins=10,edgecolor='black',density=True)
plt.scatter(radius_m15,fitted_m15/max(fitted_m15),s=2,zorder=100,color='red')
plt.scatter(radius_m15,init_m15/max(init_m15),s=2,zorder=100,color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radii x')
plt.ylabel('Normalised N(x)')
plt.show()

def n_integrand_min(x):
    Nsat = Nmodel(min(x),max(x),Nsat_m15,*minimized_params)
    return 4*np.pi*(x**2)*n(x,A,Nsat,*minimized_params)

print('Nsat is',trapezoid(n_integrand_min,0.01,5,100))

print('chi2 value',np.abs(chi2_minimize(*minimized_params)))

#Problem 1c

print('Problem 1c:')

def ln_factorial(k): #compute the natural log of k! for an integer k 
    value = 0.0 #if k=0, we return 0.0 as 0! = 1 (and ln(1) = 0)
    for i in range(1,k+1): #k+1 is not included
        value += np.float32(np.log(i)) #sum values ln(1)+ln(2)+..+ln(k)
    return value #returns sum of all values of ln(i) within [1,k]

import math

def gamma_func(n): #simple gamma function implementation, not too accurate
    def integral(x):
        return x**(n-1)*np.exp(-x)
    return trapezoid(integral,0.01,100,1000)

def log_likelihood_poisson(xdata,y,mu,bin_edges,a,b,c):
    
    N_model = []
    for i in range(len(bin_edges)-1):
        N_model.append(Nmodel(bin_edges[i],bin_edges[i+1],Nsat_m15,a,b,c))
        
    factorial_array = np.zeros(len(xdata))
    for i in range(len(xdata)):
        factorial_array[i] = math.gamma(xdata[i]+1)
        
    log_poisson = y*np.log(mu) - mu - factorial_array
    
    #print(log_poisson)
    
    return -np.sum(log_poisson[:-1])

MLE = log_likelihood_poisson(radius_m15,radius_m15,Nsat_m15,\
                             bin_edges_weighted,a,b,c)
print('MLE:',MLE)

def LL_poisson_min(a,b,c):
    return log_likelihood_poisson(radius_m15,radius_m15,Nsat_m15,\
                                  bin_edges_weighted,a,b,c)

#print(downhill_simplex(LL_poisson_min,1e-6,a_values,b_values,c_values))

"""
print(Nsat_m15_weighted)
print('n weighted',Nsat_m15_weighted)
print('logaritmic bin edges weighted',bin_edges_weighted)

testing = Nmodel(bin_edges_weighted[0],bin_edges_weighted[1],A,Nsat_m15,a,b,c)
testing2 = Nmodel(bin_edges_weighted[1],bin_edges_weighted[2],A,Nsat_m15,a,b,c)
#print('integral value',testing)
#print('int val',testing2)
"""

#Problem 1d

from scipy.special import gammainc

def G_test(observed,expected):
    return 2*np.sum(observed*np.log(observed/expected))

def Q(G,M,x):
    k = len(x) - M #compute the degrees of freedom
    cdf = gammainc(0.5*k,0.5*x)/math.gamma(0.5*k)

print(G_test(init_m15,fitted_m15))

print(Q(G_test(init_m15,fitted_m15),3,radius_m15))





