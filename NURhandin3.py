import numpy as np
import matplotlib.pyplot as plt

#Problem 1A

def bracketing(f,a,b): #bracketing algorithm to find initial a,b,c 
    w = (1+np.sqrt(5))*0.5 #compute golden ratio phi and set w = phi
    if f(b) > f(a):
        a,b = b,a #swap a and b if f(b) is above f(a)
    c = b + (b-a)*w
    while f(c) > f(b): #use > sign to find a maximum of f, < to find a minimum
        #first, fit a parabola and set d as the abscissa of the minimum
        r = (b-a)*(f(b)-f(c))
        q = (b-c)*(f(b)-f(c))
        d = b - (0.5*(b-c)*q-(b-a)*r)/((q-r)) #fit parabola & compute abscissa
        if d > b and d < c: #d between b and c
            if f(d) > f(c): 
                return [b,d,c]
            elif f(d) < f(b):
                return [a,b,d]
            else:
                d = c + (c-b)*w
        elif np.abs(d-b) > 100*np.abs(c-b): #d very far past c 
            d = c + (c-b)*w
        a = b #shift the values 
        b = c 
        c = d
    return [a,b,c]

def golden_ratio(f,a,b,err): #uses golden ratio algorithm to find maximum
    a,b,c = bracketing(f,a,b) #initial guesses for a,b,c 
    phi = (1+np.sqrt(5))*0.5 
    w = 2-phi
    iterations = 0 #count the iterations
    while np.abs(c-a) > err: #keep going if target accuracy is not reached
        if np.abs(c-b) > np.abs(b-a):
            d = b + (c-b)*w
        else:
            d = b + (a-b)*w
        if f(d) > f(b): #use > sign to find a maximum of f, < to find a minimum
            if d > b and d < c: #d in between b and c 
                a = b
                b = d
            else: #d in between a and b 
                c = b
                b = d 
        else:
            if d > b and d < c:
                c = d
            else:
                a = d 
        iterations += 1 
    print(f'the algorithm took {iterations} iterations!')
    if f(d) < f(b): #check if either d or b describes the minimum value 
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

x_maximum = golden_ratio(n_integrand,0.01,5,1e-3) #append GR-algo to find max
N_maximum = n_integrand(x_maximum) #find N(x = x_maximum)
print('Problem 1a:')
print(f'The maximum value of N, {N_maximum}, is found at x = {x_maximum}!')

output = x_maximum,N_maximum
np.savetxt('NURhandin3problem1a.txt',output,fmt='%f')

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
        if i_min != i: #swap elements if new element is lower than minimum elem
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
    f_values = f(*args) #make an array of f_values 
    x_vector = np.dstack((args))[0] #make an array of x_vectors 
    N = len(f_values)-1 #find the amount of points
    
    """
    Used these while debugging:
    print('f',f_values) 
    print('the x vector is',*x_vector[-1])
    print('f(x_vector[-1]) is', f(*x_vector[-1]))
    """
    
    fractional_old = 0 #use this value to compare the fractional!
    iterations = 0 #count the iterations
    while iterations < 50: #choose some arbitrary # of iters as a maximum
        
        f_values,x_vector = selection_sort_Ndim(f_values,x_vector) #sort f(x)
        centroid = 1/N * np.sum(x_vector[:-1],axis=0) #compute the centroid 
        
        fractional = 2*(f(*x_vector[-1]) - f(*x_vector[0])) / \
            (f(*x_vector[-1]) + f(*x_vector[0]))
        if np.abs(fractional-fractional_old) < err:
            print(f'the algorithm took {iterations} iterations!')
            return x_vector[0] #return the best estimate 
        
        x_try = 2*centroid - x_vector[-1]
        
        #Below are the 4 different cases from the slides
        
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
                            
radius_m11, nhalo_m11 = readfile('satgals_m11.txt')
radius_m12, nhalo_m12 = readfile('satgals_m12.txt')
radius_m13, nhalo_m13 = readfile('satgals_m13.txt')
radius_m14, nhalo_m14 = readfile('satgals_m14.txt')
radius_m15, nhalo_m15 = readfile('satgals_m15.txt')

bins = np.logspace(np.log10(0.01),np.log10(5),10) #create the radii bins 

a_values = np.linspace(2.4,2.9,100) #try these a,b,c values to find best fit!
b_values = np.linspace(0.2,0.3,100)
c_values = np.linspace(1,2,100)

"""
Below, the weighted <Nsat> values are computed per bin per data set 
"""

Nsat_m11 = len(radius_m11)/nhalo_m11
weight_m11 = np.full(len(radius_m11),Nsat_m11)

Nsat_m12 = len(radius_m12)/nhalo_m12
weight_m12 = np.full(len(radius_m12),Nsat_m12)

Nsat_m13 = len(radius_m13)/nhalo_m13
weight_m13 = np.full(len(radius_m13),Nsat_m13)

print(Nsat_m11,Nsat_m12,Nsat_m13)

Nsat_m11_weighted, binedges_weighted_m11 = np.histogram(radius_m11,bins=bins,\
                                                     weights=weight_m11)

Nsat_m12_weighted, binedges_weighted_m12 = np.histogram(radius_m12,bins=bins,\
                                                     weights=weight_m12)

Nsat_m13_weighted, binedges_weighted_m13 = np.histogram(radius_m13,bins=bins,\
                                                     weights=weight_m13)

Nsat_m14 = len(radius_m14)/nhalo_m14
weight_m14 = np.full(len(radius_m14),Nsat_m14)

Nsat_m15 = len(radius_m15)/nhalo_m15
weight_m15 = np.full(len(radius_m15),Nsat_m15)


print(Nsat_m14,Nsat_m15)

Nsat_m14_weighted, binedges_weighted_m14 = np.histogram(radius_m14,bins=bins,\
                                                     weights=weight_m14)

Nsat_m15_weighted, binedges_weighted_m15 = np.histogram(radius_m15,bins=bins,\
                                                     weights=weight_m15)
    
"""
Next, a series of functions follow which we use to find the minimum chi^2 value
At the end, we find the minimum parameters for each data file. They are all the
same and the chi^2 values are negative, so something goes wrong unfortunately.

"""

def Nmodel(radius_min,radius_max,Nsat,a,b,c):
    """
    Proposed model for N(x) using various input parameters. 
    
    """
    
    Nsat_int = Nsat
    a_int = a
    b_int = b 
    c_int = c
    lower_bound = radius_min #lower bin edge
    upper_bound = radius_max #upper bin edge 
    
    def n_findA(x): #integrate for A=1 to find the correct norm. constant A
        return n(x,1,Nsat_int,a_int,b_int,c_int)
    
    A_int = Nsat_int/trap_loweropen(n_findA,0,5,100)
    
    def integrand(x): #integrate 4pi x^2 n(x) for correct A value 
        return n(x,A_int,Nsat_int,a_int,b_int,c_int)
    N_model = trap_loweropen(integrand,lower_bound,upper_bound,100)
    
    return N_model

def chi2(x,Nsat,a,b,c,bin_edges,Nhist): #computes chi squared (wrong!)
    
    N_model = [] #gets the model estimate N_i per bin 
    for i in range(len(bin_edges)-1):
        N_model.append(Nmodel(bin_edges[i],bin_edges[i+1],Nsat,a,b,c))
    
    chi2 = ((Nhist[:,None] - N_model)**2)/N_model #compute the chi2 
    
    if type(a) == np.float64:
        #print('a is number',np.sum(chi2))
        return np.sum(chi2) #output chi2 statistic 
    else:
        #print('a is list',np.sum(chi2,axis=1))
        return np.sum(chi2,axis=1) #output chi2 statistic 

def chi2_minimize_m11(a,b,c):
    return chi2(radius_m11,Nsat,a,b,c,binedges_weighted_m11,Nsat_m11_weighted)
    
def chi2_minimize_m12(a,b,c):
    return chi2(radius_m12,Nsat,a,b,c,binedges_weighted_m12,Nsat_m12_weighted)

def chi2_minimize_m13(a,b,c):
    return chi2(radius_m13,Nsat,a,b,c,binedges_weighted_m13,Nsat_m13_weighted)

def chi2_minimize_m14(a,b,c):
    return chi2(radius_m14,Nsat,a,b,c,binedges_weighted_m14,Nsat_m14_weighted)

def chi2_minimize_m15(a,b,c):
    return chi2(radius_m15,Nsat,a,b,c,binedges_weighted_m15,Nsat_m15_weighted)

min_params_m11 = downhill_simplex(chi2_minimize_m11,1e-6,a_values,b_values,\
                                    c_values)
    
min_params_m12 = downhill_simplex(chi2_minimize_m12,1e-6,a_values,b_values,\
                                    c_values)

min_params_m13 = downhill_simplex(chi2_minimize_m13,1e-6,a_values,b_values,\
                                    c_values)  
    
print('the best-fit for m11 a,b,c are',min_params_m11)
print('the best-fit for m12 a,b,c are',min_params_m12)
print('the best-fit for m13 a,b,c are',min_params_m13)


min_params_m14 = downhill_simplex(chi2_minimize_m14,1e-6,a_values,b_values,\
                                    c_values)
    
min_params_m15 = downhill_simplex(chi2_minimize_m15,1e-6,a_values,b_values,\
                                    c_values)
    
print('the best-fit for m14 a,b,c are',min_params_m14)
print('the best-fit for m15 a,b,c are',min_params_m15)

N_model = []
for i in range(len(binedges_weighted_m15)-1):
    N_model.append(Nmodel(binedges_weighted_m15[i],binedges_weighted_m15[i+1],\
                          Nsat_m15,*min_params_m15))

def n_function(x,a,b,c): #n(x) as an additional function of a,b,c fit params
    Nsat = Nmodel(min(x),max(x),Nsat_m15,a,b,c)
    return 4*np.pi*(x**2)*n(x,A,Nsat,a,b,c)

"""
Below we plot the fitted N_i against the data histogram bins.

"""

fitted_m11 = n_function(radius_m11,*min_params_m11)
init_m11 = n_function(radius_m11,a,b,c)

plt.hist(radius_m11,bins=10,edgecolor='black',density=True,label='data hist')
plt.scatter(radius_m11,fitted_m11/max(fitted_m11),s=2,zorder=100,color='red',\
            label='best-fit profile')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radii x')
plt.ylabel('Normalised N(x)')
plt.legend()
plt.savefig('./1b_figure1.png')
plt.close()

fitted_m12 = n_function(radius_m12,*min_params_m12)
init_m12 = n_function(radius_m12,a,b,c)

plt.hist(radius_m12,bins=10,edgecolor='black',density=True,label='data hist')
plt.scatter(radius_m12,fitted_m12/max(fitted_m12),s=2,zorder=100,color='red',\
            label='best-fit profile')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radii x')
plt.ylabel('Normalised N(x)')
plt.legend()
plt.savefig('./1b_figure2.png')
plt.close()

fitted_m13 = n_function(radius_m13,*min_params_m13)
init_m13 = n_function(radius_m13,a,b,c)

plt.hist(radius_m13,bins=10,edgecolor='black',density=True,label='data hist')
plt.scatter(radius_m13,fitted_m13/max(fitted_m13),s=2,zorder=100,color='red',\
            label='best-fit profile')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radii x')
plt.ylabel('Normalised N(x)')
plt.legend()
plt.savefig('./1b_figure3.png')
plt.close()

fitted_m14 = n_function(radius_m14,*min_params_m14)
init_m14 = n_function(radius_m14,a,b,c)

plt.hist(radius_m14,bins=10,edgecolor='black',density=True,label='data hist')
plt.scatter(radius_m14,fitted_m14/max(fitted_m14),s=2,zorder=100,color='red',\
            label='best-fit profile')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radii x')
plt.ylabel('Normalised N(x)')
plt.legend()
plt.savefig('./1b_figure4.png')
plt.close()

fitted_m15 = n_function(radius_m15,*min_params_m15)
init_m15 = n_function(radius_m15,a,b,c)

plt.hist(radius_m15,bins=10,edgecolor='black',density=True,label='data hist')
plt.scatter(radius_m15,fitted_m15/max(fitted_m15),s=2,zorder=100,color='red',\
            label='best-fit profile')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Radii x')
plt.ylabel('Normalised N(x)')
plt.legend()
plt.savefig('./1b_figure5.png')
plt.close()

output2 = Nsat_m11,*min_params_m11,np.around(np.abs(chi2_minimize_m11(*min_params_m11)),3)
output3 = Nsat_m12,*min_params_m12,np.abs(chi2_minimize_m12(*min_params_m12))
output4 = Nsat_m13,*min_params_m13,np.abs(chi2_minimize_m13(*min_params_m13))
output5 = Nsat_m14,*min_params_m14,np.abs(chi2_minimize_m14(*min_params_m14))
output6 = Nsat_m15,*min_params_m15,np.abs(chi2_minimize_m15(*min_params_m15))

print('output of m_11',output2)

print('output of m_12',output3)

print('output of m_13',output4)

print('output of m_14',output5)

print('output of m_15',output6)

np.savetxt('NURhandin3problem1b.txt',[output2,output3,output4,\
                                      output5,output6],fmt='%f')
    

#Problem 1c

print('Problem 1c:')

def ln_factorial(k): #compute the natural log of k! for an integer k 
    value = 0.0 #if k=0, we return 0.0 as 0! = 1 (and ln(1) = 0)
    for i in range(1,k+1): #k+1 is not included
        value += np.float32(np.log(i)) #sum values ln(1)+ln(2)+..+ln(k)
    return value #returns sum of all values of ln(i) within [1,k]

import math #for the gamma function

def gamma_func(n): #simple gamma function implementation, unused
    def integral(x):
        return x**(n-1)*np.exp(-x)
    return trapezoid(integral,0.01,100,1000)#use library instead as its allowed

def log_likelihood_poisson(y,Nhalo,bin_edges,a,b,c): #log likelihood of poisson
    
    N_model = [] #fill modelled N(x) per bin 
    for i in range(len(bin_edges)-1):
        N_model.append(Nmodel(bin_edges[i],bin_edges[i+1],Nhalo,a,b,c))
        
    factorial_array = np.zeros(len(y)) #compute factorial of each radius value
    for i in range(len(y)):
        factorial_array[i] = math.gamma(y[i]+1)
    """
    I tried the following during debugging to differentiate between cases
    where you use just one input vector (a,b,c) vs a range of a,b,c values:
    
    print(type(a))
        
    if type(a) == float:
        print('a is number')
        print(N_model)
        log_poisson = y*np.log(N_model[4]) - N_model[4] - factorial_array
        return -np.sum(log_poisson[:-1])
    else:
        print('a is list')
        print(np.shape(N_model))
        N_model = N_model[0]
        print(np.shape(N_model))
        #model_mean = np.sum(N_model,axis=1)
        log_poisson = y[:,None]*np.log(N_model[4]) - N_model[4] - factorial_array[:,None]
        return -np.sum(log_poisson[:-1],axis=1)
    """
     
    log_poisson = y*np.log(N_model[4]) - N_model[4] - factorial_array
    
    
    return -np.sum(log_poisson[:-1])

MLE_m11 = log_likelihood_poisson(radius_m11,Nsat_m11,\
                             binedges_weighted_m11,a,b,c)
MLE_m12 = log_likelihood_poisson(radius_m12,Nsat_m12,\
                             binedges_weighted_m12,a,b,c)
MLE_m13 = log_likelihood_poisson(radius_m13,Nsat_m13,\
                             binedges_weighted_m13,a,b,c)
MLE_m14 = log_likelihood_poisson(radius_m14,Nsat_m14,\
                             binedges_weighted_m14,a,b,c)
MLE_m15 = log_likelihood_poisson(radius_m15,Nsat_m15,\
                             binedges_weighted_m15,a,b,c)
   
print('the value of MLE for m11 is',MLE_m11)
print('the value of MLE for m12 is',MLE_m12)
print('the value of MLE for m13 is',MLE_m13)
print('the value of MLE for m14 is',MLE_m14)
print('the value of MLE for m15 is',MLE_m15)

def LL_poisson_min(a,b,c):
    return log_likelihood_poisson(radius_m15,Nsat_m15,\
                                  binedges_weighted_m15,a,b,c)

#trying to minimize log_likelihood_poisson, this does not run!

#print(downhill_simplex(LL_poisson_min,1e-6,a_values,b_values,c_values))

MLE_values = [] #testing what MLE values we would get
for i in range(len(a_values)):
    MLE_values.append(log_likelihood_poisson(radius_m15,Nsat_m15,\
                                 binedges_weighted_m15,a_values[i],b_values[i]\
                                     ,c_values[i]))
        
"""
Trying some values, I can see that the MLE is monotonically increasing, 
so something is wrong.
"""

#Problem 1d

print('Problem 1d:')

from scipy.special import gammainc #import incomplete gamma function

def G_test(observed,expected): #G-test formula implementation
    return 2*np.sum(observed*np.log(observed/expected))

def Q(G,M): #Q significance
    k = len(bins) - M #compute the degrees of freedom, 10 bins and 3 params, 
    cdf = gammainc(0.5*k,G)/math.gamma(0.5*k)
    return 1-cdf

G_array = np.array([G_test(init_m11,fitted_m11),G_test(init_m12,fitted_m12),\
                    G_test(init_m13,fitted_m13),G_test(init_m14,fitted_m14),\
                        G_test(init_m15,fitted_m15)])
Q_array = []
Q_array.append(Q(G_array[0],3))
Q_array.append(Q(G_array[1],3))
Q_array.append(Q(G_array[2],3))
Q_array.append(Q(G_array[3],3))
Q_array.append(Q(G_array[4],3))

print(G_array)
print(Q_array)

np.savetxt('NURhandin3problem1d.txt',[G_array,Q_array],fmt='%f')





