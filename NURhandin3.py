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

x_range = np.linspace(0.01,5,10000)
y_range = n_integrand(x_range)
plt.plot(x_range,y_range)
plt.show()

x_maximum = golden_ratio(n_integrand,0.01,5,1e-3)
N_maximum = n_integrand(x_maximum)
print(f'The maximum value of N, {N_maximum}, is found at x = {x_maximum}!')

#Problem 1b

def trap_loweropen(f,a,b,N): #eval. at semi open interval (a,b]
    x_values = np.linspace(a,b,N+1)[1:]
    y_values = f(x_values)
    h = (b-a)/N #step size 
    if N > 1:
        y0_exterp = (f(x_values[1])-f(x_values[0]))/(x_values[1]-x_values[0])*\
            x_values[0] + f(x_values[0])
        return 0.5*h*(y0_exterp + y_values[-1]+2*np.sum(y_values[0:N-1]))
    return 0.5*h*(y_values[-1]+2*np.sum(y_values[0:N-1]))

def romberg_loweropen(f,a,b,m): #input: function f, start a, stop b, order m
    h = (b-a) #stepsize
    r = np.zeros(m) #array of initial guesses
    r[0] = trap_loweropen(f,a,b,1) #using N=1 for initial guess  
    N_p = 1
    for i in range(1,m-1): #range from 1 to m-1 
        r[i] = 0 #set the other initial estimates to 0 
        delta = h
        h = 0.5*h #reduce stepsize 
        x = a+h
        for j in range(N_p): #0 to N_p 
            r[i] = r[i] + f(x)
            x = x + delta 
        r[i] = 0.5*(r[i-1]+delta*r[i]) #new estimate of r[i]
        N_p = 2*N_p #increase N_p as i increases
    N_p = 1 #reset N_p to 1
    for i in range(1,m-1):
        N_p = 4*N_p #increase N_p (very fast) for increasing i
        for j in range(0,m-i):
            r[j] = (N_p * r[j+1] - r[j])/(N_p - 1) #new estimate 
    return r[0] #final value: all estimates merged into a final result at r[0]

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

Nsat_m15 = len(radius_m15)/nhalo_m15
print(Nsat_m15)

bins = np.logspace(np.log10(0.01),np.log10(5),10)

Nsat_binned_m15 = np.histogram(radius_m15,bins=bins)[0]/nhalo_m15
print(Nsat_binned_m15)
print(A)






