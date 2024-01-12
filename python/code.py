import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

####################################################################
######## Fixed Parameters & Useful vectors #########################
N=20                               # Num of shells 
nu=10**-6                          # viscosity 
dt=10**-5                          # integration step 
a,b,c=(1.0,-0.50,-0.50)                
(k,ek,forcing)=([],[],[])
for n in range(N): 
    k.append(2**n)
    ek.append(np.exp(-nu*dt*k[n]*k[n]/2.0))
    if n==0 :
        forcing.append(1+1j)
Tmax=100.0
time=np.arange(0,int(Tmax/dt))
####################################################################
####################################################################

def G(u) :
    # Since the velocities u(n) must be zero when n<0 and n>Num-1, 
    # the non-linear coupling G[u,u] must respect the boundary condition u(-1)=u(-2)=u(Num)=u(Num+1)=0

    coupling=np.zeros(N,dtype=complex)

    for n in range(0,N): 
        if n==0 : 
            coupling[n]=(a*k[n+1]*np.conj(u[n+1])*u[n+2])*1j                               #Boundary condition: u(-1)=u(-2)=0
        elif n==1 : 
            coupling[n]=(a*k[n+1]*np.conj(u[n+1])*u[n+2]+b*k[n]*np.conj(u[n-1])*u[n+1])*1j #Boundary condition: u(-1)=0
        elif n==N-2 : 
            coupling[n]=(b*k[n]*np.conj(u[n-1])*u[n+1]-c*k[n-1]*u[n-1]*u[n-2])*1j          #Boundary condition: u(Num)=0
        elif n==N-1 : 
            coupling[n]=(-c*k[n-1]*u[n-1]*u[n-2])*1j                                       #Boundary condition: u(Num)=u(Num+1)=0
        else :
            coupling[n]=(a*k[n+1]*np.conj(u[n+1])*u[n+2]+b*k[n]*np.conj(u[n-1])*u[n+1]-c*k[n-1]*u[n-1]*u[n-2])*1j
    return coupling

  

    # This subroutine integrates with RK4 scheme the Sabra shell model of turbulence
    # Each variable u(n) evolves through: 
    #                                   du/dt=G[u,u]-nu k^2 u+forcing
    # If you define "the integral factor" ek(n)=exp(-nu*t*k^2), the evolution can be rewritten in term
    # of a new variable y(n)= u(n)*ek(n)^(-1) as:
    #                                   dy/dt=(G[y,y]+forcing)*ek^(-1)
    # In other words, through the transformation you can integrate exactly the linear dissipative term "-nu k^2 u"

def RK4(u):

    #The presence of integral factor changes the explicit form of Runge-Kutta increments
    A1=dt*(forcing+G(u))          
    A2=dt*(forcing+G(np.multiply(ek,(u+A1/2))))
    A3=dt*(forcing+G(np.multiply(ek,u+A2/2)))
    A4=dt*(forcing+G(np.multiply(u,np.multiply(ek,ek)+np.multiply(ek,A3))))
    
    # In terms of the original variable the evolution rule become:
    u=np.multiply(np.multiply(ek,ek),(u+A1/6))+np.multiply(ek,(A2+A3)/3)+A4/6

    return u


np.random.seed(42)  
# Initialize random velocity
u = np.zeros((N,int(Tmax/dt)),dtype=complex)  # Store velocity vector 

for n in range(6):  # Only the first 6 shells are not zero
    ran = np.random.random()
    u[n] = 0.01 * k[n]**(-0.33) * (np.cos(2 * np.pi * ran) + 1j * np.sin(2 * np.pi * ran))
       
for t in time-1: 
    u[:,t+1]=RK4(u[:,t])

    
for t in time-1:
    u[15,t+1] = u[14,t+1] / 3
    u[16,t+1] = u[15,t+1] / 3
    u[17,t+1] = u[16,t+1] / 3
    u[18,t+1] = u[17,t+1] / 3
    u[19,t+1] = u[18,t+1] / 3


print(u.shape) # -- n * time-1



# plt.plot(2*time*dt,np.real(u[4,:]),label=r'$u_5(t)$')
# plt.plot(2*time*dt,np.real(u[10,:]),label=r'$u_11(t)$')
# plt.xlabel(r'$t/\tau_0$')
# plt.ylabel(r'$u_0(t)$')
# plt.savefig("time_VS_u")
# plt.legend(loc="best")
# plt.show()

# u[15,t+1] = u[14,t+1] / 3
# u[16,t+1] = u[15,t+1] / 3
# u[17,t+1] = u[16,t+1] / 3
# u[18,t+1] = u[17,t+1] / 3
# u[19,t+1] = u[18,t+1] / 3

u_avg = np.mean(u[:, int((len(time)+1)/2):], axis=1)
print(u_avg.shape)

s1 = np.sqrt(np.real(u_avg*np.conj(u_avg)))
s2 = np.sqrt(np.real(u_avg*np.conj(u_avg)))**2
s3 = np.sqrt(np.real(u_avg*np.conj(u_avg)))**3
s4 = np.sqrt(np.real(u_avg*np.conj(u_avg)))**4
s5 = np.sqrt(np.real(u_avg*np.conj(u_avg)))**5
s6 = np.sqrt(np.real(u_avg*np.conj(u_avg)))**6



n = np.arange(1,21)

plt.figure()
plt.plot(n, s1, label = r'$S_1$', marker = 'o')
plt.plot(n, s2, label = r'$S_2$', marker = 'o')
plt.plot(n, s3, label = r'$S_3$', marker = 'o')
plt.plot(n, s4, label = r'$S_4$', marker = 'o')
plt.plot(n, s5, label = r'$S_5$', marker = 'o')
plt.plot(n, s6, label = r'$S_6$', marker = 'o')
# plt.plot(kn, flux, label = 'Flux    ', marker = 'o')
plt.legend(loc='lower left')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$S_n$', fontsize = 16)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))   # Set the x-axis locator to show multiples of 2
#plt.xscale('log', base = 2)
plt.yscale('log', base = 2)
plt.tight_layout()
plt.savefig('struct_functions.png')
plt.close()

