import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

filename = 'case1/kn_S1_6.csv'
table = np.genfromtxt(filename, delimiter='')

kn = table[:,0]
s1 = table[:,1]         # <|u|> 
s2 = table[:,2]         # <|u|^2>
s3 = table[:,3]         #  ...  
s4 = table[:,4]         # <|u|^p> 
s5 = table[:,5]
s6 = table[:,6]
flux = table[:,7]

t0 = 1 / np.sqrt(kn[0]**2 * s2[0])  # characteristic time scale -- useful for non-dimensionalisation
kn = np.log2(table[:,0])

plt.figure()
plt.plot(kn, s1, label = r'$S_1$', marker = 'o')
plt.plot(kn, s2, label = r'$S_2$', marker = 'o')
plt.plot(kn, s3, label = r'$S_3$', marker = 'o')
plt.plot(kn, s4, label = r'$S_4$', marker = 'o')
plt.plot(kn, s5, label = r'$S_5$', marker = 'o')
plt.plot(kn, s6, label = r'$S_6$', marker = 'o')
# plt.plot(kn, flux, label = 'Flux    ', marker = 'o')
plt.legend(loc='lower left')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$S_n$', fontsize = 16)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))   # Set the x-axis locator to show multiples of 2
#plt.xscale('log', base = 2)
plt.yscale('log', base = 2)
plt.tight_layout()
plt.savefig('case1/struct_functions.png')

filename2 = 'case1/time__average_input_flux_dissipated.csv'
table2 = np.genfromtxt(filename2, delimiter = '')

time = table2[:,0]
input_flux = table2[:,1]
flux_ = table2[:,2]
dissipation = table2[:,3]

nd_time = time / t0

plt.figure()
plt.plot(nd_time, dissipation)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'$\langle D_n \rangle$', fontsize = 16)
plt.savefig('case1/avg_dissipation.png')

plt.figure()
plt.plot(nd_time, input_flux)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'$\langle D_n \rangle$', fontsize = 16)
plt.savefig('case1/input_flux_avg.png')

plt.figure()
plt.plot(nd_time, flux_)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'flux_', fontsize = 16)
plt.savefig('case1/_flux_avg.png')

