import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def compute_pdf(data_vector, num_bins):
    # Compute the histogram
    hist, bins = np.histogram(data_vector, bins=num_bins, density=True)

    # Calculate the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute the PDF
    pdf = hist / np.sum(hist)

    return pdf, bin_centers


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
plt.close()


# Just plot the flux by itself

plt.figure()
plt.plot(kn, flux, label = 'Flux', marker = 'o')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'Flux', fontsize = 16)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))   # Set the x-axis locator to show multiples of 2
#plt.xscale('log', base = 2)
plt.yscale('log', base = 2)
plt.tight_layout()
plt.savefig('case1/flux.png')
plt.close()

# plot k41 theory slopes and data points for inertial range

plt.figure()
plt.scatter(kn[5:15], s1[5:15], marker = 'o')
plt.scatter(kn[5:15], s2[5:15], marker = 'o')
plt.scatter(kn[5:15], s3[5:15], marker = 'o')
plt.scatter(kn[5:15], s4[5:15], marker = 'o')
plt.scatter(kn[5:15], s5[5:15], marker = 'o')
plt.scatter(kn[5:15], s6[5:15], marker = 'o')

diff1 = s1[5] / (table[:,0])[5]**(-1/3) 
diff2 = s2[5] / (table[:,0])[5]**(-2/3) 
diff3 = s3[5] / (table[:,0])[5]**(-3/3) 
diff4 = s4[5] / (table[:,0])[5]**(-4/3) 
diff5 = s5[5] / (table[:,0])[5]**(-5/3) 
diff6 = s6[5] / (table[:,0])[5]**(-6/3) 

plt.plot(kn[5:15], diff1 * ((table[:,0])[5:15])**(-1/3) , label = 'Slope -1/3')
plt.plot(kn[5:15], diff2 * ((table[:,0])[5:15])**(-2/3) , label = 'Slope -2/3')
plt.plot(kn[5:15], diff3 * ((table[:,0])[5:15])**(-3/3), label = 'Slope -1')
plt.plot(kn[5:15], diff4 * ((table[:,0])[5:15])**(-4/3), label = 'Slope -4/3')
plt.plot(kn[5:15], diff5 * ((table[:,0])[5:15])**(-5/3), label = 'Slope -5/3')
plt.plot(kn[5:15], diff6 * ((table[:,0])[5:15])**(-6/3), label = 'Slope -6/3')
plt.legend(loc='lower left')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$S_n$', fontsize = 16)
plt.yscale('log', base = 2)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.tight_layout()
plt.savefig('case1/struct_slope_k41.png')
plt.close()


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
plt.close()

plt.figure()
plt.plot(nd_time, input_flux)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'$\langle D_n \rangle$', fontsize = 16)
plt.savefig('case1/input_flux_avg.png')
plt.close()

plt.figure()
plt.plot(nd_time, flux_)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'flux_', fontsize = 16)
plt.savefig('case1/_flux_avg.png')
plt.close()




filename3 = 'time_velocity.csv'
table3 = np.genfromtxt(filename3, delimiter='')

time3 = table3[:,0]
veloc_n5 = table3[:,1]
veloc_n10 = table3[:,2]
veloc_n15 = table3[:,3]

veloc_n5_sqr = veloc_n5 ** 2
veloc_n10_sqr = veloc_n10 ** 2
veloc_n15_sqr = veloc_n15 ** 2

avg_vel_n5 = np.average(veloc_n5_sqr) ** 0.5 
avg_vel_n10 = np.average(veloc_n10_sqr) ** 0.5
avg_vel_n15 = np.average(veloc_n15_sqr) ** 0.5

v_n5_nd = veloc_n5 / avg_vel_n5
v_n10_nd = veloc_n10 / avg_vel_n10
v_n15_nd = veloc_n15 / avg_vel_n15

# your_data_vector = np.random.randn(1000)  # Example random data
nbins = 250
pdf5, bin_centers5 = compute_pdf(v_n5_nd, nbins)
pdf10, bin_centers10 = compute_pdf(v_n10_nd, nbins)
pdf15, bin_centers15 = compute_pdf(v_n15_nd, nbins)

plt.plot(bin_centers5, pdf5, label='n = 5', lw =2)
plt.plot(bin_centers10, pdf10, label='n = 10', lw = 2)
plt.plot(bin_centers15, pdf15, label='n = 15', lw = 2)
# plt.title('Probability Density Function (PDF)')
plt.legend()
plt.xlabel(r'$\Re(u_n) / \langle (\Re(u_n))^2 \rangle^{\frac{1}{2}}$ ', fontsize = 16)
plt.yscale('log', base = 10)
plt.ylabel('pdf', fontsize = 16)
plt.tight_layout()
plt.savefig('case1/pdf.png')
plt.close() 

filename4 = 'time_physical.csv'

table4 = np.genfromtxt(filename4, delimiter='')

time4 = table4[:,0]
i_flux = table4[:,1]
conv_flux = table4[:,2]
d_flux = table4[:,3]

time4_nd = time4 / t0

plt.plot(time4_nd, d_flux, lw = 1)
plt.ylabel(r'$D_N$', fontsize=16)
plt.xlabel(r'$t/t_0$', fontsize =16)
plt.savefig('case1/dissipation.png')
plt.close()
