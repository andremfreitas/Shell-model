import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


def compute_pdf(data_vector, num_bins):
    # Compute the histogram
    hist, bins = np.histogram(data_vector, bins=num_bins, density=True)

    # Calculate the bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute the PDF
    pdf = hist / np.sum(hist)

    return pdf, bin_centers

def compute_kurtosis(pdf_values, x_values):
    """
    Computes the kurtosis of a probability density function (PDF).

    Parameters:
    - pdf_values (array): Array of PDF values corresponding to x_values.
    - x_values (array): Array of x values corresponding to the PDF values.

    Returns:
    - kurtosis (float): Kurtosis of the PDF.
    """
    mean = np.trapz(x_values * pdf_values, x=x_values)
    variance = np.trapz((x_values - mean)**2 * pdf_values, x=x_values)
    skewness = np.trapz((x_values - mean)**3 * pdf_values, x=x_values) / (variance**1.5)
    fourth_moment = np.trapz((x_values - mean)**4 * pdf_values, x=x_values)
    
    kurtosis = fourth_moment / (variance**2) - 3.0
    
    return kurtosis

def lagrangian_struct_func(u, tau_min, tau_max, p):
    """
    Computes L_tau^p for a given vector u(t), a range of time lags [tau_min, tau_max], and power p.

    Parameters:
    - u (array): Input vector u(t).
    - tau_min (float): Minimum time lag.
    - tau_max (float): Maximum time lag.
    - p (int): Power.

    Returns:
    - tau_values (array): Array of time lags.
    - L_tau_p_vector (array): Vector of L_tau^p values for each time lag.
    """
    N = len(u)
    
    if tau_max >= N:
        raise ValueError("Maximum time lag (tau_max) should be less than the length of the vector (N).")

    # tau_values = np.linspace(tau_min, tau_max, num=int((tau_max - tau_min) / tau_min) + 1)
    tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), num = 10, base = 10)
    L_tau_p_vector = np.zeros_like(tau_values, dtype=float)

    for i, tau in enumerate(tau_values):
        L_tau_p_vector[i] = np.mean(np.abs(u[:-int(tau*N)] - u[int(tau*N):]) ** p)

    return tau_values, L_tau_p_vector

# where to save the figures
folder = 'case1/'

filename = folder + 'kn_S1_6.csv'
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
plt.savefig(folder + 'struct_functions.png')
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
plt.savefig(folder + 'struct_slope_k41.png')
plt.close()


filename2 = folder + 'time__average_input_flux_dissipated.csv'
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
plt.savefig(folder + 'moving_avg_dissipation.png')
plt.close()

plt.figure()
plt.plot(nd_time, input_flux)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'$\langle I_n \rangle$', fontsize = 16)
plt.savefig(folder + 'moving_avg_input.png')
plt.close()

plt.figure()
plt.plot(nd_time, flux_)
plt.xlabel(r'$t/T_0$', fontsize = 16)
plt.ylabel(r'$\langle \Pi_n \rangle$', fontsize = 16)
plt.savefig(folder + 'moving_avg_flux.png')
plt.close()




filename3 = folder + 'time_velocity.csv'
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
plt.savefig(folder + 'pdf.png')
plt.close() 

filename4 = folder + 'time_physical.csv'

table4 = np.genfromtxt(filename4, delimiter='')

time4 = table4[:,0]
i_flux = table4[:,1]        # time series of sum of input flux
conv_flux = table4[:,2]     #  ""   ""   ""         conv flux
d_flux = table4[:,3]        #  ""   ""   ""         dissipation flux

time4_nd = time4 / t0

plt.plot(time4_nd, d_flux, lw = 1)
plt.ylabel(r'$D_N$', fontsize=16)
plt.xlabel(r'$t/t_0$', fontsize =16)
plt.savefig(folder + 'dissipation.png')
plt.close()


# Lagrangian structure functions

filename5 = folder + 'lagrangian.csv'

table5 = np.genfromtxt(filename5, delimiter='')

# v_sum = table3[:,4]
v_sum = table5[:,1]

tau1, lagr1 = lagrangian_struct_func(v_sum, 1e-4, 1e-2, 1)
tau2, lagr2 = lagrangian_struct_func(v_sum, 1e-4, 1e-2, 2)
tau3, lagr3 = lagrangian_struct_func(v_sum, 1e-4, 1e-2, 3)
tau4, lagr4 = lagrangian_struct_func(v_sum, 1e-4, 1e-2, 4)
tau5, lagr5 = lagrangian_struct_func(v_sum, 1e-4, 1e-2, 5)

plt.figure()
plt.loglog(tau1, lagr1, label = r'$S_1$', marker='o')
plt.loglog(tau2, lagr2, label = r'$S_2$', marker='o')
plt.loglog(tau3, lagr3, label = r'$S_3$', marker='o')
plt.loglog(tau4, lagr4, label = r'$S_4$', marker='o')
plt.loglog(tau5, lagr5, label = r'$S_5$', marker='o')
plt.ylabel(r'$L^p_{\tau}$', fontsize = 16)
plt.xlabel(r'$\tau$', fontsize = 16)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(folder + 'lagrangian_struct_func.png')
plt.close()

kurt = s4 / s2**2
plt.plot(kn, kurt, marker='o')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$S_4 / (S_2)^2$', fontsize = 16)
# plt.title('Kurtosis')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.yscale('log', base = 2)
plt.tight_layout()
plt.savefig(folder + 'kurtosis.png')
plt.close()

#########
# Attempt at fluxes vs n

filename6 = folder + 'n_physical.csv'

table6 = np.genfromtxt(filename6, delimiter='')

time6 = table6[:,0]

I_n = np.array([np.average(table6[:,1]), np.average(table6[:,2]), np.average(table6[:,3]), np.average(table6[:,4]), np.average(table6[:,5])
                , np.average(table6[:,6]), np.average(table6[:,7]), np.average(table6[:,8]), np.average(table6[:,9]), np.average(table6[:,10])
                , np.average(table6[:,11]), np.average(table6[:,12]), np.average(table6[:,13]), np.average(table6[:,14]), np.average(table6[:,15])
                , np.average(table6[:,16]), np.average(table6[:,17]), np.average(table6[:,18]), np.average(table6[:,19]), np.average(table6[:,20])])

Pi_n = np.cumsum(np.array([np.average(table6[:,21]), np.average(table6[:,22]), np.average(table6[:,23]), np.average(table6[:,24]), np.average(table6[:,25])
                , np.average(table6[:,26]), np.average(table6[:,27]), np.average(table6[:,28]), np.average(table6[:,29]), np.average(table6[:,30])
                , np.average(table6[:,31]), np.average(table6[:,32]), np.average(table6[:,33]), np.average(table6[:,34]), np.average(table6[:,35])
                , np.average(table6[:,36]), np.average(table6[:,37]), np.average(table6[:,38]), np.average(table6[:,39]), np.average(table6[:,40])]))

D_n = np.cumsum(np.array([np.average(table6[:,41]), np.average(table6[:,42]), np.average(table6[:,43]), np.average(table6[:,44]), np.average(table6[:,45])
                , np.average(table6[:,46]), np.average(table6[:,47]), np.average(table6[:,48]), np.average(table6[:,49]), np.average(table6[:,50])
                , np.average(table6[:,51]), np.average(table6[:,52]), np.average(table6[:,53]), np.average(table6[:,54]), np.average(table6[:,55])
                , np.average(table6[:,56]), np.average(table6[:,57]), np.average(table6[:,58]), np.average(table6[:,59]), np.average(table6[:,60])]))


plt.figure()
plt.plot(kn, I_n, marker = 'o')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$\langle I_n \rangle$', fontsize  = 16)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(folder + 'I_n.png')
plt.close()

plt.figure()
plt.plot(kn, Pi_n, marker = 'o')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$\langle \Pi_n \rangle$', fontsize  = 16)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(folder + 'Pi_n.png')
plt.close()

plt.figure()
plt.plot(kn, D_n, marker = 'o')
plt.xlabel(r'$n$', fontsize = 16)
plt.ylabel(r'$\langle D_n \rangle$', fontsize  = 16)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(folder + 'D_n.png')
plt.close()

#################
# pdf of convective fluxes (pi_n)
################

pdf_pi18, bin_pi18 = compute_pdf(np.average(table6[:,35]), 20)

plt.figure()
plt.plot(bin_pi18, pdf_pi18, lw=2)
plt.xlabel(r'$\Pi_n$', fontsize  = 16)
plt.ylabel('pdf', fontsize = 16)
plt.tight_layout()
plt.savefig(folder + 'pdf_conv_flux.png')
plt.close()
