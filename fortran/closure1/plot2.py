import matplotlib.pyplot as plt
import csv
import re
import numpy as np
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.cm import ScalarMappable

def compute_pdf(data_vector, num_bins):
    hist, bins = np.histogram(data_vector, bins=num_bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    pdf = hist / np.sum(hist)
    return pdf, bin_centers

# Define a regular expression pattern to extract values inside parentheses
pattern = re.compile(r'\((-?\d+\.\d+(?:E[+-]\d+)?),(-?\d+\.\d+(?:E[+-]\d+)?)\)')

# Lists to store the extracted values
real_u_values = []
im_u_values = []

# Open the CSV file
with open('time_velocity.csv', 'r') as csvfile:
    # Create a CSV reader
    reader = csv.reader(csvfile, delimiter=' ')

    # Iterate through each row
    for row in reader:
        # Join the row into a single string
        row_str = ' '.join(row)

        # Find all matches of the pattern in the row
        matches = pattern.findall(row_str)

        # Extract and append the values to the respective lists
        for match in matches:
            real_u_values.append(float(match[0]))
            im_u_values.append(float(match[1]))

# Convert the lists to NumPy arrays
real_u = np.array(real_u_values)
im_u = np.array(im_u_values)

real_u_matrix = np.reshape(real_u, (500000, 20))
im_u_matrix = np.reshape(im_u, (500000, 20))

complex_u = real_u_matrix + im_u_matrix * 1j

def sf (u_matrix, p):
    return  np.average(np.sqrt(np.real(u_matrix[100000:]*np.conj(u_matrix[100000:]))), axis=0)**p

# closure 1
complex_u_cutoff = complex_u[:, 0:15]

un_cutoff_mag = np.abs(complex_u_cutoff[:, -1])

un_cutoff_mag_squared = un_cutoff_mag ** 2

u_np1_mag_sqr = un_cutoff_mag_squared * 2**(-2/3)
u_np2_mag_sqr = un_cutoff_mag_squared * 2**(-4/3)

u_np1_mag = np.sqrt(u_np1_mag_sqr)
u_np2_mag = np.sqrt(u_np2_mag_sqr)

u_np1 = u_np1_mag * 1j
u_np2 = u_np2_mag * 1j

complex_u_updt = np.column_stack((complex_u_cutoff, u_np1, u_np2))





# # pdf of phase wrt n
# phases_matrix = np.angle(complex_u)
# print(phases_matrix.shape)

# num_columns = 20  # Replace with the actual number of columns
# num_bins = 20  # Replace with the desired number of bins

# phase_pdfs = []
# bins = []

# for i in range(num_columns):
#     phase_pdf, bin_edges = compute_pdf(phases_matrix[:, i], num_bins)
#     phase_pdfs.append(phase_pdf)
#     bins.append(bin_edges)

# num_sets = num_columns
# colors = plt.cm.viridis(np.linspace(0, 1, num_sets))

# fig, ax = plt.subplots()

# for i in range(num_sets):
#     line = ax.plot(bins[i], phase_pdfs[i], label=f'n = {i+1}', color=colors[i])

# ax.set_ylabel('pdf')
# ax.set_xlabel('angle')

# # Create a ScalarMappable for the colorbar
# sm = ScalarMappable(cmap=plt.cm.viridis)
# sm.set_array(np.arange(1, num_sets + 1))
# cbar = plt.colorbar(sm, ax=ax, label='n')
# cbar.set_ticks(np.arange(1, num_sets + 1))


# s = [sf(complex_u_updt, i) for i in range(1, 7)]

# n = np.arange(1,18)

# plt.figure()
# plt.plot(n, s[0], label = r'$S_1$', marker = 'o')
# plt.plot(n, s[1], label = r'$S_2$', marker = 'o')
# plt.plot(n, s[2], label = r'$S_3$', marker = 'o')
# plt.plot(n, s[3], label = r'$S_4$', marker = 'o')
# plt.plot(n, s[4], label = r'$S_5$', marker = 'o')
# plt.plot(n, s[5], label = r'$S_6$', marker = 'o')

# ymin = min(s[5])
# ymax = max(s[5])
# plt.vlines(15, ymin, ymax, colors='black', label='Cut-off', lw = 2, ls = 'dashed')

# plt.legend(loc='lower left')
# plt.xlabel(r'$n$', fontsize = 16)
# plt.ylabel(r'$S_n$', fontsize = 16)
# plt.gca().xaxis.set_major_locator(MultipleLocator(2))   
# plt.yscale('log', base = 2)
# plt.tight_layout()
# plt.savefig('struct_functions_k41_phase_pi2.png')
# plt.close()