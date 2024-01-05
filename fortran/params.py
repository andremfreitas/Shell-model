import math

dn = 2.5
nu = 1e-6

eta = (nu**3 / dn)**0.25     # lenght scale
k_eta = 1 / eta

print('Kolmogorov wavenumber:', "{:e}".format(k_eta))

n = math.log2(k_eta)

print(f"The power of 2 closest to {k_eta:e} is 2^{round(n)}")

tau_eta = (nu / dn) ** 0.5 

print('Kolmogorov time scale:', "{:e}".format(tau_eta))
