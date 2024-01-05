import math

dn = 2.5
nu = 1e-6

eta = (nu**3 / dn)**0.25     # lenght scale
k_eta = 1 / eta

print('Kolmogorov wavenumber:', k_eta)

n = math.log2(k_eta)

print(f"The power of 2 closest to {k_eta} is 2^{round(n)}")
