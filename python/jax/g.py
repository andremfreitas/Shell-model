import jax 
import jax.numpy as jnp
from jax import random

# Here I will test the jax implementation of G [u,u] nonlinear coupling

# Fixed Parameters & Useful vectors
N = 20                               # Num of shells 
nu = 10**-6                           # viscosity 
dt = 10**-5                           # integration step 
a, b, c = (1.0, -0.50, -0.50)                
k, ek, forcing = [], [], []
for n in range(N): 
    k.append(2**n)
    ek.append(jnp.exp(-nu * dt * k[n] * k[n] / 2.0))
    if n == 0:
        forcing.append(1 + 1j)

# Convert the lists to JAX arrays
k = jnp.array(k)
ek = jnp.array(ek)
forcing = jnp.array(forcing)


def G(u):
    # Since the velocities u(n) must be zero when n<0 and n>Num-1, 
    # the non-linear coupling G[u,u] must respect the boundary condition u(-1)=u(-2)=u(Num)=u(Num+1)=0

    coupling = jnp.zeros(N, dtype=jnp.complex64)
    
    for n in range(N): 
        if n == 0:
            coupling = coupling.at[n].set(a * k[n + 1] * jnp.conj(u[n + 1]) * u[n + 2]) * 1j  # Boundary condition: u(-1)=u(-2)=0
        elif n == 1:
            coupling = coupling.at[n].set(a * k[n + 1] * jnp.conj(u[n + 1]) * u[n + 2] + b * k[n] * jnp.conj(u[n - 1]) * u[n + 1]) * 1j  # Boundary condition: u(-1)=0
        elif n == N - 2:
            coupling = coupling.at[n].set(b * k[n] * jnp.conj(u[n - 1]) * u[n + 1] - c * k[n - 1] * u[n - 1] * u[n - 2]) * 1j  # Boundary condition: u(Num)=0
        elif n == N - 1:
            coupling = coupling.at[n].set(-c * k[n - 1] * u[n - 1] * u[n - 2]) * 1j  # Boundary condition: u(Num)=u(Num+1)=0
        else:
            coupling = coupling.at[n].set(a * k[n + 1] * jnp.conj(u[n + 1]) * u[n + 2] + b * k[n] * jnp.conj(u[n - 1]) * u[n + 1] - c * k[n - 1] * u[n - 1] * u[n - 2]) * 1j
    
    return coupling




def loss_function(u):
    # Define a scalar loss based on the output of the solver
    output = G(u)
    loss = jnp.sum(output)  # You can use any aggregation function depending on your objective
    return loss

# Compute the gradient of the loss function with respect to the input
grad_loss = jax.grad(loss_function, holomorphic=True)  # holomorphic defines whether the function in complex or not

# Example usage
u = jnp.zeros(N,dtype=complex) # Input to your solver
gradient_at_x = grad_loss(u)

print("Gradient at x =", gradient_at_x)

key = random.PRNGKey(123)
random_u = random.normal(key, shape=(N,), dtype=complex)
gradient_at_random_u = grad_loss(random_u)
print("Gradient at random input =", gradient_at_random_u)