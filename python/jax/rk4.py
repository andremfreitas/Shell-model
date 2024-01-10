import jax 
import jax.numpy as jnp
from jax import random

# test rk4 implementation in jax


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


def RK4(u):
    # The presence of integral factor changes the explicit form of Runge-Kutta increments
    A1 = dt * (forcing + G(u))          
    A2 = dt * (forcing + G(jnp.multiply(ek, (u + A1/2))))
    A3 = dt * (forcing + G(jnp.multiply(ek, (u + A2/2))))
    A4 = dt * (forcing + G(jnp.multiply(u, jnp.multiply(ek**2, ek) + jnp.multiply(ek, A3))))
    
    # In terms of the original variable, the evolution rule becomes:
    u = jnp.multiply(jnp.multiply(ek, ek), (u + A1/6)) + jnp.multiply(ek, (A2 + A3)/3) + A4/6

    return u

def loss_function_rk4(u):
    # Define a scalar loss based on the output of the RK4 function
    output = RK4(u)
    loss = jnp.sum(output)
    return loss

key = random.PRNGKey(123)
random_u = random.normal(key, shape=(N,), dtype=complex)
grad_loss_rk4 = jax.grad(loss_function_rk4, holomorphic=True)
gradient_rk4_at_u = grad_loss_rk4(random_u)
print("Gradient at u for RK4 =", gradient_rk4_at_u)
