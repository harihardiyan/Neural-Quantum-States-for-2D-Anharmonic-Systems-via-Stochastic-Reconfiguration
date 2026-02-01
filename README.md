
# Neural Quantum States for 2D Anharmonic Systems via Stochastic Reconfiguration
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![JAX](https://img.shields.io/badge/framework-JAX-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Field](https://img.shields.io/badge/field-Quantum%20Physics-blueviolet.svg)

**Author:** Hari Hardiyan  
**Email:** lorozloraz@gmail.com  
**Date:** January 31, 2026  
**License:** MIT

---

## Abstract
This project implements a monolithic solver for the 2D Schrödinger equation using **Neural Quantum States (NQS)** optimized through **Stochastic Reconfiguration (SR)**. Unlike traditional Variational Monte Carlo (VMC) that relies on stochastic sampling, this implementation utilizes a **deterministic full-grid integration** to eliminate sampling noise. The ansatz combines a symmetry-preserving Multi-Layer Perceptron (MLP) with an analytical boundary factor to enforce Dirichlet conditions. The kinetic energy is computed using a matrix-free 4th-order finite difference operator, ensuring high-order accuracy while adhering to the variational principle.

---

## Technical Specifications

### 1. Wavefunction Ansatz
The trial wavefunction $\Psi_{\theta}(\mathbf{r})$ is defined as:
$$\Psi_{\theta}(x, y) = e^{f_{\text{MLP}}(x, y)} \cdot g(x, y)$$

### Symmetry Enforcement
To ensure the ground state resides in the even-parity sector, the MLP output $f_{\text{MLP}}$ is symmetrized across all four quadrants to enforce global parity:
$$f_{\text{sym}}(x, y) = \frac{1}{4} [f(x, y) + f(-x, y) + f(x, -y) + f(-x, -y)]$$

### Boundary Factor
Dirichlet boundary conditions ($\Psi=0$ at the edges) are analytically enforced using a geometric factor:
$$g(x, y) = (1 - (2x/L_x)^2) \cdot (1 - (2y/L_y)^2)$$
This factor restricts the Hilbert space to physically valid states, allowing the neural network to focus exclusively on the high-curvature regions of the potential.

---

### 2. Hamiltonian Operator

###  Matrix-Free Kinematics
The kinetic operator is implemented via a **4th-order Finite Difference stencil**. To maintain memory efficiency, the Laplacian is applied using a matrix-free approach, avoiding the construction of large $N^2 \times N^2$ matrices while providing $O(\Delta x^4)$ precision.

### Potential Energy
The solver supports arbitrary 2D potentials, with built-in configurations for:
- **Anharmonic Oscillator:** $V(r) = \frac{1}{2}r^2 + \lambda r^4$.
- **Double-Well Potential:** $V(x, y) = \lambda(x^2 - a^2)^2 + \frac{1}{2}y^2$.

---

### 3. Optimization Logic

### Stochastic Reconfiguration (SR)
Parameter updates follow the **Natural Gradient** in the Hilbert space manifold. The update rule is governed by the Fisher Information Matrix (S-matrix):
$$\theta_{t+1} = \theta_t - \eta S^{-1} \nabla E$$
A Tikhonov regularization (damping) of $10^{-3}$ is added to the S-matrix diagonal to ensure numerical stability during inversion.

---

## Numerical Benchmarks

Evaluated on the 2D Anharmonic Oscillator ($\lambda=0.1, L=8, N=32$):

| Metric | Value |
| :--- | :--- |
| **NQS Ground State Energy** | $\approx 1.150059$ |
| **FD Gold Standard (Orde-4)** | $\approx 1.14918$ |
| **Local Energy Variance** | $\approx 5.06 \times 10^{-4}$ |
| **S-Matrix Condition Number** | $\approx 3.89 \times 10^2$ |

### Interpretation
The NQS result successfully converges to a variational upper bound within **0.07%** of the high-precision finite difference benchmark. The low variance of the local energy and the well-conditioned S-matrix indicate that the neural ansatz has effectively approximated the true eigenstate within the grid's resolution.

---

## Implementation Details

###  Dependencies
The script is monolithic and requires:
- `jax`
- `jaxlib`

###  Execution
Run the monolithic solver using the following command:
```bash
python nqs_sr_2d_monolithic.py
```

## Citation
> Hardiyan, H. (2026). Neural Quantum States for 2D Anharmonic Systems via Stochastic Reconfiguration. lorozloraz@gmail.com.

## License
Licensed under the **MIT License**. See the [LICENSE](https://opensource.org/licenses/MIT) file for more details.
```

