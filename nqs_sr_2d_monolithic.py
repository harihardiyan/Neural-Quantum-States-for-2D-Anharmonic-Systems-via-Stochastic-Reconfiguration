from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import jax
import jax.numpy as jnp
from jax import jacfwd
from jax.flatten_util import ravel_pytree


# ============================================================
# 0. Physical constants & backend
# ============================================================

@dataclass
class PhysicalConstants:
    hbar: float = 1.0
    mass: float = 1.0


@dataclass
class FlatFD2DBackend:
    Nx: int
    Ny: int
    Lx: float
    Ly: float
    phys: PhysicalConstants

    def coords(self):
        x = jnp.linspace(-self.Lx / 2, self.Lx / 2, self.Nx)
        y = jnp.linspace(-self.Ly / 2, self.Ly / 2, self.Ny)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        return X, Y

    def spacings(self) -> Tuple[float, float]:
        dx = self.Lx / (self.Nx - 1)
        dy = self.Ly / (self.Ny - 1)
        return dx, dy


# ============================================================
# 1. Potentials
# ============================================================

def V_anharmonic_2d(X: jnp.ndarray,
                    Y: jnp.ndarray,
                    lam: float = 0.1) -> jnp.ndarray:
    r2 = X**2 + Y**2
    return 0.5 * r2 + lam * r2**2


def V_double_well_2d(X: jnp.ndarray,
                     Y: jnp.ndarray,
                     lam: float = 1.0,
                     a: float = 2.0) -> jnp.ndarray:
    return lam * (X**2 - a**2)**2 + 0.5 * Y**2


# ============================================================
# 2. Boundary factor (Dirichlet)
# ============================================================

def boundary_factor(X: jnp.ndarray,
                    Y: jnp.ndarray,
                    Lx: float,
                    Ly: float) -> jnp.ndarray:
    gx = 1.0 - (2.0 * X / Lx) ** 2
    gy = 1.0 - (2.0 * Y / Ly) ** 2
    g = gx * gy
    g = jnp.where(g > 0.0, g, 0.0)
    return g


# ============================================================
# 3. NQS MLP (symmetry-aware)
# ============================================================

def init_mlp_params(key: jax.Array,
                    in_dim: int = 2,
                    h1: int = 32,
                    h2: int = 32) -> Dict[str, jnp.ndarray]:
    k1, k2, k3 = jax.random.split(key, 3)

    def glorot(k, shape):
        fan_in, fan_out = shape[0], shape[1]
        scale = jnp.sqrt(2.0 / (fan_in + fan_out))
        return scale * jax.random.normal(k, shape)

    W1 = glorot(k1, (in_dim, h1))
    b1 = jnp.zeros((h1,))
    W2 = glorot(k2, (h1, h2))
    b2 = jnp.zeros((h2,))
    W3 = glorot(k3, (h2, 1))
    b3 = jnp.zeros((1,))

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


def mlp_forward(params: Dict[str, jnp.ndarray],
                x: jnp.ndarray) -> jnp.ndarray:
    h1 = jnp.tanh(x @ params["W1"] + params["b1"])
    h2 = jnp.tanh(h1 @ params["W2"] + params["b2"])
    out = h2 @ params["W3"] + params["b3"]
    return out[..., 0]


def mlp_forward_sym(params: Dict[str, jnp.ndarray],
                    x: jnp.ndarray) -> jnp.ndarray:
    # enforce even parity in x,y
    x1 = x
    x2 = jnp.stack([-x[:, 0],  x[:, 1]], axis=-1)
    x3 = jnp.stack([ x[:, 0], -x[:, 1]], axis=-1)
    x4 = -x

    f1 = mlp_forward(params, x1)
    f2 = mlp_forward(params, x2)
    f3 = mlp_forward(params, x3)
    f4 = mlp_forward(params, x4)

    return 0.25 * (f1 + f2 + f3 + f4)


def psi_on_grid(params: Dict[str, jnp.ndarray],
                xy_flat: jnp.ndarray,
                g_flat: jnp.ndarray) -> jnp.ndarray:
    f = mlp_forward_sym(params, xy_flat)   # (N,)
    psi = jnp.exp(f) * g_flat              # BC-aware
    norm = jnp.sqrt(jnp.vdot(psi, psi))
    psi = psi / (norm + 1e-12)
    return psi


# ============================================================
# 4. Matrix-free 4th-order Laplacian & Hamiltonian
# ============================================================

def laplacian_4th_2d_apply_mf(psi_flat: jnp.ndarray,
                              Nx: int,
                              Ny: int,
                              dx: float,
                              dy: float) -> jnp.ndarray:
    psi = psi_flat.reshape(Nx, Ny)

    # interior indices: 2..Nx-3, 2..Ny-3
    center = psi[2:-2, 2:-2]

    # x-direction
    x_p1 = psi[3:-1, 2:-2]
    x_m1 = psi[1:-3, 2:-2]
    x_p2 = psi[4:  , 2:-2]
    x_m2 = psi[0:-4, 2:-2]

    lap_x = (-x_p2 + 16.0 * x_p1 - 30.0 * center + 16.0 * x_m1 - x_m2) / (12.0 * dx * dx)

    # y-direction
    y_p1 = psi[2:-2, 3:-1]
    y_m1 = psi[2:-2, 1:-3]
    y_p2 = psi[2:-2, 4:  ]
    y_m2 = psi[2:-2, 0:-4]

    lap_y = (-y_p2 + 16.0 * y_p1 - 30.0 * center + 16.0 * y_m1 - y_m2) / (12.0 * dy * dy)

    lap_int = lap_x + lap_y

    lap = jnp.zeros_like(psi)
    lap = lap.at[2:-2, 2:-2].set(lap_int)
    return lap.reshape(-1)


def kinetic_apply_flat(psi_flat: jnp.ndarray,
                       Nx: int,
                       Ny: int,
                       dx: float,
                       dy: float,
                       hbar: float,
                       mass: float) -> jnp.ndarray:
    lap = laplacian_4th_2d_apply_mf(psi_flat, Nx, Ny, dx, dy)
    pref = - (hbar ** 2) / (2.0 * mass)
    return pref * lap


def hamiltonian_apply_flat(psi_flat: jnp.ndarray,
                           V_flat: jnp.ndarray,
                           Nx: int,
                           Ny: int,
                           dx: float,
                           dy: float,
                           hbar: float,
                           mass: float) -> jnp.ndarray:
    Tpsi = kinetic_apply_flat(psi_flat, Nx, Ny, dx, dy, hbar, mass)
    return Tpsi + V_flat * psi_flat


# ============================================================
# 5. Energy & SR (full-state)
# ============================================================

def energy_expectation(params: Dict[str, jnp.ndarray],
                       xy_flat: jnp.ndarray,
                       g_flat: jnp.ndarray,
                       V_flat: jnp.ndarray,
                       Nx: int,
                       Ny: int,
                       dx: float,
                       dy: float,
                       hbar: float,
                       mass: float) -> float:
    psi = psi_on_grid(params, xy_flat, g_flat)
    Hpsi = hamiltonian_apply_flat(psi, V_flat, Nx, Ny, dx, dy, hbar, mass)
    num = jnp.vdot(psi, Hpsi)
    den = jnp.vdot(psi, psi)
    return float(jnp.real(num / den))


def sr_step(params: Dict[str, jnp.ndarray],
            xy_flat: jnp.ndarray,
            g_flat: jnp.ndarray,
            V_flat: jnp.ndarray,
            Nx: int,
            Ny: int,
            dx: float,
            dy: float,
            hbar: float,
            mass: float,
            lr: float = 0.01,
            diag_reg: float = 1e-3):

    flat_params, unravel = ravel_pytree(params)

    def logpsi_flat_params(p_flat):
        p = unravel(p_flat)
        f = mlp_forward_sym(p, xy_flat)
        logpsi = f + jnp.log(g_flat + 1e-12)
        return logpsi  # (N,)

    # J: (N, P)
    J = jacfwd(logpsi_flat_params)(flat_params)

    logpsi = logpsi_flat_params(flat_params)  # (N,)
    psi = jnp.exp(logpsi)
    norm = jnp.sqrt(jnp.vdot(psi, psi))
    psi = psi / (norm + 1e-12)

    # probabilities
    w = jnp.abs(psi)**2
    Z = jnp.sum(w)
    p = w / (Z + 1e-12)               # (N,)

    # local energy
    Hpsi = hamiltonian_apply_flat(psi, V_flat, Nx, Ny, dx, dy, hbar, mass)
    eps = 1e-12
    Eloc = Hpsi / (psi + eps)         # (N,)

    # expectations
    O = J                             # (N, P)
    O_bar = jnp.sum(p[:, None] * O, axis=0)  # (P,)
    Eloc_bar = jnp.sum(p * Eloc)             # scalar

    # gradient
    g = jnp.sum(p[:, None] * O * Eloc[:, None], axis=0) - O_bar * Eloc_bar  # (P,)

    # SR metric
    S = jnp.einsum("i,ia,ib->ab", p, O, O) - jnp.outer(O_bar, O_bar)        # (P, P)
    S_reg = S + diag_reg * jnp.eye(S.shape[0], dtype=S.dtype)

    delta = jnp.linalg.solve(S_reg, -g)
    new_flat = flat_params + lr * delta
    new_params = unravel(new_flat)

    # diagnostics
    E = jnp.real(Eloc_bar)
    var_Eloc = jnp.real(jnp.sum(p * (Eloc - Eloc_bar)**2))
    evals = jnp.linalg.eigvalsh(S_reg)
    cond_S = jnp.max(jnp.abs(evals)) / (jnp.min(jnp.abs(evals)) + 1e-12)

    return new_params, float(E), float(var_Eloc), float(cond_S)


# ============================================================
# 6. Builders & training loop
# ============================================================

def make_anharmonic_problem(Nx=32, Ny=32, Lx=8.0, Ly=8.0, lam=0.1):
    phys = PhysicalConstants()
    backend = FlatFD2DBackend(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, phys=phys)

    X, Y = backend.coords()
    V_grid = V_anharmonic_2d(X, Y, lam=lam)
    V_flat = V_grid.reshape(-1)

    dx, dy = backend.spacings()
    hbar = backend.phys.hbar
    mass = backend.phys.mass

    xy = jnp.stack([X, Y], axis=-1).reshape(-1, 2)  # (N, 2)
    g = boundary_factor(X, Y, backend.Lx, backend.Ly).reshape(-1)

    return xy, g, V_flat, Nx, Ny, dx, dy, hbar, mass


def run_nqs_sr_2d(xy_flat: jnp.ndarray,
                  g_flat: jnp.ndarray,
                  V_flat: jnp.ndarray,
                  Nx: int,
                  Ny: int,
                  dx: float,
                  dy: float,
                  hbar: float,
                  mass: float,
                  key: jax.Array,
                  n_iter: int = 80,
                  lr: float = 0.05,
                  h1: int = 32,
                  h2: int = 32,
                  tag: str = ""):

    params = init_mlp_params(key, in_dim=2, h1=h1, h2=h2)

    for it in range(n_iter):
        params, E, varE, condS = sr_step(
            params, xy_flat, g_flat, V_flat,
            Nx, Ny, dx, dy, hbar, mass,
            lr=lr, diag_reg=1e-3
        )
        if it % 5 == 0 or it == n_iter - 1:
            print(f"{tag} iter {it:4d} | E ≈ {E: .6f} | Var(E_loc) ≈ {varE: .3e} | cond(S) ≈ {condS: .3e}")

    return params


def demo_anharmonic(lam=0.1,
                    Nx=32,
                    Ny=32,
                    Lx=8.0,
                    Ly=8.0,
                    h1=32,
                    h2=32,
                    n_iter=80,
                    lr=0.05,
                    seed=0,
                    tag_prefix="[AH]"):
    xy, g_flat, V_flat, Nx, Ny, dx, dy, hbar, mass = make_anharmonic_problem(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, lam=lam
    )
    key = jax.random.PRNGKey(seed)
    tag = f"{tag_prefix} λ={lam}"
    _ = run_nqs_sr_2d(
        xy, g_flat, V_flat, Nx, Ny, dx, dy, hbar, mass,
        key, n_iter=n_iter, lr=lr, h1=h1, h2=h2, tag=tag
    )


if __name__ == "__main__":
    # baseline λ=0.1
    demo_anharmonic(lam=0.1)

    # contoh strongly anharmonic λ=10
    # demo_anharmonic(lam=10.0, n_iter=120, lr=0.03, tag_prefix="[AH-strong]")
