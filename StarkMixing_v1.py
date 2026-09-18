import numpy as np
import time
from pathlib import Path

PI = np.pi
TWO_PI = 2.0 * np.pi
Mega = 1e6
Kilo = 1e3
Giga = 1e9

N_number = 1

gamma_2 = TWO_PI * 5.2 * Mega
gamma_3 = TWO_PI * 3.9 * Kilo
gamma_4 = TWO_PI * 1.7 * Kilo
gamma_c = 0.0
gamma = 0.0

OMEGA_p = TWO_PI * 5.7 * Mega
OMEGA_c = TWO_PI * 0.97 * Mega
OMEGA_S = TWO_PI * 10 * Mega

omega_34_0 = TWO_PI * 6.95 * Giga

E_z = 50.0
beta = 0.223 * Giga
betta = beta
OMEGA_ast = beta * E_z
theta = 0.5 * np.arctan2(OMEGA_ast, omega_34_0)
omega_34_dressed = np.sqrt(omega_34_0**2 + OMEGA_ast**2)

omega_S = omega_34_dressed
Delta_S = omega_S - omega_34_dressed

OMEGA_c_3 = OMEGA_c * np.cos(theta)
OMEGA_c_4 = OMEGA_c * np.sin(theta)
OMEGA_S_eff = OMEGA_S * np.cos(2.0 * theta)

detuning_min = -150.0 * Mega * TWO_PI
detuning_max = 150.0 * Mega * TWO_PI
detuning_step = 0.01 * Mega * TWO_PI

detuning_range = np.arange(detuning_min, detuning_max + 0.5*detuning_step, detuning_step)


phi_s = 0.0


def H0_matrix(detuning_p, detuning_c):
    return -np.array(
        [
            [0.0, OMEGA_p, 0.0, 0.0],
            [OMEGA_p, -2.0 * detuning_p, OMEGA_c_3, 0.0],
            [0.0, OMEGA_c_3, -2.0 * (detuning_p + detuning_c), OMEGA_S_eff],
            [0.0, 0.0, OMEGA_S_eff, -2.0 * (detuning_p + detuning_c + Delta_S)],
        ],
        dtype=complex,
    )


def Hplus_matrix(phi_s):
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, OMEGA_c_4 * np.exp(-1j * phi_s)],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )


def Hminus_matrix(phi_s):
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, OMEGA_c_4 * np.exp(+1j * phi_s), 0.0, 0.0],
        ],
        dtype=complex,
    )


Gamma = np.array(
    [
        [gamma, 0.0, 0.0, 0.0],
        [0.0, gamma + gamma_2, 0.0, 0.0],
        [0.0, 0.0, gamma + gamma_3 + gamma_c, 0.0],
        [0.0, 0.0, 0.0, gamma + gamma_4],
    ],
    dtype=complex,
)

I4x4 = np.eye(4, dtype=complex)
I16x16 = np.eye(16, dtype=complex)


def vec_F(X):
    return X.reshape((16,), order="F")


def mat_F(v):
    return v.reshape((4, 4), order="F")


l0_16x1 = np.zeros(16, dtype=complex)
l0_16x1[0] = gamma

L16x16 = np.zeros((16, 16), dtype=complex)
L16x16[0, 5] = gamma_2
L16x16[0, 15] = gamma_4
L16x16[5, 10] = gamma_3

t_1x16 = np.zeros((16,), dtype=complex)
t_1x16[0] = 1.0
t_1x16[5] = 1.0
t_1x16[10] = 1.0
t_1x16[15] = 1.0


def build_superoperator_A_n(n_harmonic, detuning_p, detuning_c):
    H0 = H0_matrix(detuning_p, detuning_c)
    C0 = np.kron(I4x4, H0) - np.kron(H0.T, I4x4)
    R = 0.5 * (np.kron(I4x4, Gamma) + np.kron(Gamma.T, I4x4))
    A_n = (1j * n_harmonic * omega_S) * I16x16 + 1j * C0 + R - L16x16
    return A_n


def build_superoperator_B_pm(phi_s):
    Hp = Hplus_matrix(phi_s)
    Hm = Hminus_matrix(phi_s)
    Cp = np.kron(I4x4, Hp) - np.kron(Hp.T, I4x4)
    Cm = np.kron(I4x4, Hm) - np.kron(Hm.T, I4x4)
    B_p = 1j * Cp
    B_m = 1j * Cm
    return B_p, B_m


def build_A_cal(detuning_p, detuning_c, phi_s):
    n_list = list(range(-N_number, N_number + 1))
    dim = 16 * (2 * N_number + 1)
    A_cal = np.zeros((dim, dim), dtype=complex)
    B_p, B_m = build_superoperator_B_pm(phi_s)

    for idx, n in enumerate(n_list):
        r0 = 16 * idx
        A_cal[r0 : r0 + 16, r0 : r0 + 16] = build_superoperator_A_n(
            n, detuning_p, detuning_c
        )

        if idx > 0:
            A_cal[r0 : r0 + 16, r0 - 16 : r0] = B_p
        if idx < len(n_list) - 1:
            A_cal[r0 : r0 + 16, r0 + 16 : r0 + 32] = B_m

    return A_cal


def build_Y():
    n_list = list(range(-N_number, N_number + 1))
    dim = 16 * (2 * N_number + 1)
    Y = np.zeros((dim,), dtype=complex)

    for idx, n in enumerate(n_list):
        if n == 0:
            r0 = 16 * idx
            Y[r0 : r0 + 16] = l0_16x1

    return Y


def build_T():
    n_list = list(range(-N_number, N_number + 1))
    dim = 16 * (2 * N_number + 1)
    T = np.zeros((len(n_list), dim), dtype=complex)

    for idx, _n in enumerate(n_list):
        c0 = 16 * idx
        T[idx, c0 : c0 + 16] = t_1x16

    return T


def build_C():
    n_list = list(range(-N_number, N_number + 1))
    C = np.zeros((len(n_list),), dtype=complex)
    C[n_list.index(0)] = 1.0
    return C


def solve_constrained_system(A_cal, Y):
    n_list = list(range(-N_number, N_number + 1))
    A_mod = A_cal.copy()
    b_mod = Y.copy()
    C = build_C()

    for idx, _n in enumerate(n_list):
        row = 16 * idx
        A_mod[row, :] = 0.0
        A_mod[row, 16 * idx : 16 * idx + 16] = t_1x16
        b_mod[row] = C[idx]

    return np.linalg.solve(A_mod, b_mod)


_FLOQUET_CACHE = {}


def solve_floquet_P_cal(detuning_p, detuning_c):
    key = (float(detuning_p), float(detuning_c), float(phi_s), int(N_number))
    if key in _FLOQUET_CACHE:
        return _FLOQUET_CACHE[key]

    n_list = list(range(-N_number, N_number + 1))
    A_cal = build_A_cal(detuning_p, detuning_c, phi_s)
    Y = build_Y()
    x = solve_constrained_system(A_cal, Y)

    P = {}
    for idx, n in enumerate(n_list):
        r0 = 16 * idx
        P[n] = mat_F(x[r0 : r0 + 16])

    _FLOQUET_CACHE[key] = P
    return P


def calculate_rho_n(n, Phi_S, detuning_p, detuning_c):
    P_cal = solve_floquet_P_cal(detuning_p, detuning_c)
    if n not in P_cal:
        raise ValueError("n must satisfy -N <= n <= N")
    return P_cal[n] * np.exp(1j * n * Phi_S)


def calculate_rho_time(t, Phi_S, detuning_p, detuning_c):
    rho_t = np.zeros((4, 4), dtype=complex)
    for n in range(-N_number, N_number + 1):
        rho_t += calculate_rho_n(n, Phi_S, detuning_p, detuning_c) * np.exp(
            1j * n * omega_S * t
        )
    return rho_t


def rho21_vs_probe_detuning(Phi_S, t=0.0):
    re = np.zeros(detuning_range.shape[0], dtype=float)
    im = np.zeros(detuning_range.shape[0], dtype=float)

    for k, detuning_p in enumerate(detuning_range):
        rho_t = calculate_rho_time(t, Phi_S, detuning_p, 0.0)
        rho21 = rho_t[1, 0]
        re[k] = np.real(rho21)
        im[k] = np.imag(rho21)

    detuning_MHz = detuning_range / (TWO_PI * Mega)
    return re, im, detuning_MHz


def calculate_P_n(n,  detuning_p, detuning_c):
    P_cal = solve_floquet_P_cal(detuning_p, detuning_c)
    if n not in P_cal:
        raise ValueError("n must satisfy -N <= n <= N")
    return P_cal[n] 

def wrap_2pi(x):
    return float(x - TWO_PI * np.floor(x / TWO_PI))

def demo():
    P_n = calculate_P_n(1, 0, 0)
    P_21 = P_n[1, 0]
    print("P_21 =", P_21)
    print("|P_21| =", abs(P_21))



if __name__ == "__main__":
    t0 = time.time()
    demo()
    #print(time.time() - t0)
