"""
Generate multi-input / multi-output (MIMO) engineering benchmark datasets for
surrogate-model validation. Every dataset is produced by a deterministic
physics model (analytic or numerical), sampled with a Latin hypercube design.

Run:  python generate_datasets.py   -> writes ./datasets/*.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

OUT = "datasets"
os.makedirs(OUT, exist_ok=True)
SEED = 2026


def lhs(bounds, n, seed=SEED, log=()):
    names = list(bounds)
    u = qmc.LatinHypercube(d=len(names), seed=seed).random(n)
    X = {}
    for j, k in enumerate(names):
        lo, hi = bounds[k]
        if k in log:
            X[k] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            X[k] = lo + u[:, j] * (hi - lo)
    return pd.DataFrame(X)


def add_split(df, seed=SEED, test_frac=0.2):
    rng = np.random.default_rng(seed)
    df.insert(0, "sample_id", np.arange(len(df)))
    df["split"] = np.where(rng.random(len(df)) < test_frac, "test", "train")
    return df


# ---------------------------------------------------------------------------
# A1. NACA 4-digit airfoil: linear-strength vortex panel method + viscous estimate
# ---------------------------------------------------------------------------
def naca4(m, p, t, n=80):
    beta = np.linspace(0, np.pi, n + 1)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                  + 0.2843 * x**3 - 0.1036 * x**4)  # closed TE
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p)**2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    X = np.concatenate([xl[::-1], xu[1:]])
    Y = np.concatenate([yl[::-1], yu[1:]])
    return X, Y  # TE(lower) -> LE -> TE(upper), clockwise


def vortex_panel(X, Y, alpha):
    """Linear-strength vortex panel method (Kuethe & Chow). Returns Cl, Cm_c/4, Cp."""
    N = len(X) - 1
    xc, yc = 0.5 * (X[:-1] + X[1:]), 0.5 * (Y[:-1] + Y[1:])
    S = np.hypot(np.diff(X), np.diff(Y))
    th = np.arctan2(np.diff(Y), np.diff(X))
    sin_t, cos_t = np.sin(th), np.cos(th)
    Xi, Xj = xc[:, None], X[None, :-1]
    Yi, Yj = yc[:, None], Y[None, :-1]
    A = -(Xi - Xj) * cos_t[None] - (Yi - Yj) * sin_t[None]
    B = (Xi - Xj) ** 2 + (Yi - Yj) ** 2
    C = np.sin(th[:, None] - th[None])
    D = np.cos(th[:, None] - th[None])
    E = (Xi - Xj) * sin_t[None] - (Yi - Yj) * cos_t[None]
    F = np.log(1 + S[None] * (S[None] + 2 * A) / B)
    G = np.arctan2(E * S[None], B + A * S[None])
    P = (Xi - Xj) * np.sin(th[:, None] - 2 * th[None]) + (Yi - Yj) * np.cos(th[:, None] - 2 * th[None])
    Q = (Xi - Xj) * np.cos(th[:, None] - 2 * th[None]) - (Yi - Yj) * np.sin(th[:, None] - 2 * th[None])
    Cn2 = D + 0.5 * Q * F / S[None] - (A * C + D * E) * G / S[None]
    Cn1 = 0.5 * D * F + C * G - Cn2
    Ct2 = C + 0.5 * P * F / S[None] + (A * D - C * E) * G / S[None]
    Ct1 = 0.5 * C * F - D * G - Ct2
    I = np.eye(N, dtype=bool)
    Cn1[I], Cn2[I], Ct1[I], Ct2[I] = -1, 1, np.pi / 2, np.pi / 2
    An = np.zeros((N + 1, N + 1)); At = np.zeros((N, N + 1))
    An[:N, 0] = Cn1[:, 0]; An[:N, N] = Cn2[:, -1]
    An[:N, 1:N] = Cn1[:, 1:] + Cn2[:, :-1]
    At[:, 0] = Ct1[:, 0]; At[:, N] = Ct2[:, -1]
    At[:, 1:N] = Ct1[:, 1:] + Ct2[:, :-1]
    An[N, 0] = An[N, N] = 1  # Kutta condition
    rhs = np.append(np.sin(th - alpha), 0)
    g = np.linalg.solve(An, rhs)
    Vt = np.cos(th - alpha) + At @ g
    Cp = 1 - Vt**2
    # forces from pressure integration
    nx, ny = -sin_t, cos_t  # outward normal for clockwise ordering
    fx, fy = -Cp * S * nx, -Cp * S * ny
    Cl = fy * np.cos(alpha) - fx * np.sin(alpha)
    Cm = -(fy * (xc - 0.25)) + fx * yc
    return Cl.sum(), Cm.sum(), Cp


def gen_airfoil(n=1500):
    b = {"max_camber_m": (0.0, 0.06), "camber_pos_p": (0.2, 0.6),
         "thickness_t": (0.06, 0.20), "alpha_deg": (-4.0, 8.0),
         "mach": (0.05, 0.50), "reynolds": (2e5, 1e7)}
    X = lhs(b, int(n * 1.6), log=("reynolds",))
    rows = []
    for _, r in X.iterrows():
        xs, ys = naca4(r.max_camber_m, r.camber_pos_p, r.thickness_t)
        a = np.radians(r.alpha_deg)
        cl0, cm0, cp = vortex_panel(xs, ys, a)
        beta = np.sqrt(1 - r.mach**2)  # Prandtl-Glauert
        cl, cm = cl0 / beta, cm0 / beta
        cpmin = cp.min() / beta
        # viscous drag: turbulent flat plate (Schlichting) x form factor x lift penalty
        cf = 0.455 / np.log10(r.reynolds) ** 2.58
        ff = 1 + 2.0 * r.thickness_t + 60 * r.thickness_t**4
        cd = 2 * cf * ff * (1 + 0.06 * cl**2)
        g = 1.4; M = r.mach
        cp_crit = 2 / (g * M**2) * (((2 + (g - 1) * M**2) / (g + 1)) ** (g / (g - 1)) - 1)
        rows.append((cl, cd, cm, cl / cd, cpmin, cpmin > cp_crit))
    Y = pd.DataFrame(rows, columns=["CL", "CD", "CM_c4", "L_over_D", "Cp_min", "ok"])
    df = pd.concat([X, Y], axis=1)
    df = df[df.ok].drop(columns="ok").iloc[:n].reset_index(drop=True)  # subcritical flow only
    return add_split(df)


# ---------------------------------------------------------------------------
# A2. Finite wing: Prandtl lifting-line (Glauert Fourier series)
# ---------------------------------------------------------------------------
def gen_wing(n=2000, NM=24):
    b = {"aspect_ratio": (4.0, 14.0), "taper_ratio": (0.2, 1.0),
         "twist_deg": (-6.0, 0.0), "alpha_root_deg": (-2.0, 10.0),
         "airfoil_cl_alpha_per_rad": (5.6, 6.8), "alpha_L0_deg": (-4.0, 0.0)}
    X = lhs(b, n)
    theta = np.arange(1, NM + 1) * np.pi / (2 * NM)  # half span, root at pi/2
    y = -np.cos(theta)  # 2y/b in [-1, 0]
    nn = np.arange(1, 2 * NM, 2)  # symmetric loading: odd terms
    rows = []
    for _, r in X.iterrows():
        AR, lam = r.aspect_ratio, r.taper_ratio
        c_root = 2 / (AR * (1 + lam))  # chord normalised by span b=1... c/b
        chord = c_root * (1 - (1 - lam) * np.abs(y))
        alpha = np.radians(r.alpha_root_deg + r.twist_deg * np.abs(y) - r.alpha_L0_deg)
        a0 = r.airfoil_cl_alpha_per_rad
        mu = chord * a0 / 4  # (c a0)/(4 b)
        M = np.sin(np.outer(theta, nn)) * (mu[:, None] * nn[None] / np.sin(theta)[:, None] + 1)
        rhs = mu * alpha
        An = np.linalg.lstsq(M, rhs, rcond=None)[0]
        CL = np.pi * AR * An[0]
        delta = np.sum(nn[1:] * (An[1:] / An[0]) ** 2) if abs(An[0]) > 1e-12 else 0.0
        CDi = CL**2 / (np.pi * AR) * (1 + delta)
        e = 1 / (1 + delta)
        # sectional cl & root bending moment coefficient
        circ = 2 * np.sin(np.outer(theta, nn)) @ An  # Gamma/(b V)
        cl_sec = 2 * circ / chord
        yy = np.linspace(0, 1, 400)
        th_y = np.arccos(-yy)
        G = 2 * np.sin(np.outer(th_y, nn)) @ An
        CMroot = AR * np.trapezoid(G * yy, yy)  # M_root / (q S b/2)
        y_cmax = np.abs(y[np.argmax(cl_sec)])
        rows.append((CL, CDi, cl_sec.max(), y_cmax, CMroot))
    Y = pd.DataFrame(rows, columns=["CL", "CDi", "cl_section_max",
                                    "eta_of_cl_max", "root_bending_coeff"])
    return add_split(pd.concat([X, Y], axis=1))


# ---------------------------------------------------------------------------
# S1. Classic 10-bar planar truss (linear FEM, direct stiffness)
# ---------------------------------------------------------------------------
def gen_truss(n=3000):
    L = 9.144  # m (360 in)
    nodes = np.array([[2 * L, L], [2 * L, 0], [L, L], [L, 0], [0, L], [0, 0]])
    elems = np.array([[4, 2], [2, 0], [5, 3], [3, 1], [2, 3], [0, 1], [4, 3], [5, 2], [2, 1], [3, 0]])
    fixed = [8, 9, 10, 11]  # nodes 5,6 (index 4,5) pinned
    free = [i for i in range(12) if i not in fixed]
    b = {f"A{i+1}_cm2": (0.645, 225.8) for i in range(10)}  # 0.1–35 in^2
    b.update({"E_GPa": (60.0, 210.0), "P2_kN": (200.0, 600.0), "P4_kN": (200.0, 600.0),
              "density_kg_m3": (2700.0, 7850.0)})
    X = lhs(b, n)
    rows = []
    for _, r in X.iterrows():
        A = np.array([r[f"A{i+1}_cm2"] for i in range(10)]) * 1e-4
        E = r.E_GPa * 1e9
        K = np.zeros((12, 12)); Ls = []; cs = []
        for e, (i, j) in enumerate(elems):
            d = nodes[j] - nodes[i]; le = np.hypot(*d); c, s = d / le
            k = E * A[e] / le * np.outer([-c, -s, c, s], [-c, -s, c, s])
            idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K[np.ix_(idx, idx)] += k; Ls.append(le); cs.append((c, s))
        F = np.zeros(12); F[3] = -r.P2_kN * 1e3; F[7] = -r.P4_kN * 1e3
        u = np.zeros(12); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        stress = []
        for e, (i, j) in enumerate(elems):
            c, s = cs[e]
            stress.append(E / Ls[e] * np.dot([-c, -s, c, s], u[[2*i, 2*i+1, 2*j, 2*j+1]]) / 1e6)
        mass = r.density_kg_m3 * np.dot(A, Ls)
        disp = np.hypot(u[0::2], u[1::2]) * 1e3
        rows.append([mass, disp.max(), u[3] * 1e3, u[7] * 1e3, max(np.abs(stress))] + stress)
    cols = ["mass_kg", "max_disp_mm", "v_node2_mm", "v_node4_mm", "max_abs_stress_MPa"] + \
           [f"stress{i+1}_MPa" for i in range(10)]
    return add_split(pd.concat([X, pd.DataFrame(rows, columns=cols)], axis=1))


# ---------------------------------------------------------------------------
# S2. Thick-walled pressure cylinder (Lamé) with closed ends
# ---------------------------------------------------------------------------
def gen_cylinder(n=3000):
    b = {"r_inner_mm": (20.0, 200.0), "wall_ratio_ro_ri": (1.1, 3.0),
         "p_internal_MPa": (5.0, 150.0), "p_external_MPa": (0.0, 30.0),
         "E_GPa": (70.0, 210.0), "poisson": (0.25, 0.35), "yield_MPa": (250.0, 900.0)}
    X = lhs(b, n)
    ri = X.r_inner_mm; ro = ri * X.wall_ratio_ro_ri; pi, po = X.p_internal_MPa, X.p_external_MPa
    E, nu = X.E_GPa * 1e3, X.poisson
    Acoef = (pi * ri**2 - po * ro**2) / (ro**2 - ri**2)
    Bcoef = (pi - po) * ri**2 * ro**2 / (ro**2 - ri**2)
    sh_i, sr_i = Acoef + Bcoef / ri**2, -pi
    sh_o, sr_o = Acoef + Bcoef / ro**2, -po
    sa = Acoef  # axial (closed ends)
    vm_i = np.sqrt(0.5 * ((sh_i - sr_i)**2 + (sr_i - sa)**2 + (sa - sh_i)**2))
    tresca_i = np.maximum.reduce([abs(sh_i - sr_i), abs(sr_i - sa), abs(sa - sh_i)])
    ur_i = ri / E * (sh_i - nu * (sr_i + sa))
    ur_o = ro / E * (sh_o - nu * (sr_o + sa))
    Y = pd.DataFrame({"hoop_stress_inner_MPa": sh_i, "hoop_stress_outer_MPa": sh_o,
                      "axial_stress_MPa": sa, "von_mises_inner_MPa": vm_i,
                      "tresca_inner_MPa": tresca_i, "radial_disp_inner_um": ur_i * 1e3,
                      "radial_disp_outer_um": ur_o * 1e3,
                      "safety_factor_vm": X.yield_MPa / vm_i})
    return add_split(pd.concat([X, Y], axis=1))


# ---------------------------------------------------------------------------
# H1. Pin fin with convective tip (analytic 1-D fin theory)
# ---------------------------------------------------------------------------
def gen_fin(n=3000):
    b = {"diameter_mm": (2.0, 20.0), "length_mm": (10.0, 150.0), "k_W_mK": (15.0, 400.0),
         "h_W_m2K": (5.0, 500.0), "T_base_C": (50.0, 300.0), "T_inf_C": (0.0, 40.0)}
    X = lhs(b, n, log=("k_W_mK", "h_W_m2K"))
    D, L, k, h = X.diameter_mm / 1e3, X.length_mm / 1e3, X.k_W_mK, X.h_W_m2K
    th_b = X.T_base_C - X.T_inf_C
    P, Ac = np.pi * D, np.pi * D**2 / 4
    m = np.sqrt(h * P / (k * Ac)); M = np.sqrt(h * P * k * Ac) * th_b; hk = h / (m * k)
    den = np.cosh(m * L) + hk * np.sinh(m * L)
    q = M * (np.sinh(m * L) + hk * np.cosh(m * L)) / den
    T_tip = X.T_inf_C + th_b / den
    Af = P * L + Ac
    Y = pd.DataFrame({"heat_rate_W": q, "T_tip_C": T_tip,
                      "fin_efficiency": q / (h * Af * th_b),
                      "fin_effectiveness": q / (h * Ac * th_b),
                      "biot_number": h * (D / 4) / k})
    return add_split(pd.concat([X, Y], axis=1))


# ---------------------------------------------------------------------------
# H2. 2-D steady conduction in a plate with heat generation (finite differences)
# ---------------------------------------------------------------------------
def gen_plate(n=800, N=41):
    """Square plate W x W, uniform generation q''', convective top (h, T_inf),
    fixed temperatures left/right, insulated bottom."""
    b = {"width_mm": (20.0, 200.0), "k_W_mK": (5.0, 200.0), "q_gen_MW_m3": (0.001, 1.0),
         "h_top_W_m2K": (10.0, 2000.0), "T_inf_C": (15.0, 40.0),
         "T_left_C": (20.0, 150.0), "T_right_C": (20.0, 150.0)}
    X = lhs(b, n, log=("k_W_mK", "h_top_W_m2K", "q_gen_MW_m3"))
    rows = []
    for _, r in X.iterrows():
        W = r.width_mm / 1e3; dx = W / (N - 1); k = r.k_W_mK; q = r.q_gen_MW_m3 * 1e6
        idx = lambda i, j: j * N + i  # i: x, j: y (j=0 bottom)
        A = lil_matrix((N * N, N * N)); bvec = np.zeros(N * N)
        Bi = r.h_top_W_m2K * dx / k
        for j in range(N):
            for i in range(N):
                p = idx(i, j)
                if i == 0:
                    A[p, p] = 1; bvec[p] = r.T_left_C; continue
                if i == N - 1:
                    A[p, p] = 1; bvec[p] = r.T_right_C; continue
                if j == N - 1:  # convective top: half-cell energy balance
                    A[p, idx(i - 1, j)] = 0.5; A[p, idx(i + 1, j)] = 0.5
                    A[p, idx(i, j - 1)] = 1
                    A[p, p] = -(2 + Bi)
                    bvec[p] = -Bi * r.T_inf_C - q * dx**2 / (2 * k)
                    continue
                if j == 0:  # insulated bottom (mirror)
                    A[p, p] = -4; A[p, idx(i - 1, j)] = 1; A[p, idx(i + 1, j)] = 1
                    A[p, idx(i, j + 1)] = 2; bvec[p] = -q * dx**2 / k; continue
                A[p, p] = -4
                for ii, jj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    A[p, idx(ii, jj)] = 1
                bvec[p] = -q * dx**2 / k
        T = spsolve(A.tocsr(), bvec).reshape(N, N)  # [j, i]
        jmax, imax = np.unravel_index(T.argmax(), T.shape)
        # heat to convection at top (per unit depth), trapezoid
        q_top = r.h_top_W_m2K * np.trapezoid(T[-1, :] - r.T_inf_C, dx=dx)
        q_total = q * W * W
        rows.append((T.max(), T.mean(), T[N // 2, N // 2], imax * dx * 1e3, jmax * dx * 1e3,
                     q_top, q_total - q_top))
    Y = pd.DataFrame(rows, columns=["T_max_C", "T_mean_C", "T_center_C", "x_Tmax_mm",
                                    "y_Tmax_mm", "q_top_conv_W_per_m", "q_sides_W_per_m"])
    return add_split(pd.concat([X, Y], axis=1))


# ---------------------------------------------------------------------------
# H3. Counter-flow heat exchanger (effectiveness-NTU)
# ---------------------------------------------------------------------------
def gen_hx(n=3000):
    b = {"m_hot_kg_s": (0.05, 5.0), "m_cold_kg_s": (0.05, 5.0),
         "cp_hot_J_kgK": (1000.0, 4200.0), "cp_cold_J_kgK": (1000.0, 4200.0),
         "UA_W_K": (100.0, 20000.0), "T_hot_in_C": (80.0, 300.0), "T_cold_in_C": (5.0, 60.0)}
    X = lhs(b, n, log=("m_hot_kg_s", "m_cold_kg_s", "UA_W_K"))
    Ch, Cc = X.m_hot_kg_s * X.cp_hot_J_kgK, X.m_cold_kg_s * X.cp_cold_J_kgK
    Cmin, Cmax = np.minimum(Ch, Cc), np.maximum(Ch, Cc)
    Cr, NTU = Cmin / Cmax, X.UA_W_K / Cmin
    eps = np.where(np.isclose(Cr, 1, atol=1e-9), NTU / (1 + NTU),
                   (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr))))
    Q = eps * Cmin * (X.T_hot_in_C - X.T_cold_in_C)
    Tho, Tco = X.T_hot_in_C - Q / Ch, X.T_cold_in_C + Q / Cc
    lmtd = Q / X.UA_W_K  # identical to counter-flow LMTD, without 0/0 at eps -> 1
    Y = pd.DataFrame({"Q_kW": Q / 1e3, "T_hot_out_C": Tho, "T_cold_out_C": Tco,
                      "effectiveness": eps, "NTU": NTU, "LMTD_K": lmtd})
    return add_split(pd.concat([X, Y], axis=1))


if __name__ == "__main__":
    jobs = {"aero_naca4_airfoil_panel.csv": gen_airfoil,
            "aero_finite_wing_liftingline.csv": gen_wing,
            "struct_10bar_truss_fem.csv": gen_truss,
            "struct_thick_cylinder_lame.csv": gen_cylinder,
            "heat_pin_fin.csv": gen_fin,
            "heat_2d_plate_conduction_fdm.csv": gen_plate,
            "heat_counterflow_hx_entu.csv": gen_hx}
    for f, fn in jobs.items():
        df = fn()
        df.to_csv(os.path.join(OUT, f), index=False, float_format="%.6g")
        print(f, df.shape)
