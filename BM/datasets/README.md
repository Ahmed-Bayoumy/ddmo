# Engineering MIMO Benchmark Datasets for Surrogate Models

Seven CSV datasets (aerodynamics, structural mechanics, heat transfer), each generated
by a deterministic physics model and sampled with a Latin hypercube design (seed 2026).
Every file has:

- `sample_id` – row index
- input columns (listed first), then output columns
- `split` – a fixed ~80/20 `train`/`test` split, so results are comparable across models

Data are noise-free. To test robustness, add your own noise to outputs (e.g. 1–5 % Gaussian).
`generate_datasets.py` regenerates everything and can produce more samples or new ranges.

| File | Rows | Inputs | Outputs | Model |
|---|---|---|---|---|
| aero_naca4_airfoil_panel.csv | 1500 | 6 | 5 | Linear-vortex panel method + Prandtl–Glauert + viscous drag estimate |
| aero_finite_wing_liftingline.csv | 2000 | 6 | 5 | Prandtl lifting-line (Glauert series, 24 modes) |
| struct_10bar_truss_fem.csv | 3000 | 14 | 15 | 2-D truss FEM (direct stiffness) |
| struct_thick_cylinder_lame.csv | 3000 | 7 | 8 | Lamé thick-walled cylinder, closed ends |
| heat_pin_fin.csv | 3000 | 6 | 5 | 1-D fin theory, convective tip |
| heat_2d_plate_conduction_fdm.csv | 800 | 7 | 7 | 2-D steady conduction, finite differences (41×41) |
| heat_counterflow_hx_entu.csv | 3000 | 7 | 6 | Counter-flow heat exchanger, ε-NTU |

(`*` = sampled log-uniformly)

---

## 1. aero_naca4_airfoil_panel.csv
**Inputs:** `max_camber_m` [0–0.06], `camber_pos_p` [0.2–0.6], `thickness_t` [0.06–0.20] (NACA 4-digit, fraction of chord),
`alpha_deg` [−4–8], `mach` [0.05–0.5], `reynolds`* [2e5–1e7]
**Outputs:** `CL`, `CD`, `CM_c4` (about quarter chord), `L_over_D`, `Cp_min`

160-panel linear-strength vortex method with Kutta condition, Prandtl–Glauert correction.
CD = 2·Cf·FF·(1+0.06·CL²), with Schlichting turbulent Cf and a thickness form factor.
Only subcritical cases are kept (Cp_min > Cp_crit).
**Validated:** NACA 0012 at 5° → CL 0.599; NACA 2412 at 0° → CL 0.258, CM −0.055 (matches XFOIL inviscid).
**Limits:** inviscid lift, so no stall; CD is an engineering estimate, not a boundary-layer solve.

## 2. aero_finite_wing_liftingline.csv
**Inputs:** `aspect_ratio` [4–14], `taper_ratio` [0.2–1], `twist_deg` [−6–0] (linear washout, tip),
`alpha_root_deg` [−2–10], `airfoil_cl_alpha_per_rad` [5.6–6.8], `alpha_L0_deg` [−4–0]
**Outputs:** `CL`, `CDi` (induced drag), `cl_section_max`, `eta_of_cl_max` (2y/b where section cl peaks, i.e. stall-onset location),
`root_bending_coeff` = M_root / (q·S·b/2)
**Validated:** AR 8 at 5°: taper 0.4 → e = 0.987; rectangular → e = 0.937 (textbook values).

## 3. struct_10bar_truss_fem.csv
The classic 10-bar cantilever truss (bays 9.144 m / 360 in, nodes 5 and 6 pinned).
**Inputs:** `A1_cm2`…`A10_cm2` [0.645–225.8] (member areas), `E_GPa` [60–210],
`P2_kN`, `P4_kN` [200–600] (downward loads at nodes 2 and 4), `density_kg_m3` [2700–7850]
**Outputs:** `mass_kg`, `max_disp_mm`, `v_node2_mm`, `v_node4_mm`, `max_abs_stress_MPa`, `stress1_MPa`…`stress10_MPa` (tension +)
A good high-dimensional test (14 → 15). Small areas give very large stresses/displacements (heavy tails);
consider log-transforming or filtering if your application needs it.

## 4. struct_thick_cylinder_lame.csv
**Inputs:** `r_inner_mm` [20–200], `wall_ratio_ro_ri` [1.1–3], `p_internal_MPa` [5–150], `p_external_MPa` [0–30],
`E_GPa` [70–210], `poisson` [0.25–0.35], `yield_MPa` [250–900]
**Outputs:** `hoop_stress_inner_MPa`, `hoop_stress_outer_MPa`, `axial_stress_MPa`, `von_mises_inner_MPa`,
`tresca_inner_MPa`, `radial_disp_inner_um`, `radial_disp_outer_um`, `safety_factor_vm`
Exact closed-form solution, useful as a zero-model-error reference. `safety_factor_vm` is heavy-tailed.

## 5. heat_pin_fin.csv
**Inputs:** `diameter_mm` [2–20], `length_mm` [10–150], `k_W_mK`* [15–400], `h_W_m2K`* [5–500], `T_base_C` [50–300], `T_inf_C` [0–40]
**Outputs:** `heat_rate_W`, `T_tip_C`, `fin_efficiency`, `fin_effectiveness`, `biot_number`
Exact 1-D fin solution with convective tip.

## 6. heat_2d_plate_conduction_fdm.csv
Square plate W×W (per unit depth), uniform generation, fixed-T left/right edges,
convective top edge (h, T_inf), insulated bottom.
**Inputs:** `width_mm` [20–200], `k_W_mK`* [5–200], `q_gen_MW_m3`* [0.001–1], `h_top_W_m2K`* [10–2000], `T_inf_C` [15–40], `T_left_C`, `T_right_C` [20–150]
**Outputs:** `T_max_C`, `T_mean_C`, `T_center_C`, `x_Tmax_mm`, `y_Tmax_mm` (location of hottest point),
`q_top_conv_W_per_m` (heat to fluid), `q_sides_W_per_m` (net heat out through the fixed-T edges; negative = heat enters)
**Validated:** 41×41 vs 81×81 grid, T_max differs by < 0.02 °C.
`x_Tmax_mm`/`y_Tmax_mm` are non-smooth (they jump between locations), a deliberately hard target.

## 7. heat_counterflow_hx_entu.csv
**Inputs:** `m_hot_kg_s`*, `m_cold_kg_s`* [0.05–5], `cp_hot_J_kgK`, `cp_cold_J_kgK` [1000–4200], `UA_W_K`* [100–20000],
`T_hot_in_C` [80–300], `T_cold_in_C` [5–60]
**Outputs:** `Q_kW`, `T_hot_out_C`, `T_cold_out_C`, `effectiveness`, `NTU`, `LMTD_K`
Exact ε-NTU relations. Effectiveness saturates near 1 at high NTU.

---

## Suggested benchmarking practice
- Train on `split == "train"`, report on `split == "test"` only.
- Normalize inputs/outputs; log-transform inputs marked * and heavy-tailed outputs.
- Report per-output R², RMSE, and normalized max error; MIMO models often do well on average but miss one output.
- Learning curves: subsample the training set (e.g. 50, 100, 200, 500, all) to compare sample efficiency.
