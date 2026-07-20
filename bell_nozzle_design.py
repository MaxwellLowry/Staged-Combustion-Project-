"""
Bell (Rao-type) Supersonic Nozzle Contour Generator
====================================================
Sizes a converging-diverging nozzle and generates its contour for expanding
hot preburner exhaust to supersonic conditions against ambient backpressure.

Method:
1. Isentropic compressible flow relations size the throat-to-exit area ratio
   needed to fully expand from chamber (stagnation) pressure to ambient.
2. Rao's parabolic approximation defines the diverging (bell) section: a
   circular arc of radius 0.382*Rt downstream of the throat, followed by a
   parabola (quadratic Bezier) out to the exit plane.
3. A conical + circular-arc convergent section is added upstream of the
   throat to close the contour.

theta_n/theta_e (initial/exit divergence angles) are normally read off Rao's
percent-bell-length charts; here they're approximated with a coarse
interpolation table -- refine with actual chart/MOC data for production use.

Units: mm for lengths, bar for pressure. Sizing is ratio-based up until Rt
is computed, so swap units as needed as long as Pc/Pa stay consistent.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# 1. USER INPUTS
# ---------------------------------------------------------------------------
Pc = 175.1                  # chamber (upstream stagnation) pressure [bar, abs]
Pa = 1.01325                 # ambient back pressure [bar, abs]
gamma = 1.3135                # ratio of specific heats of the exhaust gas
percent_bell = 0.80         # fraction of equivalent 15 deg conical nozzle length
conv_half_angle_deg = 30.0  # convergent section half-angle [deg]
Rc_mm = 70                 # combustion chamber radius [mm]

# --- throat sizing inputs (mass-flow method) --------------------------------
mdot_kg_s = 7.2349              # total propellant mass flow [kg/s]
Tc_K = 800.0                 # chamber stagnation temperature [K]
MW_g_mol = 31.921               # combustion product molecular weight [g/mol]
c_star_override = None       # set c* [m/s] directly to bypass Tc/MW calc

# ---------------------------------------------------------------------------
# 2. THROAT SIZING from mass flow (choked-flow relation via c*)
# ---------------------------------------------------------------------------
Ru = 8314.46  # universal gas constant [J/(kmol*K)]
Pc_pa = Pc * 1e5

if c_star_override is not None:
    c_star = c_star_override
else:
    R_specific = Ru / MW_g_mol  # J/(kg*K)
    c_star = np.sqrt(R_specific * Tc_K / gamma) * (
        (gamma + 1) / 2
    ) ** ((gamma + 1) / (2 * (gamma - 1)))

At_mm2 = mdot_kg_s * c_star / Pc_pa * 1e6   # m^2 -> mm^2
Rt_mm = np.sqrt(At_mm2 / np.pi)

print(f"c* (characteristic velocity):  {c_star:.1f} m/s")
print(f"Throat area:                   {At_mm2:.2f} mm^2")
print(f"Throat radius:                 {Rt_mm:.3f} mm  (throat dia {2*Rt_mm:.3f} mm)")
print(f"Contraction ratio Rc/Rt:       {(Rc_mm/Rt_mm):.2f}")

# ---------------------------------------------------------------------------
# 3. ISENTROPIC SIZING: exit Mach number & area ratio for full expansion
# ---------------------------------------------------------------------------
def mach_from_pressure_ratio(p0_over_p, g):
    """Solve p0/p = (1 + (g-1)/2 * M^2)^(g/(g-1)) for M."""
    f = lambda M: (1 + (g - 1) / 2 * M ** 2) ** (g / (g - 1)) - p0_over_p
    return brentq(f, 1.0001, 30.0)


def area_ratio_from_mach(M, g):
    """A/At from 1-D isentropic area-Mach relation."""
    return (1 / M) * ((2 / (g + 1)) * (1 + (g - 1) / 2 * M ** 2)) ** (
        (g + 1) / (2 * (g - 1))
    )


Me = mach_from_pressure_ratio(Pc / Pa, gamma)
eps = area_ratio_from_mach(Me, gamma)          # Ae / At
Re_mm = Rt_mm * np.sqrt(eps)

print(f"Design exit Mach number:      {Me:.3f}")
print(f"Required area ratio Ae/At:    {eps:.3f}")
print(f"Exit radius:                  {Re_mm:.3f} mm  (exit dia {2*Re_mm:.3f} mm)")

# ---------------------------------------------------------------------------
# 4. theta_n / theta_e -- initial & exit divergence angles (80%-bell table)
#    Coarse interpolation of published Rao-chart values; override manually
#    if you have exact chart/MOC-derived numbers for your area ratio.
# ---------------------------------------------------------------------------
eps_pts = [5, 10, 25, 50, 100]
theta_n_pts = [28.0, 30.0, 32.0, 33.0, 34.0]     # deg
theta_e_pts = [13.0, 10.5, 8.5, 7.5, 7.0]        # deg (80% bell)

theta_n_deg = float(np.interp(eps, eps_pts, theta_n_pts))
theta_e_deg = float(np.interp(eps, eps_pts, theta_e_pts))
theta_n = np.radians(theta_n_deg)
theta_e = np.radians(theta_e_deg)

print(f"theta_n (approx):             {theta_n_deg:.2f} deg")
print(f"theta_e (approx):             {theta_e_deg:.2f} deg")

# ---------------------------------------------------------------------------
# 5. NOZZLE LENGTH via the percent-of-equivalent-15deg-cone method
# ---------------------------------------------------------------------------
Rn = 0.382 * Rt_mm   # throat-downstream arc radius (Rao standard constant)
alpha15 = np.radians(15.0)
Lc15 = (Rt_mm * (np.sqrt(eps) - 1) + Rn * (1 / np.cos(alpha15) - 1)) / np.tan(alpha15)
Ln = percent_bell * Lc15
print(f"Nozzle length (throat->exit): {Ln:.3f} mm")

# ---------------------------------------------------------------------------
# 6. CONTOUR GENERATION
# ---------------------------------------------------------------------------
def divergent_contour(Rt, Rn, theta_n, theta_e, Ln, Re, n_arc=60, n_bell=200):
    # circular arc immediately downstream of the throat: theta = 0 -> theta_n
    th = np.linspace(0, theta_n, n_arc)
    x_arc = Rn * np.sin(th)
    y_arc = Rt + Rn * (1 - np.cos(th))
    xN, yN = x_arc[-1], y_arc[-1]

    # Bezier control point = intersection of the tangent lines at N and E
    xE, yE = Ln, Re
    C1 = yN - xN * np.tan(theta_n)
    C2 = yE - xE * np.tan(theta_e)
    Qx = (C2 - C1) / (np.tan(theta_n) - np.tan(theta_e))
    Qy = C1 + Qx * np.tan(theta_n)

    t = np.linspace(0, 1, n_bell)
    x_bell = (1 - t) ** 2 * xN + 2 * (1 - t) * t * Qx + t ** 2 * xE
    y_bell = (1 - t) ** 2 * yN + 2 * (1 - t) * t * Qy + t ** 2 * yE

    x = np.concatenate([x_arc, x_bell[1:]])
    y = np.concatenate([y_arc, y_bell[1:]])
    return x, y, (Qx, Qy)


def convergent_contour(Rt, Rc, half_angle_deg, Rn_conv_mult=1.5, n=60):
    """Circular arc + straight cone upstream of the throat (negative x)."""
    half_angle = np.radians(half_angle_deg)
    Rn_conv = Rn_conv_mult * Rt
    th = np.linspace(0, half_angle, n)
    x_arc = -Rn_conv * np.sin(th)
    y_arc = Rt + Rn_conv * (1 - np.cos(th))
    x_tan, y_tan = x_arc[-1], y_arc[-1]

    y_chamber = Rc
    dx = (y_chamber - y_tan) / np.tan(half_angle)
    x_start = x_tan - dx
    x_cone = np.array([x_start, x_tan])
    y_cone = np.array([y_chamber, y_tan])

    x = np.concatenate([x_cone, x_arc[::-1][1:]])
    y = np.concatenate([y_cone, y_arc[::-1][1:]])
    return x, y


x_div, y_div, Q = divergent_contour(Rt_mm, Rn, theta_n, theta_e, Ln, Re_mm)
x_conv, y_conv = convergent_contour(Rt_mm, Rc_mm, conv_half_angle_deg)

x_full = np.concatenate([x_conv, x_div])
y_full = np.concatenate([y_conv, y_div])

# ---------------------------------------------------------------------------
# 7. SAVE OUTPUTS: CSV (raw coordinates), PNG (plot), DXF (CAD import)
# ---------------------------------------------------------------------------
with open("nozzle_contour.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x_mm", "y_mm"])
    for xx, yy in zip(x_full, y_full):
        w.writerow([f"{xx:.5f}", f"{yy:.5f}"])

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(x_full, y_full, "b-", lw=2)
ax.plot(x_full, -y_full, "b-", lw=2)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls=":")
ax.set_aspect("equal")
ax.set_xlabel("Axial distance from throat [mm]")
ax.set_ylabel("Radius [mm]")
ax.set_title(
    f"Bell Nozzle Contour  (Ae/At={eps:.2f}, Me={Me:.2f}, "
    f"{percent_bell*100:.0f}% bell, Rt={Rt_mm} mm)"
)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("nozzle_contour.png", dpi=150)
print("Saved: nozzle_contour.csv, nozzle_contour.png")

try:
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    upper = list(zip(x_full, y_full))
    lower = list(zip(x_full, -y_full))
    msp.add_lwpolyline(upper)
    msp.add_lwpolyline(lower)
    doc.saveas("nozzle_contour.dxf")
    print("Saved: nozzle_contour.dxf")
except ImportError:
    print("(ezdxf not installed -- skipping DXF export; `pip install ezdxf` to enable)")