# Rocket Propulsion Analysis

A self-directed series of rocket propulsion simulations in Python, built from first principles and validated against reference data.

## Roadmap
1. **Quasi-1D isentropic nozzle flow solver** (in progress)
2. Chamber thermochemistry with NASA CEA (RocketCEA): propellant trade studies
3. Engine sizing tool and Rao bell nozzle contour
4. Regenerative cooling analysis (Bartz heat-transfer correlation)
5. Injector and feed system analysis
6. Axisymmetric nozzle CFD (SU2/OpenFOAM), validated against the 1D solver

## Milestone 1: quasi-1D isentropic nozzle solver
**Assumptions:** calorically perfect gas, steady, adiabatic, inviscid, 1D, frozen composition.

- **1.1 Area–Mach solver:** M from A/A* and γ, with a subsonic/supersonic flag; bracketed root finder (`scipy.optimize.brentq`) with justified brackets. Check: γ = 1.4, A/A* = 2.0 → M ≈ 0.306 / 2.197.
- **1.2 Isentropic ratios:** p/p₀, T/T₀, ρ/ρ₀ vs M and γ, checked against a standard table.
- **1.3 Geometry:** conical converging-diverging nozzle r(x) from throat radius, contraction ratio, expansion ratio, and half-angles (15° divergent). Bonus: circular-arc throat blend.
- **1.4 Flow field:** M(x), p(x), T(x), u(x), switching branches at the throat; stacked-subplot figure.
- **1.5 Performance:** mass flow, c*, exit M/p/u, C_F, thrust, Isp (sea level and vacuum), printed as a table with units.
- **1.6 Trade study:** C_F vs ε (2–100) at sea level, 50 kPa, and vacuum; mark and explain the optimum ε. Stretch: locate a normal shock inside an overexpanded nozzle.

**Test case**

| Pc | Tc | γ | Molar mass | Throat diameter | ε | Ambient pressure |
|---|---|---|---|---|---|---|
| 7.0 MPa | 3500 K | 1.20 | 22.0 kg/kmol | 50 mm | 10 | 101,325 Pa |

## Setup
```bash
python -m venv .venv
.venv/bin/pip install numpy scipy matplotlib
```

## References
- Sutton & Biblarz, *Rocket Propulsion Elements*
- Huzel & Huang, *Design of Liquid Propellant Rocket Engines* (NASA SP-125)
- Hill & Peterson, *Mechanics and Thermodynamics of Propulsion*
