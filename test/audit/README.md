# Strata audit test suite

Tests written during an independent audit of Strata (fork based on upstream `1dda5b9`). They are meant to fail on
the unmodified upstream code and to pass once the corresponding fixes are applied.

| File | Purpose |
|---|---|
| `unit_tests.cpp` | Assertion-based regression tests (layer management, Bessel zeros, quadrature, closed forms, spectral/quasistatic API state, DCIM, interpolation, floating-point hygiene). Each test runs in a forked child with a time limit, so crashes and endless loops are reported per test. Build target `strata_unit_tests`; `./strata_unit_tests [FILTER]`. |
| `strata_probe.cpp` | Command-line probe: builds a stack-up from a JSON/YAML job, evaluates `ComputeMGF`, `ComputeCurlMGF`, `ComputeQMGF`, spectral and quasistatic kernels, `FindLayer`, and an independent Sommerfeld integration (`refint`) of Strata's own spectral kernels. Build target `strata_probe`. |
| `reference/layered_ref.py` | Independent reference: formulation-C spectral kernels from a global transmission-line solution derived from Maxwell's equations, the true H-field kernel, and an independent Sommerfeld integrator. Nothing is taken from Strata's source. |
| `reference/selfcheck.py` | Checks of the reference itself: homogeneous and PEC closed forms, TL reciprocity, the formulation-C E-field identities (with finite-difference derivatives), spatial closed form. |
| `physics_tests.py` | Strata (through the probe) against the reference: spectral kernels for random lossy/magnetic/PEC stacks, and spatial MGF and curl for representative stack-ups. |
| `compare_builtin.py` | Compares the outputs of the upstream example programs with the reference outputs committed in `test/examples`. |

Sign convention found by the audit: `ComputeCurlMGF` with `MGF_GHJ` returns the negative of the H-field kernel
(H = -int G_HJ J); `MGF_GEM` returns the E-field kernel of a magnetic current. The physics tests compare accordingly.
