"""Self-checks of the independent reference (no Strata involved).

1. homogeneous medium: K0 = K3 = mu_r e/(2 j kz), K4 = (eps0/eps) e/(2 j kz), K1 = K2 = 0
2. PEC ground: image forms, K2 = 0
3. TL reciprocity V_v(z, z') = -I_i(z', z) in random lossy, magnetic stacks
4. the E-field identities of formulation C, with z and z' derivatives taken by finite differences of K4:
       Ez(x dipole) = -j w mu0 Gzx + (kx/(w eps0)) dK4/dz,         Gzx = -j kx K2
       Ex(z dipole) = -w mu0 kx K1 - (kx/(w eps0)) dK4/dz'
       Ez(z dipole) = -j w mu0 K3 - (1/(j w eps0)) d2K4/dz dz'
   where the left-hand sides come straight from the TL fields (E_z = krho I^e/(w eps_o)).
5. spatial homogeneous closed form through the Sommerfeld integrator.
"""
import sys
import numpy as np
from layered_ref import Stack, EPS0, MU0, kz_of, mgf_spatial

rng = np.random.default_rng(1)
worst = {}
where = {}
CTX = {}


def rec(name, err):
    err = float(err)
    if err >= worst.get(name, 0.0):
        worst[name] = err
        where[name] = dict(CTX)


def rel(a, b):
    return np.max(np.abs(a - b)/np.maximum(np.abs(b), 1e-300))


# 1. homogeneous medium
for f, er, mr, sig in [(1e9, 4.0, 1.0, 0.0), (3e9, 2.0, 2.0, 0.02), (1e6, 11.9, 1.0, 10.0)]:
    st = Stack([(-1e-3, 1e-3, er, 0.0, mr, sig)], top=(er, mr, sig, 0), bot=(er, mr, sig, 0), f=f)
    k = st.k(0)
    kr = np.array([0.1*abs(k), 0.9*abs(k) + 0.1j*abs(k), 3*abs(k), 50.0/1e-3 + 0j])
    for z, zp in [(0.3e-3, -0.2e-3), (-0.5e-3, 0.4e-3), (0.2e-3, 0.2e-3)]:
        K, c = st.spectral(kr, z, zp, 0, 0)
        kz = kz_of(k, kr)
        e = np.exp(-1j*kz*abs(z - zp))/(2j*kz)
        rec('homog K0', rel(K[0], mr*e)); rec('homog K3', rel(K[3], mr*e))
        rec('homog K4', rel(K[4], EPS0/st.eps[0]*e))
        rec('homog K1,K2', np.max(np.abs(K[1:3]))/np.max(np.abs(e)))

# 2. PEC ground under a homogeneous half-space (layer + same-material top half-space)
st = Stack([(0.0, 2e-3, 3.0, 0.0, 1.0, 0.0)], top=(3.0, 1.0, 0.0, 0), bot=(1.0, 1.0, 0.0, 1), f=5e9)
k = st.k(0)
kr = np.array([0.5*abs(k) + 0.2j*abs(k), 2*abs(k), 1e4 + 0j])
z, zp = 1.2e-3, 0.7e-3
K, c = st.spectral(kr, z, zp, 0, 0)
kz = kz_of(k, kr)
e0 = np.exp(-1j*kz*abs(z - zp))/(2j*kz)
ei = np.exp(-1j*kz*(z + zp))/(2j*kz)
rec('PEC K0', rel(K[0], e0 - ei)); rec('PEC K3', rel(K[3], e0 + ei)); rec('PEC K4', rel(K[4], (EPS0/st.eps[0])*(e0 - ei)))
rec('PEC K2', np.max(np.abs(K[2]))/np.max(np.abs(e0)))


def random_stack(f):
    n = rng.integers(1, 5)
    th = rng.uniform(0.2e-3, 1.5e-3, n)
    z = np.concatenate([[0.0], np.cumsum(th)])
    layers = []
    for i in range(n):
        layers.append((z[i], z[i + 1], rng.uniform(1, 12), -rng.uniform(0, 0.5), rng.choice([1.0, 1.0, 2.5]), rng.choice([0.0, 0.0, 0.05])))
    top = (rng.uniform(1, 4), 1.0, 0.0, int(rng.random() < 0.25))
    bot = (rng.uniform(1, 12), rng.choice([1.0, 3.0]), rng.choice([0.0, 0.1]), int(rng.random() < 0.3))
    return Stack(layers, top=top, bot=bot, f=f), z


def rand_point(st, z):
    # inside a layer or a half-space (if not PEC)
    choices = list(range(st.N))
    if not st.top_pec: choices.append(-1)
    if not st.bot_pec: choices.append(st.N)
    r = rng.choice(choices)
    if r == -1:
        return st.zmax[0] + rng.uniform(0.05, 1.0)*1e-3, r
    if r == st.N:
        return st.zmin[-1] - rng.uniform(0.05, 1.0)*1e-3, r
    lo, hi = st.zmin[r], st.zmax[r]
    return lo + rng.uniform(0.05, 0.95)*(hi - lo), r


for trial in range(40):
    f = 10**rng.uniform(8, 10.5)
    st, zz = random_stack(f)
    km = st.kmax()
    kr = np.array([km*rng.uniform(0.05, 2.0) + 1j*km*rng.uniform(0, 0.3) for _ in range(4)] + [km*rng.uniform(2, 20) + 0j])
    z, o = rand_point(st, zz)
    zp, s = rand_point(st, zz)
    if abs(z - zp) < 1e-6:
        continue
    CTX.update(trial=trial, f=f, z=z, zp=zp, s=s, o=o, N=st.N, top_pec=st.top_pec, bot_pec=st.bot_pec)
    # 3. reciprocity
    t1 = st.tl(kr, z, zp, s, o)
    t2 = st.tl(kr, zp, z, o, s)
    for pol in 'eh':
        rec('TL reciprocity Vv(z,zp) = -Ii(zp,z)', rel(t1['Vv_' + pol], -t2['Ii_' + pol]))
        rec('TL reciprocity Vi symmetric', rel(t1['Vi_' + pol], t2['Vi_' + pol]))
        rec('TL reciprocity Iv symmetric', rel(t1['Iv_' + pol], t2['Iv_' + pol]))
    # 4. E-field identities with finite differences of K4. Only krho values where the vector- and scalar-potential
    #    terms are of comparable size are used (otherwise the identity cannot see an error in K1, K3), and the error
    #    is measured against the larger of the two terms. Steps scale with 1/|kz| (Richardson-extrapolated).
    w = st.w
    eps_o = st.region(o)[0]
    eps_s = st.region(s)[0]
    ks = abs(st.k(s))
    krs = np.array([ks*rng.uniform(0.2, 1.5) + 1j*ks*rng.uniform(0.01, 0.3) for _ in range(4)])
    K, c = st.spectral(krs, z, zp, s, o)
    t1 = st.tl(krs, z, zp, s, o)
    kzmax = max(abs(st.k(s)), abs(st.k(o)), np.max(np.abs(krs)))
    if kzmax*abs(z - zp) < 0.05:
        # finite differences need steps << |z - zp| but >> 1e-4/|kz| (round-off); not possible here
        continue

    def K4(zz_, zpp_):
        return st.spectral(krs, zz_, zpp_, s, o)[0][4]

    def derivs(h):
        dz = (K4(z + h, zp) - K4(z - h, zp))/(2*h)
        dzp = (K4(z, zp + h) - K4(z, zp - h))/(2*h)
        dzdzp = (K4(z + h, zp + h) - K4(z + h, zp - h) - K4(z - h, zp + h) + K4(z - h, zp - h))/(4*h*h)
        return dz, dzp, dzdzp

    h0 = min(1e-2/kzmax, 0.01*abs(z - zp))
    d1, d2 = derivs(h0), derivs(h0/2)
    dz, dzp, dzdzp = [(4*b - a)/3 for a, b in zip(d1, d2)]
    kx = krs/np.sqrt(2)
    t_a = -1j*w*MU0*(-1j*kx*K[2]); t_p = kx/(w*EPS0)*dz
    rec('E identity Ez(x-dipole)', np.max(np.abs(t_a + t_p - (-kx*t1['Ii_e']/(w*eps_o)))/np.maximum(np.abs(t_a), np.abs(t_p))))
    t_a = -w*MU0*kx*K[1]; t_p = -kx/(w*EPS0)*dzp
    rec('E identity Ex(z-dipole)', np.max(np.abs(t_a + t_p - (-(kx/(w*eps_s))*t1['Vv_e']))/np.maximum(np.abs(t_a), np.abs(t_p))))
    t_a = -1j*w*MU0*K[3]; t_p = -dzdzp/(1j*w*EPS0)
    rec('E identity Ez(z-dipole)', np.max(np.abs(t_a + t_p - (-(krs*krs/(w*w*eps_o*eps_s))*t1['Iv_e']))/np.maximum(np.abs(t_a), np.abs(t_p))))

# 5. spatial homogeneous closed form
import time
if '--skip-spatial' in sys.argv:
    pass
else:
  st = Stack([(-1e-2, 1e-2, 4.0, -0.1, 1.5, 0.0)], top=(4.0, 1.5, 0.0, 0), bot=(4.0, 1.5, 0.0, 0), f=2e9, top_epsr_im=-0.1, bot_epsr_im=-0.1)
  k = st.k(0)
  for rho, z, zp in [(3e-3, 1e-3, 0.0), (1e-2, 0.0, 0.0), (1e-10, 4e-3, -2e-3), (0.0, 1e-3, 0.0)]:
    t0 = time.time()
    scal, curl, err = mgf_spatial(st, rho, z, zp, 0, 0)
    R = np.hypot(rho, z - zp)
    g = np.exp(-1j*k*R)/(4*np.pi*R)
    rec('spatial homog Gxx', rel(scal[0], 1.5*g))
    rec('spatial homog Gphi', rel(scal[4], g*EPS0/st.eps[0]))
    rec('spatial homog Gzz', rel(scal[3], 1.5*g))
    print('spatial point rho=%g dz=%g: %.1f s, Gxx err %.2e' % (rho, z - zp, time.time() - t0, rel(scal[0], 1.5*g)), flush=True)

bad = False
for kname, v in worst.items():
    tol = 1e-5 if kname.startswith('E identity') else 1e-9
    flag = 'ok  ' if v < tol else 'FAIL'
    bad |= v >= tol
    print('%s %-45s %.3e' % (flag, kname, v))
    if v >= tol:
        print('     at', where.get(kname))
sys.exit(1 if bad else 0)
