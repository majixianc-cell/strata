"""Independent reference for the Michalski-Zheng formulation-C Green's functions of a planar layered medium.

Nothing here is taken from Strata's source. The spectral kernels follow from a direct solution of the TE/TM
transmission-line (TL) equations that Maxwell's equations reduce to for each lateral wavenumber, as one global
linear system (every interface and the source plane at once, exponentials referenced so that none grows); Strata
instead uses recursive interface impedances and closed-form transfer factors.

Derivation (exp(+jwt), fields ~ exp(+j(kx x + ky y)), u = (kx, ky)/krho, v = z x u):
    E_t = V^e u + V^h v,   H_t = -I^h u + I^e v,   E_z = krho I^e/(w eps) - J_z/(j w eps),   H_z = -krho V^h/(w mu)
    dV/dz = -j kz Z I + v_src,  dI/dz = -j kz Y V + i_src,  Z^e = kz/(w eps), Z^h = w mu/kz
    sources: J_u, J_v -> shunt currents i^e = -J_u, i^h = -J_v;  J_z -> series voltage v^e = -krho J_z/(w eps).
V_i, I_i (V_v, I_v) denote the response to a unit shunt current (series voltage) source. Matching E and H of an x-
and a z-directed dipole with E = -jw A - grad(phi), A = mu0 G_A . p, phi = (1/(j w eps0)) p . grad' G_phi,
H = curl(A)/mu, and using d/dz' V_i = j kz Z V_v (z != z') gives, with s = source and o = observation layer,
    K0 = G_xx  = V_i^h/(j w mu0)
    K1 = G_xz  = (mu_s/mu0) (V_v^e - V_v^h)/krho^2           (spatial: cos(phi) S1[K1])
    K2 = G_zx  = (mu_o/mu0) (I_i^e - I_i^h)/krho^2           (spatial: cos(phi) S1[K2])
    K3 = G_zz  = (mu_o/mu0) I_v^e/(j w eps_s) + (mu_s/mu0) j kz_o (Z_o^e I_v^e - Z_o^h I_v^h)/krho^2
    K4 = G_phi = j w eps0 (V_i^e - V_i^h)/krho^2
and the magnetic field of the dipoles (true H-field kernel, H = int G_HJ J):
    G_HJ scalars in Strata's dyadic layout: c0 = (I_i^e - I_i^h)/krho^2 (order 2), c1 = -(I_i^e + I_i^h) (order 0),
    c2 = -I_v^e/(j w eps_s) (order 1), c3 = -V_i^h/(j w mu_o) (order 1).
In a homogeneous medium these reduce to K0 = K3 = mu_r e/(2 j kz), K4 = (eps0/eps) e/(2 j kz), K1 = K2 = 0.

Spatial transforms (inverse transform exp(+j k.r)): T_n[F] = (1/2pi) int F J_n(krho rho) krho dkrho,
S1[F] = T_1[krho F]; G_xx = T0[K0], G_xz = cos(phi) S1[K1], G_zx = cos(phi) S1[K2], G_zz = T0[K3], G_phi = T0[K4];
curl scalars: G0 = T2[krho^2 c0], G1 = T0[c1], G2 = S1[c2], G3 = S1[c3].
"""
import numpy as np
from scipy import special

EPS0 = 8.854187817e-12
MU0 = 4e-7*np.pi


def kz_of(k, kr):
    kz = np.sqrt(k*k - kr*kr + 0j)
    flip = (kz.imag > 0) | ((kz.imag == 0) & (kz.real < 0))
    return np.where(flip, -kz, kz)


class Stack:
    """layers: iterable of (zmin, zmax, epsr, epsr_im, mur, sigma); top/bot: (epsr, mur, sigma, pec)."""

    def __init__(self, layers, top=(1.0, 1.0, 0.0, 0), bot=(1.0, 1.0, 0.0, 0), f=1e9, top_epsr_im=0.0, bot_epsr_im=0.0):
        self.f = f
        self.w = 2*np.pi*f
        L = sorted(layers, key=lambda l: -l[1])
        for a, b in zip(L[:-1], L[1:]):
            assert abs(a[0] - b[1]) <= 1e-15*max(1.0, abs(a[0])), 'layers must be contiguous'
        self.zmax = np.array([l[1] for l in L])
        self.zmin = np.array([l[0] for l in L])
        w = self.w
        self.eps = np.array([EPS0*(l[2] + 1j*l[3]) - 1j*l[5]/w for l in L])
        self.mu = np.array([MU0*l[4] for l in L], dtype=complex)
        self.top_pec, self.bot_pec = bool(top[3]), bool(bot[3])
        self.eps_top = EPS0*(top[0] + 1j*top_epsr_im) - 1j*top[2]/w
        self.mu_top = MU0*top[1] + 0j
        self.eps_bot = EPS0*(bot[0] + 1j*bot_epsr_im) - 1j*bot[2]/w
        self.mu_bot = MU0*bot[1] + 0j
        self.N = len(L)

    def region(self, idx):
        """(eps, mu) of layer idx, -1 = top half-space, N = bottom half-space."""
        if idx == -1:
            return self.eps_top, self.mu_top
        if idx == self.N:
            return self.eps_bot, self.mu_bot
        return self.eps[idx], self.mu[idx]

    def find_layer(self, z):
        """Same convention as Strata: interface points belong to the layer above; the top surface to layer 0."""
        if z > self.zmax[0]:
            return -1
        if z < self.zmin[-1]:
            return self.N
        for i in range(self.N):
            if self.zmin[i] <= z <= self.zmax[i]:
                return i
        raise ValueError(z)

    def k(self, idx):
        e, m = self.region(idx)
        return self.w*np.sqrt(e*m)

    def kmax(self):
        ks = [abs(self.k(i)) for i in range(-1, self.N + 1)]
        return max(ks)

    # ------------------------------------------------------------------------------------------
    # Transmission-line solution
    # ------------------------------------------------------------------------------------------

    def _segments(self, zp, s):
        """Segments from top to bottom: (region index, lo, hi); the source region s is split at zp."""
        segs = []
        if not self.top_pec:
            segs.append((-1, self.zmax[0], np.inf))
        for i in range(self.N):
            segs.append((i, self.zmin[i], self.zmax[i]))
        if not self.bot_pec:
            segs.append((self.N, -np.inf, self.zmin[-1]))
        out = []
        split = None
        for j, (r, lo, hi) in enumerate(segs):
            if r == s and lo <= zp <= hi:
                out.append((r, zp, hi))
                split = len(out) - 1
                out.append((r, lo, zp))
            else:
                out.append((r, lo, hi))
        if split is None:
            raise ValueError('source point %g not in region %d' % (zp, s))
        return out, split

    def tl(self, kr, z, zp, s, o):
        """V and I at z (region o) for unit shunt-current and series-voltage sources at zp (region s).
        Returns dict with arrays of shape kr.shape for keys Vi_e, Ii_e, Vv_e, Iv_e, Vi_h, Ii_h, Vv_h, Iv_h."""
        kr = np.atleast_1d(np.asarray(kr, dtype=complex))
        segs, split = self._segments(zp, s)
        w = self.w
        out = {}
        nseg = len(segs)
        # unknown layout: per segment [a, b] (a: up-going referenced at lo, b: down-going referenced at hi)
        has_a = [np.isfinite(lo) for (_, lo, hi) in segs]
        has_b = [np.isfinite(hi) for (_, lo, hi) in segs]
        col = []
        n = 0
        for j in range(nseg):
            ca = n if has_a[j] else None
            n += has_a[j]
            cb = n if has_b[j] else None
            n += has_b[j]
            col.append((ca, cb))
        nk = kr.size
        # observation segment: the segment of region o that contains z (closest if on a boundary)
        cand = [j for j, (r, lo, hi) in enumerate(segs) if r == o and lo <= z <= hi]
        if not cand:
            raise ValueError('observation point %g not in region %d' % (z, o))
        # z on the source plane: average of both sides (the currents jump there; Strata uses sign(0) = 0)
        cand = cand[:2] if (len(cand) > 1 and z == zp) else cand[:1]
        for pol in ('e', 'h'):
            kzs, Zs = [], []
            for (r, lo, hi) in segs:
                e, m = self.region(r)
                kz = kz_of(w*np.sqrt(e*m), kr)
                kzs.append(kz)
                Zs.append(kz/(w*e) if pol == 'e' else w*m/kz)
            A = np.zeros((nk, n, n), dtype=complex)
            B = np.zeros((nk, n, 2), dtype=complex)  # rhs: shunt current, series voltage
            row = 0
            if self.top_pec:
                ca, cb = col[0]
                lo, hi = segs[0][1], segs[0][2]
                # V(hi) = a e^{-jkz(hi-lo)} + b = 0
                A[:, row, ca] = np.exp(-1j*kzs[0]*(hi - lo))
                A[:, row, cb] = 1.0
                row += 1
            for j in range(nseg - 1):
                zb = segs[j][1]
                (ua, ub), (la, lb) = col[j], col[j + 1]
                hu = segs[j][2] - segs[j][1]
                hl = segs[j + 1][2] - segs[j + 1][1]
                eu_a = 1.0
                eu_b = np.exp(-1j*kzs[j]*hu) if np.isfinite(hu) else 0.0
                el_a = np.exp(-1j*kzs[j + 1]*hl) if np.isfinite(hl) else 0.0
                el_b = 1.0
                ratio = Zs[j]/Zs[j + 1]
                # V continuity: V_above(zb) - V_below(zb) = jumpV
                if ua is not None: A[:, row, ua] += eu_a
                if ub is not None: A[:, row, ub] += eu_b
                if la is not None: A[:, row, la] -= el_a
                if lb is not None: A[:, row, lb] -= el_b
                # I continuity scaled by Z_above: Z_u (I_above - I_below) = Z_u jumpI
                r2 = row + 1
                if ua is not None: A[:, r2, ua] += eu_a
                if ub is not None: A[:, r2, ub] -= eu_b
                if la is not None: A[:, r2, la] -= ratio*el_a
                if lb is not None: A[:, r2, lb] += ratio*el_b
                if j == split:
                    B[:, row, 1] = 1.0          # series voltage: V(zp+) - V(zp-) = 1
                    B[:, r2, 0] = Zs[j]         # shunt current: I(zp+) - I(zp-) = 1
                row += 2
            if self.bot_pec:
                ca, cb = col[-1]
                lo, hi = segs[-1][1], segs[-1][2]
                A[:, row, ca] = 1.0
                A[:, row, cb] = np.exp(-1j*kzs[-1]*(hi - lo))
                row += 1
            assert row == n, (row, n)
            X = np.linalg.solve(A, B)
            V = 0.0
            I = 0.0
            for jo in cand:
                ca, cb = col[jo]
                lo, hi = segs[jo][1], segs[jo][2]
                ea = np.exp(-1j*kzs[jo]*(z - lo))[:, None] if ca is not None else 0.0
                eb = np.exp(1j*kzs[jo]*(z - hi))[:, None] if cb is not None else 0.0
                a = X[:, ca, :] if ca is not None else 0.0
                b = X[:, cb, :] if cb is not None else 0.0
                V = V + (a*ea + b*eb)/len(cand)
                I = I + (a*ea - b*eb)/Zs[jo][:, None]/len(cand)
            out['Vi_' + pol], out['Vv_' + pol] = V[:, 0], V[:, 1]
            out['Ii_' + pol], out['Iv_' + pol] = I[:, 0], I[:, 1]
            out['kz_o'] = kzs[cand[0]]
            out['Z_o_' + pol] = Zs[cand[0]]
        return out

    # ------------------------------------------------------------------------------------------
    # Formulation-C spectral kernels (Strata normalisation)
    # ------------------------------------------------------------------------------------------

    def spectral(self, kr, z, zp, s=None, o=None):
        """K[0..4] and the true H-field curl scalars c[0..3]; arrays of shape (5, nk) and (4, nk)."""
        s = self.find_layer(zp) if s is None else s
        o = self.find_layer(z) if o is None else o
        kr = np.atleast_1d(np.asarray(kr, dtype=complex))
        t = self.tl(kr, z, zp, s, o)
        w = self.w
        eps_s, mu_s = self.region(s)
        eps_o, mu_o = self.region(o)
        k2 = kr*kr
        K = np.empty((5, kr.size), dtype=complex)
        K[0] = t['Vi_h']/(1j*w*MU0)
        K[1] = (mu_s/MU0)*(t['Vv_e'] - t['Vv_h'])/k2
        K[2] = (mu_o/MU0)*(t['Ii_e'] - t['Ii_h'])/k2
        K[3] = (mu_o/MU0)*t['Iv_e']/(1j*w*eps_s) + (mu_s/MU0)*1j*t['kz_o']*(t['Z_o_e']*t['Iv_e'] - t['Z_o_h']*t['Iv_h'])/k2
        K[4] = 1j*w*EPS0*(t['Vi_e'] - t['Vi_h'])/k2
        c = np.empty((4, kr.size), dtype=complex)
        c[0] = (t['Ii_e'] - t['Ii_h'])/k2
        c[1] = -(t['Ii_e'] + t['Ii_h'])
        c[2] = -t['Iv_e']/(1j*w*eps_s)
        c[3] = -t['Vi_h']/(1j*w*mu_o)
        return K, c


# ----------------------------------------------------------------------------------------------
# Sommerfeld integration (independent of Strata's path, quadrature and extrapolation)
# ----------------------------------------------------------------------------------------------

_xk, _wk = None, None


def _gk21():
    """Gauss-Kronrod 10/21 nodes and weights on [-1, 1]."""
    global _xk, _wk
    if _xk is None:
        xk = np.array([0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
                       0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
                       0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
                       0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
                       0.294392862701460198131126603103866, 0.148874338981631210884826001129720, 0.0])
        wk = np.array([0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
                       0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
                       0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
                       0.123491976262065851077600525452390, 0.134709217311473325928054001771707,
                       0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
                       0.149445554002916905664936468389821])
        wg = np.array([0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
                       0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
                       0.295524224714752870173892994651338])
        x = np.concatenate([-xk[:-1], xk[::-1]])
        wkk = np.concatenate([wk[:-1], wk[::-1]])
        wgg = np.zeros(21)
        # Gauss nodes are xk[1], xk[3], ..., xk[9] (and their negatives)
        gpos = {1: 0, 3: 1, 5: 2, 7: 3, 9: 4}
        for i, xv in enumerate(x):
            j = int(np.argmin(np.abs(xk - abs(xv))))
            if j in gpos:
                wgg[i] = wg[gpos[j]]
        _xk, _wk = x, (wkk, wgg)
    return _xk, _wk


def _group_tol(L1, groups, rtol):
    """Absolute tolerance per component: rtol times the largest L1 norm within its group (components of one group
    share a physical scale; an identically zero component then needs no relative accuracy of its own)."""
    tol = np.empty_like(L1)
    for g in groups:
        tol[g] = rtol*np.max(L1[g])
    return np.maximum(tol, 1e-300)


def gk_adaptive(f, a, b, rtol=1e-12, groups=None, maxpanels=3000):
    """Adaptive GK21 for a vector-valued complex f(x: (n,)) -> (m, n) on a real interval [a, b]."""
    x, (wk, wg) = _gk21()

    def panel(lo, hi):
        c, h = 0.5*(lo + hi), 0.5*(hi - lo)
        v = f(c + h*x)
        I = (v*wk).sum(axis=-1)*h
        G = (v*wg).sum(axis=-1)*h
        return I, np.abs(I - G)

    I0, E0 = panel(a, b)
    m = I0.size
    groups = groups or [list(range(m))]
    bounds = [(a, b)]
    Is, Es = [I0], [E0]
    for _ in range(maxpanels):
        Iarr, Earr = np.array(Is), np.array(Es)
        tol = _group_tol(np.abs(Iarr).sum(axis=0), groups, rtol)
        if np.all(Earr.sum(axis=0) <= tol):
            break
        score = (Earr/tol).max(axis=1)
        j = int(np.argmax(score))
        lo, hi = bounds[j]
        mid = 0.5*(lo + hi)
        I1, E1 = panel(lo, mid)
        I2, E2 = panel(mid, hi)
        bounds[j:j+1] = [(lo, mid), (mid, hi)]
        Is[j:j+1] = [I1, I2]
        Es[j:j+1] = [E1, E2]
    Iarr, Earr = np.array(Is), np.array(Es)
    return Iarr.sum(axis=0), Earr.sum(axis=0)


def wynn(seq):
    """Wynn epsilon extrapolation of a list of complex partial sums; returns the best estimate and an error."""
    n = len(seq)
    e = [[0.0]*(n + 1) for _ in range(n + 1)]
    for i in range(n):
        e[i][1] = seq[i]
    best, err = seq[-1], abs(seq[-1] - seq[-2]) if n > 1 else np.inf
    for k in range(2, n + 1):
        for i in range(n - k + 1):
            d = e[i + 1][k - 1] - e[i][k - 1]
            if d == 0:
                e[i][k] = np.inf
            else:
                e[i][k] = e[i + 1][k - 2] + 1.0/d
        if k % 2 == 1 and n - k >= 1:
            cand = e[n - k][k]
            prev = e[n - k - 1][k] if n - k - 1 >= 0 else None
            if prev is not None and np.isfinite(cand) and np.isfinite(prev):
                er = abs(cand - prev)
                if er < err:
                    best, err = cand, er
    return best, err


def sommerfeld(kernel, rho, orders, kmax, d, rtol=1e-11, groups=None):
    """int_0^inf kernel(kr)[c] J_nu(kr rho) kr^p dkr/(2 pi) for each row c with (nu, p) = orders[c].
    kernel: kr (complex array) -> (m, n) values. Path: half ellipse to A = 1.5 kmax, then the real axis (panels, then
    partitions with Wynn's epsilon algorithm). Returns the values and an error estimate per component."""
    orders = list(orders)
    m = len(orders)
    groups = groups or [list(range(m))]
    A = 1.5*kmax
    H = 0.4*A if rho == 0 else min(0.4*A, 1.0/rho)

    def integrand(kr):
        v = kernel(kr)
        out = np.empty_like(v)
        for c, (nu, p) in enumerate(orders):
            out[c] = v[c]*special.jv(nu, kr*rho)*kr**p/(2*np.pi)
        return out

    def fe(th):
        kr = 0.5*A*(1 - np.cos(th)) + 1j*H*np.sin(th)
        dk = 0.5*A*np.sin(th) + 1j*H*np.cos(th)
        return integrand(kr)*dk

    total = np.zeros(m, dtype=complex)
    err = np.zeros(m)
    for sseg in range(8):
        I, e = gk_adaptive(fe, np.pi*sseg/8, np.pi*(sseg + 1)/8, rtol=rtol, groups=groups)
        total += I
        err += e

    def fr(x):
        return integrand(x + 0j)

    def scale_of(v):
        sc = np.empty(m)
        for g in groups:
            sc[g] = np.max(np.abs(v[g]))
        return np.maximum(sc, 1e-300)

    left = A
    if rho == 0:
        if d <= 0:
            raise ValueError('rho = 0 and z = zp: singular')
        stop = max(A, 45.0/d)
        width = max(A, 1.0/d)
        while left < stop:
            I, e = gk_adaptive(fr, left, left + width, rtol=rtol, groups=groups)
            total += I
            err += e
            left += width
            width *= 1.5
        return total, err
    q = np.pi/rho
    target = A
    if d > 0 and 45.0/d < A + 200*q:
        target = max(A, 45.0/d)
    nu0 = max(nu for nu, _ in orders)
    sidx = np.ceil(target*rho/np.pi - nu0/2 + 0.25)
    first = (sidx + nu0/2 - 0.25)*np.pi/rho
    width = min(max(A, 1.0/d) if d > 0 else A, q)
    while left < first:
        right = min(first, left + width)
        I, e = gk_adaptive(fr, left, right, rtol=rtol, groups=groups)
        total += I
        err += e
        left = right
        if width < 0.5*q:
            width *= 1.5
    sums = []
    s = np.zeros(m, dtype=complex)
    est = s
    for k in range(600):
        I, e = gk_adaptive(fr, left, left + q, rtol=rtol, groups=groups)
        err += e
        left += q
        s = s + I
        sums.append(s.copy())
        sc = scale_of(total + s)
        if k > 2 and np.all(np.abs(I) <= 1e-17*sc):
            est = s
            break
        if k >= 6 and k % 2 == 0:
            window = sums[-40:]
            ests, errs = zip(*[wynn([x[c] for x in window]) for c in range(m)])
            est = np.array(ests)
            if np.all(np.array(errs) <= 1e-12*scale_of(total + est)):
                err += np.array(errs)
                break
    else:
        raise RuntimeError('tail did not converge')
    return total + est, err


def mgf_spatial(stack, rho, z, zp, s=None, o=None, rtol=1e-11):
    """Spatial formulation-C scalars [Gxx, Gxz/cos, Gzx/cos, Gzz, Gphi] and curl scalars [G0..G3] (true H kernel)."""
    s = stack.find_layer(zp) if s is None else s
    o = stack.find_layer(z) if o is None else o

    def kernel(kr):
        K, c = stack.spectral(kr, z, zp, s, o)
        return np.vstack([K, c])

    orders = [(0, 1), (1, 2), (1, 2), (0, 1), (0, 1), (2, 3), (0, 1), (1, 2), (1, 2)]
    v, err = sommerfeld(kernel, rho, orders, stack.kmax(), abs(z - zp), rtol=rtol, groups=[[0, 1, 2, 3, 4], [5, 6, 7, 8]])
    return v[:5], v[5:], float(np.max(err/np.maximum(np.abs(v), 1e-300)))


def assemble(scal, curl, phi):
    """Dyadic (row-major, 9) and G_phi from the scalars, Strata's layout; curl uses Strata's assembly."""
    c, s = np.cos(phi), np.sin(phi)
    c2, s2 = np.cos(2*phi), np.sin(2*phi)
    G = np.array([scal[0], 0, scal[1]*c, 0, scal[0], scal[1]*s, scal[2]*c, scal[2]*s, scal[3]], dtype=complex)
    g0, g1, g2, g3 = curl
    Hc = np.array([-g0*s2/2, (g0*c2 - g1)/2, g2*s, (g0*c2 + g1)/2, g0*s2/2, -g2*c, -g3*s, g3*c, 0], dtype=complex)
    return G, scal[4], Hc


def homogeneous_closed_form(epsr_c, mur, k, R):
    g = np.exp(-1j*k*R)/(4*np.pi*R)
    return mur*g, g/epsr_c
