"""Physics checks of Strata against the independent reference in reference/layered_ref.py.

    python3 physics_tests.py PROBE OUTDIR [--groups spec,spatial,methods,qs] [--workers N] [--quick]

Groups
  spec     spectral kernels K[0..4] and the curl kernels (G_EM, G_HJ) at random complex krho for random stacks
           (lossy, magnetic, PEC, half-spaces, same / different layers) -> relative error per component
  spatial  ComputeMGF (MGF_INTEGRATE, with and without quasistatic / homogeneous extraction) and ComputeCurlMGF
           against the reference Sommerfeld integrals on representative stack-ups (microwave, on-chip, thin layers,
           magnetic/lossy, PEC, high-contrast half-spaces), including near-axis, same-height, interface and
           half-space points
  methods  DCIM and interpolation against the same references (accuracy summary; gross failures flagged)
  qs       the quasistatic spectral kernel approaches the full kernel for large krho

Every result (inputs, Strata output, reference, errors) is written to OUTDIR/<group>.jsonl; OUTDIR/summary.json holds
the pass/fail counts per check. Strata's G_HJ is compared with -H (see README: Strata returns the negative of the
H-field kernel); G_EM is compared with the E-field kernel obtained from G_HJ by reciprocity.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent/'reference'))
from layered_ref import Stack, EPS0, MU0, mgf_spatial, assemble  # noqa: E402

FIND = -999


# ------------------------------------------------------------------------------------------------
# Probe driver
# ------------------------------------------------------------------------------------------------

def run_probe(probe, job, timeout=3600, threads=1):
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump(job, fh)
        path = fh.name
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1')
    t0 = time.time()
    try:
        r = subprocess.run([probe, path], capture_output=True, text=True, timeout=timeout, env=env)
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, out, err = 'timeout', (e.stdout or b'').decode() if isinstance(e.stdout, bytes) else (e.stdout or ''), ''
    finally:
        os.unlink(path)
    lines = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    res = {d['q']: d for d in lines}
    return dict(rc=rc, seconds=time.time() - t0, stderr=err[-2000:], init=res.get(-1, {}),
                results=[res.get(i, {'err': 'missing (rc=%s)' % rc}) for i in range(len(job.get('queries', [])))])


def cvec(v):
    return np.array([complex(a, b) for a, b in v])


# ------------------------------------------------------------------------------------------------
# Stack-ups
# ------------------------------------------------------------------------------------------------

def stack_job(layers, top, bot):
    return {'layers': [list(map(float, l)) for l in layers], 'top': list(top), 'bot': list(bot)}


STACKS = {
    # name: (layers [zmin, zmax, epsr, epsr_im, mur, sigma], top [epsr, mur, sigma, pec], bot, frequencies)
    'ling_jin_2000': ([(1.1e-3, 1.8e-3, 2.1, 0, 1, 0), (0.8e-3, 1.1e-3, 12.5, 0, 1, 0), (0.3e-3, 0.8e-3, 9.8, 0, 1, 0),
                       (0.0, 0.3e-3, 8.6, 0, 1, 0)], (1, 1, 0, 0), (1, 1, 0, 1), [30e9]),
    'two_layer': ([(0.0, 1e-3, 2.3, 0, 1, 0), (-1e-3, 0.0, 4.5, 0, 1, 0)], (1, 1, 0, 0), (6, 1, 0, 0), [1e9, 10e9]),
    'onchip_si': ([(5.5e-6, 6e-6, 7.0, 0, 1, 0), (3e-6, 5.5e-6, 3.9, 0, 1, 0), (1e-6, 3e-6, 4.2, 0, 1, 0), (0.0, 1e-6, 3.9, 0, 1, 0)],
                  (1, 1, 0, 0), (11.9, 1, 10.0, 0), [1e8, 1e9, 2e10]),
    'onchip_pec': ([(2e-6, 4e-6, 3.9, 0, 1, 0), (0.0, 2e-6, 11.9, 0, 1, 10.0)], (1, 1, 0, 0), (1, 1, 0, 1), [1e9]),
    'magnetic_lossy': ([(0.0, 2e-3, 3.0, -0.3, 2.5, 0.0), (-1e-3, 0.0, 6.0, 0, 1, 0.05)], (1, 1, 0, 0), (2, 1.5, 0, 0), [3e9]),
    'hi_eps_halfspace': ([(0.0, 0.2e-3, 4.0, 0, 1, 0)], (1, 1, 0, 0), (12.0, 1, 0, 0), [10e9]),
    'thin_nm': ([(25e-9, 30e-9, 7.0, 0, 1, 0), (5e-9, 25e-9, 25.0, 0, 1, 0), (0.0, 5e-9, 3.9, 0, 1, 0), (-1e-6, 0.0, 3.9, 0, 1, 0)],
                (1, 1, 0, 0), (11.9, 1, 10.0, 0), [1e9]),
    'pec_parallel_plate': ([(0.0, 1e-3, 2.2, 0, 1, 0)], (1, 1, 0, 1), (1, 1, 0, 1), [5e9]),
}


def ref_stack(name, f):
    layers, top, bot, _ = STACKS[name]
    return Stack(layers, top=top, bot=bot, f=f)


# ------------------------------------------------------------------------------------------------
# Spectral group
# ------------------------------------------------------------------------------------------------

def random_stack(rng):
    n = int(rng.integers(1, 5))
    th = rng.uniform(0.1, 1.5, n)*10.0**rng.uniform(-6, -3)
    zs = np.concatenate([[0.0], np.cumsum(th)])[::-1]
    layers = []
    for i in range(n):
        layers.append((float(zs[i + 1]), float(zs[i]), float(rng.uniform(1, 12)), float(-rng.uniform(0, 0.5)*(rng.random() < 0.5)),
                       float(rng.choice([1.0, 1.0, 2.5])), float(rng.choice([0.0, 0.0, 0.05, 10.0]))))
    top = (float(rng.uniform(1, 4)), 1.0, 0.0, int(rng.random() < 0.2))
    bot = (float(rng.uniform(1, 12)), float(rng.choice([1.0, 3.0])), float(rng.choice([0.0, 0.0, 10.0])), int(rng.random() < 0.3))
    return layers, top, bot


def point_in(rng, st, region, frac=None):
    if region == -1:
        return float(st.zmax[0] + rng.uniform(0.05, 1.0)*(st.zmax[0] - st.zmin[-1]))
    if region == st.N:
        return float(st.zmin[-1] - rng.uniform(0.05, 1.0)*(st.zmax[0] - st.zmin[-1]))
    lo, hi = st.zmin[region], st.zmax[region]
    return float(lo + (rng.uniform(0.05, 0.95) if frac is None else frac)*(hi - lo))


def spectral_cases(rng, n):
    cases = []
    for t in range(n):
        layers, top, bot = random_stack(rng)
        f = float(10**rng.uniform(6, 10.5))
        st = Stack(layers, top=top, bot=bot, f=f)
        regions = list(range(st.N)) + ([] if top[3] else [-1]) + ([] if bot[3] else [st.N])
        q = []
        for nq in range(7):
            s, o = int(rng.choice(regions)), int(rng.choice(regions))
            zp = point_in(rng, st, s)
            z = point_in(rng, st, o)
            if s == o and rng.random() < 0.2:
                z = zp
            if nq == 6 and st.N > 1:
                # source and observer on a common interface, assigned to the two adjacent layers
                s = int(rng.integers(0, st.N - 1))
                o = s + 1
                if rng.random() < 0.5:
                    s, o = o, s
                z = zp = float(st.zmin[min(s, o)])
            km = st.kmax()
            for kr in [km*rng.uniform(0.01, 2) + 1j*km*rng.uniform(0.001, 0.5), km*rng.uniform(1.5, 30),
                       rng.uniform(1, 30)/max(1e-9, st.zmax[0] - st.zmin[-1]) + 0j]:
                q.append(['spec', float(kr.real), float(kr.imag), z, zp, s, o])
        cases.append(dict(layers=layers, top=top, bot=bot, f=f, queries=q))
    return cases


def check_spectral(probe, case):
    job = {'stack': stack_job(case['layers'], case['top'], case['bot']), 'freq': case['f'], 'init': 0, 'settings': {},
           'queries': case['queries']}
    out = run_probe(probe, job)
    st = Stack(case['layers'], top=case['top'], bot=case['bot'], f=case['f'])
    rows = []
    for qq, r in zip(case['queries'], out['results']):
        _, kre, kim, z, zp, s, o = qq
        kr = np.array([complex(kre, kim)])
        row = dict(query=qq, f=case['f'])
        if 'err' in r:
            row['strata_err'] = r['err']
            rows.append(row)
            continue
        K, c = st.spectral(kr, z, zp, s, o)
        Kp, cp = st.spectral(kr, zp, z, o, s)          # swapped configuration for G_EM by reciprocity
        Ks = cvec(r['K'])
        ghj = cvec(r['GHJ'])
        gem = cvec(r['GEM'])
        Kr = K[:, 0]
        ghj_ref = -c[:, 0]
        gem_ref = np.array([-cp[0, 0], cp[1, 0], -cp[3, 0], -cp[2, 0]])

        def errs(a, b):
            scale = np.maximum(np.abs(b), 1e-9*np.max(np.abs(b)))
            return (np.abs(a - b)/np.maximum(scale, 1e-300)).tolist()
        row.update(K_err=errs(Ks, Kr), GHJ_err=errs(ghj, ghj_ref), GEM_err=errs(gem, gem_ref),
                   K=[[v.real, v.imag] for v in Ks], K_ref=[[v.real, v.imag] for v in Kr])
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------
# Spatial group
# ------------------------------------------------------------------------------------------------

def spatial_points(name, f, rng):
    st = ref_stack(name, f)
    H = st.zmax[0] - st.zmin[-1]
    lam = 2*np.pi/abs(st.k(0))
    pts = []
    N = st.N
    regions = list(range(N)) + ([] if st.top_pec else [-1]) + ([] if st.bot_pec else [N])
    # same layer, vertical separation and same height
    for r in range(N):
        lo, hi = st.zmin[r], st.zmax[r]
        h = hi - lo
        pts.append((0.7*h, lo + 0.3*h, lo + 0.6*h, r, r, 'same-layer'))
        pts.append((0.5*h, lo + 0.5*h, lo + 0.5*h, r, r, 'same-height'))
    # different layers
    for s in range(N):
        for o in range(N):
            if s != o:
                pts.append((0.5*(st.zmax[s] - st.zmin[s]), point_in(rng, st, o, 0.5), point_in(rng, st, s, 0.4), s, o, 'cross-layer'))
    # half-spaces
    for r in [-1, N]:
        if r in regions:
            zh_obs = st.zmax[0] + 0.3*H if r == -1 else st.zmin[-1] - 0.3*H
            zh_src = st.zmax[0] + 0.2*H if r == -1 else st.zmin[-1] - 0.2*H
            pts.append((0.3*H, zh_obs, point_in(rng, st, 0, 0.5), 0, r, 'obs-halfspace'))
            pts.append((0.3*H, point_in(rng, st, N - 1, 0.5), zh_src, r, N - 1, 'src-halfspace'))
    # near axis and far
    lo, hi = st.zmin[0], st.zmax[0]
    for rho in [1e-12*H, 1e-6*H]:
        pts.append((rho, lo + 0.8*(hi - lo), lo + 0.2*(hi - lo), 0, 0, 'near-axis'))
    pts.append((0.0, lo + 0.8*(hi - lo), lo + 0.2*(hi - lo), 0, 0, 'on-axis'))
    pts.append((min(3*lam, 1.0), lo + 0.6*(hi - lo), lo + 0.4*(hi - lo), 0, 0, 'far'))
    # on an interface (z = interface between layer 0 and layer 1, assigned to layer 0)
    if N > 1:
        pts.append((0.5*(hi - lo), st.zmin[0], st.zmin[0] + 0.5*(hi - lo), 0, 0, 'obs-on-interface'))
    # the top surface of the stack (FindLayer must give layer 0)
    pts.append((0.5*(hi - lo), st.zmax[0], lo + 0.5*(hi - lo), 0, FIND, 'obs-on-top-surface'))
    return pts


SETTING_VARIANTS = {
    'integrate': {'method': 'INTEGRATE'},
    'integrate_qs': {'method': 'INTEGRATE', 'extract_quasistatic': 1},
    'integrate_hom': {'method': 'INTEGRATE', 'extract_homogeneous': 1},
}


def check_spatial(probe, name, f, variant, pts, curl_type):
    layers, top, bot, _ = STACKS[name]
    settings = dict(SETTING_VARIANTS[variant], compute_curl=1, curl_type=curl_type)
    q = []
    for (rho, z, zp, s, o, kind) in pts:
        q.append(['mgf', float(rho)*np.cos(0.3), float(rho)*np.sin(0.3), float(z), float(zp), int(s), int(o)])
    job = {'stack': stack_job(layers, top, bot), 'freq': f, 'settings': settings, 'queries': q}
    out = run_probe(probe, job)
    return out


def reference_spatial(name, f, pt):
    """Reference scalars for the point and for the swapped (reciprocal) configuration."""
    st = ref_stack(name, f)
    rho, z, zp, s, o, kind = pt
    if o == FIND:
        o = st.find_layer(z)
    scal, curl, err = mgf_spatial(st, rho, z, zp, s, o)
    _, curl_sw, err_sw = mgf_spatial(st, rho, zp, z, o, s)
    return scal, curl, curl_sw, max(err, err_sw)


def compare_spatial(res, ref, curl_type, phi=0.3):
    scal, curl, curl_sw, _ = ref
    G, Gphi, H = assemble(scal, curl, phi)
    # E field of a magnetic dipole from reciprocity: G_EM(r, r') = -G_HJ(r', r)^T
    e_scal = np.array([-curl_sw[0], curl_sw[1], -curl_sw[3], -curl_sw[2]])
    _, _, E = assemble(scal, e_scal, phi)
    Gs = cvec(res['G'])
    ps = complex(*res['phi'])
    out = {}
    # component errors relative to the largest diagonal entry (off-diagonal components can be ~0)
    gscale = max(abs(G[0]), abs(G[8]))
    out['G_err'] = (np.abs(Gs - G)/gscale).tolist()
    out['phi_err'] = abs(ps - Gphi)/abs(Gphi)
    if 'curl' in res:
        Hs = cvec(res['curl'])
        Href = -H if curl_type == 'GHJ' else E
        hscale = np.max(np.abs(Href))
        out['curl_err'] = (np.abs(Hs - Href)/hscale).tolist()
    return out


# ------------------------------------------------------------------------------------------------
# Methods group: DCIM and interpolation against the reference; thread independence of table building
# ------------------------------------------------------------------------------------------------

METHOD_STACKS = [('two_layer', 1e9), ('two_layer', 10e9), ('ling_jin_2000', 30e9), ('onchip_si', 1e9), ('magnetic_lossy', 3e9)]


def method_points(name, f):
    st = ref_stack(name, f)
    lo, hi = st.zmin[0], st.zmax[0]
    h = hi - lo
    lam = 2*np.pi/abs(st.k(0))
    pts = []
    for frac_rho in (0.05, 0.2, 0.7):
        rho = frac_rho*min(lam, 20*h)
        pts.append((rho, lo + 0.7*h, lo + 0.3*h, 0, 0, 'same-layer'))
        pts.append((rho, lo + 0.5*h, lo + 0.5*h, 0, 0, 'same-height'))
        if st.N > 1:
            pts.append((rho, st.zmin[1] + 0.5*(st.zmax[1] - st.zmin[1]), lo + 0.5*h, 0, 1, 'cross-layer'))
    return pts


def methods_group(probe, workers, out):
    rows = []
    jobs = []
    for name, f in METHOD_STACKS:
        pts = method_points(name, f)
        layers, top, bot, _ = STACKS[name]
        zn = sorted(set([p[1] for p in pts] + [p[2] for p in pts]))
        rhos = [p[0] for p in pts]
        rho_nodes = list(np.linspace(0.5*min(rhos), 1.5*max(rhos), 60))
        q = [['mgf', float(r)*np.cos(0.3), float(r)*np.sin(0.3), float(z), float(zp), int(si), int(oi)] for (r, z, zp, si, oi, k) in pts]
        qn = [['mgf', float(rho_nodes[7]), 0.0, float(z), float(zp), int(si), int(oi)] for (r, z, zp, si, oi, k) in pts]
        base = {'stack': stack_job(layers, top, bot), 'freq': f, 'znodes': zn}
        for lvl in (1, 2, 3):
            jobs.append((name, f, 'dcim%d' % lvl, pts, dict(base, settings={'method': 'DCIM', 'DCIM_method': lvl}, queries=q), 1))
        for qs in (0, 1):
            st = {'method': 'INTERPOLATE', 'order': 3, 'extract_quasistatic': qs}
            jobs.append((name, f, 'interp' + ('_qs' if qs else ''), pts, dict(base, rhonodes=rho_nodes, settings=st, queries=q + qn), 1))
            jobs.append((name, f, 'interp' + ('_qs' if qs else '') + '_threads8', pts, dict(base, rhonodes=rho_nodes, settings=st, queries=q + qn), 8))
        jobs.append((name, f, 'integrate_at_nodes', pts, dict(base, settings={'method': 'INTEGRATE'}, queries=qn), 1))
    with cf.ThreadPoolExecutor(max(1, workers//2)) as ex:
        results = list(ex.map(lambda j: (j, run_probe(probe, j[4], threads=j[5])), jobs))
    refs = {}
    allpts = {(name, f, i): pt for (name, f, v, pts, job, th) in jobs for i, pt in enumerate(pts)}
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(reference_spatial, key[0], key[1], pt): key for key, pt in allpts.items()}
        for fu in cf.as_completed(futs):
            try:
                refs[futs[fu]] = fu.result()
            except Exception as e:  # noqa: BLE001
                refs[futs[fu]] = e
    by_key = {}
    for (name, f, v, pts, job, th), res in results:
        by_key[(name, f, v)] = res
        for i, (pt, r) in enumerate(zip(pts, res['results'])):
            row = dict(stack=name, f=f, variant=v, point=[float(x) if not isinstance(x, str) else x for x in pt])
            ref = refs.get((name, f, i))
            if 'err' in r:
                row['strata_err'] = r['err']
            elif isinstance(ref, Exception) or ref is None:
                row['ref_err'] = str(ref)
            elif v != 'integrate_at_nodes':
                row.update(compare_spatial(r, ref, 'GHJ'))
            rows.append(row)
    # consistency: interpolation at a rho node equals direct integration there; threads do not change the table
    cons = []
    for name, f in METHOD_STACKS:
        npts = len(method_points(name, f))
        direct = by_key[(name, f, 'integrate_at_nodes')]['results']
        for v in ('interp', 'interp_qs'):
            at_nodes = by_key[(name, f, v)]['results'][npts:]
            for a, b in zip(at_nodes, direct):
                if 'G' in a and 'G' in b:
                    ga, gb = cvec(a['G']), cvec(b['G'])
                    cons.append(dict(stack=name, f=f, variant=v, check='node_vs_integrate',
                                     err=float(np.max(np.abs(ga - gb))/max(np.max(np.abs(gb)), 1e-300))))
            r1 = by_key[(name, f, v)]['results']
            r8 = by_key[(name, f, v + '_threads8')]['results']
            same = all(a.get('G') == b.get('G') and a.get('phi') == b.get('phi') for a, b in zip(r1, r8))
            cons.append(dict(stack=name, f=f, variant=v, check='threads_1_vs_8_identical', ok=bool(same)))
    with open(out/'methods.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
        for c in cons:
            fh.write(json.dumps(c) + '\n')
    summ = {}
    for v in sorted(set(r['variant'] for r in rows)):
        sel = [r for r in rows if r['variant'] == v]
        ok = [r for r in sel if 'G_err' in r]
        errs = [max(max(r['G_err']), r['phi_err']) for r in ok]
        summ[v] = dict(n=len(sel), n_strata_err=sum('strata_err' in r for r in sel), n_nonfinite=sum(not np.isfinite(e) for e in errs),
                       median=float(np.median(errs)) if errs else None, max=float(np.nanmax(errs)) if errs else None)
    summ['node_vs_integrate_max'] = float(max([c['err'] for c in cons if c['check'] == 'node_vs_integrate'], default=np.nan))
    summ['threads_identical'] = all(c['ok'] for c in cons if c['check'] == 'threads_1_vs_8_identical')
    return summ


# ------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('probe')
    ap.add_argument('outdir')
    ap.add_argument('--groups', default='spec,spatial')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--nspec', type=int, default=60)
    ap.add_argument('--stacks', default=','.join(STACKS))
    ap.add_argument('--variants', default=','.join(SETTING_VARIANTS))
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    groups = a.groups.split(',')
    summary = {}
    rng = np.random.default_rng(20261020)

    if 'spec' in groups:
        cases = spectral_cases(rng, a.nspec)
        rows = []
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(lambda c: check_spectral(a.probe, c), cases):
                rows.extend(r)
        with open(out/'spec.jsonl', 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        tol = 1e-9
        s = {}
        for key in ('K_err', 'GHJ_err', 'GEM_err'):
            for c in range(5 if key == 'K_err' else 4):
                errs = [r[key][c] for r in rows if key in r]
                bad = [r for r in rows if key in r and not (r[key][c] < tol)]
                s['%s[%d]' % (key[:-4], c)] = dict(n=len(errs), max=float(np.nanmax(errs)) if errs else None, n_fail=len(bad))
        s['strata_errors'] = sum('strata_err' in r for r in rows)
        summary['spec'] = s
        print(json.dumps(s, indent=1))

    if 'spatial' in groups:
        jobs = []
        for name in a.stacks.split(','):
            for f in STACKS[name][3]:
                pts = spatial_points(name, f, rng)
                jobs.append((name, f, pts))
        # references (expensive) in a process pool
        refs = {}
        flat = [(name, f, i, pt) for (name, f, pts) in jobs for i, pt in enumerate(pts)]
        with cf.ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(reference_spatial, name, f, pt): (name, f, i) for (name, f, i, pt) in flat}
            for fu in cf.as_completed(futs):
                key = futs[fu]
                try:
                    refs[key] = fu.result()
                except Exception as e:  # noqa: BLE001
                    refs[key] = e
        rows = []
        tasks = [(name, f, v, ct, pts) for (name, f, pts) in jobs for v in a.variants.split(',') for ct in ('GHJ', 'GEM')]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            outs = list(ex.map(lambda t: (t, check_spatial(a.probe, t[0], t[1], t[2], t[4], t[3])), tasks))
        for (name, f, v, ct, pts), res in outs:
            for i, (pt, r) in enumerate(zip(pts, res['results'])):
                ref = refs[(name, f, i)]
                row = dict(stack=name, f=f, variant=v, curl_type=ct, point=[float(x) if not isinstance(x, str) else x for x in pt],
                           seconds=r.get('ms'))
                if isinstance(ref, Exception):
                    row['ref_err'] = str(ref)
                elif 'err' in r:
                    row['strata_err'] = r['err']
                else:
                    row.update(compare_spatial(r, ref, ct))
                    row['G'] = r['G']
                    row['phi'] = r['phi']
                    sc, cu, cusw, er = ref
                    row['ref_scalars'] = [[x.real, x.imag] for x in sc]
                    row['ref_curl'] = [[x.real, x.imag] for x in cu]
                    row['ref_curl_swapped'] = [[x.real, x.imag] for x in cusw]
                    row['curl'] = r.get('curl')
                rows.append(row)
        with open(out/'spatial.jsonl', 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        tol = 1e-5
        s = {}
        for v, ct in [(v, ct) for v in a.variants.split(',') for ct in ('GHJ', 'GEM')]:
            sel = [r for r in rows if r['variant'] == v and r['curl_type'] == ct]
            ok = [r for r in sel if 'G_err' in r]
            fails = [r for r in ok if max(max(r['G_err']), r['phi_err']) >= tol]
            cfails = [r for r in ok if 'curl_err' in r and max(r['curl_err']) >= tol]
            s[v + '/' + ct] = dict(n=len(sel), n_strata_err=sum('strata_err' in r for r in sel), n_ref_err=sum('ref_err' in r for r in sel),
                        n_fail_G=len(fails), n_fail_curl=len(cfails),
                        max_G=float(max([max(max(r['G_err']), r['phi_err']) for r in ok], default=np.nan)),
                        max_curl=float(max([max(r['curl_err']) for r in ok if 'curl_err' in r], default=np.nan)))
        summary['spatial'] = s
        print(json.dumps(s, indent=1))

    if 'methods' in groups:
        summary['methods'] = methods_group(a.probe, a.workers, out)
        print(json.dumps(summary['methods'], indent=1))

    (out/'summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
