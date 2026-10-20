"""Compare the outputs of Strata's built-in example programs with the reference outputs committed upstream
(test/examples/*/MGFdata*.txt) and with the digitized published curves (Ling & Jin 2000, Fig. 3).

usage: python3 compare_builtin.py STRATA_SRC OUTPUT_DIR
Prints, per file and column, the max and median relative deviation |new - ref| / max(|ref|, floor).
"""
import sys
from pathlib import Path
import numpy as np


def load(path):
    lines = Path(path).read_text().splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith('rho ') or l.startswith('t '))
    cols = lines[head].split()
    data = np.array([[float(x) for x in l.split()] for l in lines[head+1:] if l.strip()])
    return cols, data


def main(src, out):
    src, out = Path(src), Path(out)
    pairs = [('ling_jin_2000', n) for n in ('MGFdata.txt', 'MGFdata_interp.txt', 'MGFdata_DCIM.txt', 'MGFdata_singularity.txt')]
    pairs += [('yuan_sarkar_salazar-palma_2006', 'MGFdata_spectral.txt')]
    for ex, name in pairs:
        cr, ref = load(src/'test'/'examples'/ex/name)
        cn, new = load(out/name)
        assert cr == cn, (cr, cn)
        print(f'== {ex}/{name}: {len(ref)} ref rows, {len(new)} new rows; abscissa max rel diff '
              f'{np.max(np.abs(new[:, 0]-ref[:, 0])/np.abs(ref[:, 0]).clip(1e-300)):.2e}')
        for j, c in enumerate(cr[1:], start=1):
            r, n = ref[:, j], new[:, j]
            scale = np.maximum(np.abs(r), 1e-6*np.max(np.abs(r)) if np.max(np.abs(r)) > 0 else 1.0)
            d = np.abs(n - r)/scale
            if np.max(np.abs(r)) == 0 and np.max(np.abs(n)) == 0:
                continue
            k = int(np.argmax(d))
            print(f'   {c:8s} max {d[k]:.2e} at x={ref[k,0]:.4e} (ref {r[k]:.6e}, new {n[k]:.6e}); median {np.median(d):.2e}')
    # Published curves (digitized; only a loose check): Ling & Jin 2000, Fig. 3, |G| vs k0*rho.
    _, new = load(out/'MGFdata.txt')
    k0 = 2*np.pi*30e9*np.sqrt(8.854187817e-12*4e-7*np.pi)
    colmap = {'Gxx': 1, 'Gzx': 7, 'Gzz': 9, 'Gphi': 10}
    for comp, j in colmap.items():
        f = src/'test'/'examples'/'ling_jin_2000'/f'fig3_{comp}.csv'
        pts = np.loadtxt(f, delimiter=',')
        x, y = pts[:, 0], pts[:, 1]
        ours = np.interp(np.log(x), np.log(k0*new[:, 0]), np.log(new[:, j]))
        print(f'   paper {comp}: {len(x)} digitized pts, median |log(new/paper)| = {np.median(np.abs(ours-np.log(y))):.3f}, max = {np.max(np.abs(ours-np.log(y))):.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
