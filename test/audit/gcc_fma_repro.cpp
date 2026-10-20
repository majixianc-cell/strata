// Minimal reproducer of a GCC 15.2 miscompilation (x86-64) found during the audit in SpectralMGF::ComputeCurlGEMii():
// two members decremented by the same complex product. Wrong with -O2 -mfma (or -march=haswell/native, default
// -ffp-contract=fast); correct with -O1, without FMA, with -ffp-contract=off/on, or with -fno-tree-slp-vectorize.
// The generated code uses vfmaddsub132pd; this matches the description of GCC PR 125654.
//   g++ -O2 -mfma gcc_fma_repro.cpp && ./a.out   -> MISCOMPILED with GCC 15.2.0 (Ubuntu 15.2.0-16ubuntu1)
#include <complex>
#include <cstdio>
#include <cmath>
struct S
{
	std::complex<double> VVe, VVh, VIh, IVe, P, jkzi, Zh, Ye;
	double z, zp;
	__attribute__((noinline)) void f();
};
void S::f()
{
	std::complex<double> exp_hom = std::exp(-jkzi*std::abs(z - zp));
	VVe -= (P/2.0)*exp_hom;
	VVh -= (P/2.0)*exp_hom;
	VIh -= (Zh/2.0)*exp_hom;
	IVe -= (Ye/2.0)*exp_hom;
}
int main()
{
	S s;
	s.VVe = {0.8504128517, -0.01242719311};
	s.VVh = {0.4428295504, 0.01516171391};
	s.VIh = {1.0, 0.0};
	s.IVe = {1.0, 0.0};
	s.P = {1.0, 0.0};
	s.jkzi = {73.7, 10.85};
	s.Zh = {2.0, 0.5};
	s.Ye = {0.3, 0.1};
	s.z = 0.6e-3;
	s.zp = 0.3e-3;
	std::complex<double> e = std::exp(-s.jkzi*std::abs(s.z - s.zp));
	std::complex<double> ve = s.VVe - (s.P/2.0)*e, vh = s.VVh - (s.P/2.0)*e;
	s.f();
	double err = std::abs(s.VVe - ve) + std::abs(s.VVh - vh);
	std::printf("VVe = %.10f %+.10fj (expected %.10f %+.10fj)\nVVh = %.10f %+.10fj (expected %.10f %+.10fj)\n%s\n",
	            s.VVe.real(), s.VVe.imag(), ve.real(), ve.imag(), s.VVh.real(), s.VVh.imag(), vh.real(), vh.imag(),
	            err < 1e-12 ? "OK" : "MISCOMPILED");
	return err < 1e-12 ? 0 : 1;
}
