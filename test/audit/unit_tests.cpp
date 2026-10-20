// Strata audit: assertion-based regression tests.
//
// Each test runs in a forked child process with a time limit, so that a crash or an endless loop in one test is
// reported as such and does not stop the others. Exit status = number of tests that did not pass.
//
//   strata_unit_tests [FILTER]        run the tests whose id contains FILTER (all tests if omitted)
//
// Closed forms used below (Strata normalisation, exp(+jwt)): in a homogeneous medium
//   G_xx = G_yy = G_zz = mu_r exp(-jkR)/(4 pi R),  G_phi = (eps0/eps) exp(-jkR)/(4 pi R),  G_xz = G_zx = 0.

#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <chrono>
#include <ctime>
#include <stdlib.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(_OPENMP)
	#include <omp.h>
#endif

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include "libAmosBessel.h"

// White-box access to the interpolation-table internals (AddRhoTable, AddZTable, z_to_idx, ...).
#define private public
#include "MGF.hpp"
#include "sommerfeld_integrals.hpp"
#undef private

#include "constants.hpp"

using C = std::complex<double>;

namespace
{

struct T
{
	bool ok = true;
	std::ostringstream log;
	void check(bool cond, const std::string &what)
	{
		if (!cond) ok = false;
		log << (cond ? "      ok    " : "      FAIL  ") << what << "\n";
	}
	void note(const std::string &what) { log << "      note  " << what << "\n"; }
};

std::string fmt(double v, int p = 6)
{
	std::ostringstream o;
	o << std::setprecision(p) << v;
	return o.str();
}
std::string fmt(C v, int p = 6) { return "(" + fmt(v.real(), p) + "," + fmt(v.imag(), p) + ")"; }

double rel(C a, C b) { return std::abs(a - b)/std::max(std::abs(b), 1e-300); }

struct Case
{
	std::string id, title;
	std::function<void(T &)> fn;
	int timeout = 120;
};

std::vector<Case> &registry()
{
	static std::vector<Case> r;
	return r;
}

struct Reg
{
	Reg(const std::string &id, const std::string &title, std::function<void(T &)> fn, int timeout = 120)
	{
		registry().push_back({id, title, fn, timeout});
	}
};

// ------------------------------------------------------------------------------------------------
// Fixtures
// ------------------------------------------------------------------------------------------------

// Two dielectric layers between air and an eps_r = 6 half-space (used in the Codex review reproductions).
LayerManager two_layer(double f = 1e9)
{
	LayerManager lm;
	lm.AddLayer(0.0, 1e-3, {2.3, 0.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(-1e-3, 0.0, {4.5, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 6.0, 1.0, 0.0, false, false);
	lm.ProcessLayers(f);
	return lm;
}

// Homogeneous medium represented by one layer whose half-spaces have the same material.
LayerManager homogeneous(double epsr, double mur, double sigma, double zlo, double zhi, double f)
{
	LayerManager lm;
	lm.AddLayer(zlo, zhi, {epsr, 0.0}, mur, sigma, 0.0);
	lm.SetHalfspaces(epsr, mur, sigma, epsr, mur, sigma, false, false);
	lm.ProcessLayers(f);
	return lm;
}

C g0(C k, double R) { return std::exp(-strata::J*k*R)/(4.0*M_PI*R); }

MGF_settings quiet(int method)
{
	MGF_settings s;
	s.verbose = false;
	s.method = method;
	return s;
}

// ================================================================================================
// Layer management
// ================================================================================================

Reg lm_merge("LM-merge", "identical adjacent layers are merged, different ones are not", [](T &t) {
	LayerManager a;
	a.AddLayer(0.0, 1e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	a.AddLayer(-1e-3, 0.0, {4.0, 0.0}, 1.0, 0.0, 0.0);
	a.ProcessLayers(1e9);
	t.check(a.layers.size() == 1, "two identical eps_r=4 layers -> 1 layer (got " + std::to_string(a.layers.size()) + ")");
	if (a.layers.size() == 1)
		t.check(a.layers[0].zmin == -1e-3 && a.layers[0].zmax == 1e-3 && std::abs(a.layers[0].h - 2e-3) < 1e-15,
		        "merged extent [-1e-3, 1e-3]");
	LayerManager b;
	b.AddLayer(0.0, 1e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	b.AddLayer(-1e-3, 0.0, {4.5, 0.0}, 1.0, 0.0, 0.0);
	b.ProcessLayers(1e9);
	t.check(b.layers.size() == 2, "eps_r 4 / 4.5 stay separate");
	LayerManager c;
	c.AddLayer(0.0, 1e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	c.AddLayer(-1e-3, 0.0, {4.0, 0.0}, 1.0, 0.01, 0.0);
	c.ProcessLayers(1e9);
	t.check(c.layers.size() == 2, "different conductivity stays separate");
	LayerManager d;
	d.AddLayer(0.0, 1e-3, {4.0, -0.1}, 1.0, 0.0, 0.0);
	d.AddLayer(-1e-3, 0.0, {4.0, -0.2}, 1.0, 0.0, 0.0);
	d.ProcessLayers(1e9);
	t.check(d.layers.size() == 2, "different epsr_im stays separate");
});

Reg lm_merge_index("LM-merge-index", "merging reports the new index of every layer", [](T &t) {
	LayerManager a;
	a.AddLayer(1e-3, 2e-3, {2.0, 0.0}, 1.0, 0.0, 0.0);
	a.AddLayer(0.0, 1e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	a.AddLayer(-1e-3, 0.0, {4.0, 0.0}, 1.0, 0.0, 0.0);
	a.AddLayer(-2e-3, -1e-3, {9.0, 0.0}, 1.0, 0.0, 0.0);
	a.ProcessLayers(1e9);
	std::vector<int> expected = {0, 1, 1, 2};
	std::string got;
	for (int v : a.merged_index) got += std::to_string(v) + " ";
	t.check(a.layers.size() == 3 && a.merged_index == expected, "layers 0..3 -> 0 1 1 2 (got " + got + ")");
	a.ProcessLayers(2e9);
	t.check(a.merged_index == std::vector<int>({0, 1, 2}), "a second ProcessLayers() leaves the merged stack as it is");
});

Reg lm_find("LM-findlayer", "FindLayer at the top surface, interfaces and outside the stack", [](T &t) {
	LayerManager lm;
	lm.AddLayer(0.0, 1.0, {2.0, 0.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(-1.0, 0.0, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.ProcessLayers(1e9);
	t.check(lm.FindLayer(1.0) == 0, "top surface z=1 -> layer 0 (got " + std::to_string(lm.FindLayer(1.0)) + ")");
	t.check(lm.FindLayer(0.999) == 0, "z=0.999 -> layer 0");
	t.check(lm.FindLayer(0.0) == 0, "interior interface -> layer above");
	t.check(lm.FindLayer(-0.5) == 1, "z=-0.5 -> layer 1");
	t.check(lm.FindLayer(-1.0) == 1, "bottom surface -> bottom layer");
	t.check(lm.FindLayer(1.5) == -1, "above -> -1");
	t.check(lm.FindLayer(-1.5) == 2, "below -> N");
	LayerManager one;
	one.AddLayer(0.0, 1.0, {2.0, 0.0}, 1.0, 0.0, 0.0);
	one.ProcessLayers(1e9);
	t.check(one.FindLayer(1.0) == 0 && one.FindLayer(0.0) == 0, "single layer: both surfaces -> 0");
});

Reg lm_bounds("LM-getz-bounds", "GetZmin/GetZmax/GetHeight reject index N+1", [](T &t) {
	LayerManager lm = two_layer();
	int n = (int)lm.layers.size();
	for (int which = 0; which < 3; ++which)
	{
		bool thrown = false;
		try
		{
			if (which == 0) lm.GetZmin(n + 1);
			if (which == 1) lm.GetZmax(n + 1);
			if (which == 2) lm.GetHeight(n + 1);
		}
		catch (const std::out_of_range &) { thrown = true; }
		t.check(thrown, std::string(which == 0 ? "GetZmin" : which == 1 ? "GetZmax" : "GetHeight") + "(N+1) throws out_of_range");
	}
	t.check(lm.GetZmax(-1) > lm.layers[0].zmax && lm.GetZmin(n) < lm.layers.back().zmin, "half-space placeholders still work");
});

void write_file(const std::string &path, const std::string &text)
{
	std::ofstream o(path, std::ios::binary);
	o << text;
}

Reg lm_tech("LM-tech", ".tech parser with LF and CRLF line endings", [](T &t) {
	const std::string body[] = {"// stack-up", "Dielectric", "0 top 0 0 1.0 1.0 0.0", "1 L1 0 100 4.0 1.0 0.0",
	                            "2 L2 -100 100 2.0 1.0 0.0", "3 bot 0 0 1.0 1.0 -1"};
	for (int crlf = 0; crlf < 2; ++crlf)
	{
		std::string text;
		for (const auto &l : body) text += l + (crlf ? "\r\n" : "\n");
		std::string path = "/tmp/strata_unit_" + std::to_string(getpid()) + (crlf ? "_crlf" : "_lf") + ".tech";
		write_file(path, text);
		LayerManager lm;
		lm.ProcessTechFile(path);
		std::remove(path.c_str());
		std::string tag = crlf ? "CRLF: " : "LF: ";
		t.check(lm.layers.size() == 2, tag + "2 layers parsed (got " + std::to_string(lm.layers.size()) + ")");
		if (lm.layers.size() == 2)
			t.check(std::abs(lm.layers[0].zmax - 100e-9) < 1e-18 && std::abs(lm.layers[1].zmin + 100e-9) < 1e-18 &&
			        lm.layers[0].epsr.real() == 4.0, tag + "geometry and eps_r");
		t.check(lm.isPEC_bot, tag + "PEC bottom half-space");
	}
});

Reg lm_tech_missing("LM-tech-missing", "missing .tech file raises an error instead of looping", [](T &t) {
	LayerManager lm;
	bool thrown = false;
	try { lm.ProcessTechFile("/nonexistent/strata_missing.tech"); }
	catch (const std::exception &) { thrown = true; }
	t.check(thrown, "exception for a missing file");
}, 10);

Reg lm_hs_im("LM-halfspace-epsr-im", "epsr_im of a half-space is honoured like for layers", [](T &t) {
	std::string path = "/tmp/strata_unit_" + std::to_string(getpid()) + ".yaml";
	write_file(path, "unit: um\ndielectric_layers:\n    L1:\n        {zmin: 0, h: 1, epsr: 4, epsr_im: -0.4}\n"
	                 "top_halfspace:\n    {epsr: 2, epsr_im: -0.2}\nbottom_halfspace:\n    {epsr: 3, epsr_im: -0.3}\n");
	LayerManager lm;
	lm.ProcessTechFile(path);
	std::remove(path.c_str());
	lm.ProcessLayers(1e9);
	t.check(std::abs(lm.eps[0].imag()/strata::eps0 + 0.4) < 1e-12, "layer epsr_im = -0.4 (control)");
	t.check(std::abs(lm.eps_top.imag()/strata::eps0 + 0.2) < 1e-12, "top half-space epsr_im = -0.2 (got " + fmt(lm.eps_top.imag()/strata::eps0) + ")");
	t.check(std::abs(lm.eps_bot.imag()/strata::eps0 + 0.3) < 1e-12, "bottom half-space epsr_im = -0.3");
});

Reg lm_znodes("LM-znodes-thin", "SetZNodesInterp keeps every node inside its (thin) layer", [](T &t) {
	LayerManager lm;
	lm.AddLayer(0.0, 10e-9, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(-5e-9, 0.0, {2.0, 0.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(-1e-6, -5e-9, {3.0, 0.0}, 1.0, 0.0, 0.0);
	lm.ProcessLayers(1e9);
	std::vector<double> nodes;
	lm.SetZNodesInterp(1e9, nodes, 10);
	int outside = 0;
	for (double z : nodes)
	{
		int layer = lm.FindLayer(z);
		if (layer < 0 || layer >= (int)lm.layers.size()) ++outside;
	}
	bool ordered = true;
	// Each layer contributes at least two nodes with zmin <= z1 < z2 <= zmax.
	for (size_t k = 0; k + 1 < nodes.size(); k += 2)
		if (!(nodes[k] < nodes[k + 1])) ordered = false;
	t.check(outside == 0, "no node outside the stack");
	t.check(ordered, "node pairs ordered within each thin layer");
	int bad = 0;
	for (size_t k = 0; k < nodes.size(); ++k)
	{
		bool inside_any = false;
		for (auto &l : lm.layers)
			if (nodes[k] >= l.zmin && nodes[k] <= l.zmax) inside_any = true;
		if (!inside_any) ++bad;
	}
	t.check(bad == 0, "every node lies within some layer");
	std::ostringstream o;
	for (double z : nodes) o << z << " ";
	t.note("nodes: " + o.str());
});

Reg lm_empty("LM-empty", "ProcessLayers without any layer raises an error", [](T &t) {
	LayerManager lm;
	bool thrown = false;
	try { lm.ProcessLayers(1e9); }
	catch (const std::exception &) { thrown = true; }
	t.check(thrown, "exception for an empty stack");
}, 10);

// ================================================================================================
// Bessel zeros and quadrature
// ================================================================================================

Reg bz_table("BZ-table", "BesselJ_NextZero beyond the 500 tabulated zeros", [](T &t) {
	for (int v = 0; v <= 2; ++v)
		for (double x : {10.0, 1000.0, 1566.0, 2000.0, 5000.0})
		{
			double z = BesselJ_NextZero(v, x);
			double jz = boost::math::cyl_bessel_j(v, z);
			t.check(z >= x && z <= x + M_PI + 1e-9 && std::abs(jz) < 1e-4,
			        "J" + std::to_string(v) + " next zero after " + fmt(x) + " = " + fmt(z, 10) + " (J=" + fmt(jz, 3) + ")");
		}
});

Reg qd_tanhsinh("QD-tanhsinh", "TanhSinh reaches the requested tolerance on smooth integrals", [](T &t) {
	struct I { std::string name; std::function<C(double)> f; double a, b; C exact; };
	std::vector<I> cases = {
		{"exp(x) on [0,1]", [](double x) { return C(std::exp(x)); }, 0.0, 1.0, C(std::exp(1.0) - 1.0)},
		{"1/x on [1,2]", [](double x) { return C(1.0/x); }, 1.0, 2.0, C(std::log(2.0))},
		{"cos(10x)+i sin(3x) on [0,2]", [](double x) { return C(std::cos(10*x), std::sin(3*x)); }, 0.0, 2.0,
		 C(std::sin(20.0)/10.0, (1.0 - std::cos(6.0))/3.0)},
		{"1/(1+25x^2) on [-1,1]", [](double x) { return C(1.0/(1.0 + 25*x*x)); }, -1.0, 1.0, C(0.4*std::atan(5.0))},
	};
	for (auto &c : cases)
	{
		C v;
		TanhSinh(c.f, c.a, c.b, 1e-12, v);
		double e = rel(v, c.exact);
		t.check(e < 1e-10, c.name + ": rel err " + fmt(e, 3) + " (eps 1e-12)");
	}
});

Reg qd_partextrap("QD-partextrap-nonconv", "PartExtrap does not silently return an unconverged value", [](T &t) {
	auto f = [](double x) { return C(std::cos(x)*std::exp(-0.01*x)); };
	C v = 0.0;
	bool thrown = false;
#ifdef STRATA_SOMMERFELD_DIAGNOSTICS
	long before = SommerfeldTailNonConvergedCount();
#else
	long before = 0;
#endif
	try { PartExtrap(f, 0.0, M_PI, 0.0, v); }
	catch (const std::exception &) { thrown = true; }
#ifdef STRATA_SOMMERFELD_DIAGNOSTICS
	long after = SommerfeldTailNonConvergedCount();
#else
	long after = 0;
#endif
	t.check(thrown || after == before + 1, "non-convergence is reported (exception or diagnostic counter)");
}, 30);

Reg qd_partextrap_scale("QD-partextrap-scale", "PartExtrap stays finite for nanometre rho (partitions ~1e9 wide)", [](T &t) {
	// int_a^inf J0(rho x) x^(-1/2) dx, with int_0^inf = Gamma(1/4)/(Gamma(3/4) sqrt(2 rho)). Tolerance 0 uses every
	// partition, as an unconverged (noisy) tail does.
	for (double rho : {1e-3, 2.5e-9, 1e-10})
	{
		double a = boost::math::cyl_bessel_j_zero(0.0, 1)/rho;
		auto g = [rho](double x) { return boost::math::cyl_bessel_j(0, rho*x)/std::sqrt(x); };
		auto f = [&g](double x) { return C(g(x)); };
		boost::math::quadrature::tanh_sinh<double> ts;
		double head = ts.integrate(g, 0.0, a);
		double exact = std::tgamma(0.25)/(std::tgamma(0.75)*std::sqrt(2.0*rho));
		C tail = 0.0;
		std::string what = "finite";
		try { PartExtrap(f, a, M_PI/rho, 0.0, tail); }
		catch (const std::exception &e) { what = e.what(); }
		double err = std::abs(head + tail - exact)/exact;
		t.check(what == "finite" && err < 1e-6, "rho = " + fmt(rho) + ": " + what + ", rel error " + fmt(err, 3));
	}
}, 60);

// ================================================================================================
// Direct integration: closed forms
// ================================================================================================

Reg qd_homog("QD-homogeneous", "INTEGRATE in a homogeneous medium matches exp(-jkR)/(4 pi R)", [](T &t) {
	struct P { double epsr, mur, sigma, f, z, zp, rho; };
	std::vector<P> pts = {
		{4.0, 1.0, 0.0, 1e9, 0.0, 1e-3, 3e-3}, {4.0, 1.0, 0.0, 1e9, 0.0, 0.0, 3e-3}, {2.0, 2.0, 0.02, 3e9, -2e-3, 4e-3, 1e-2},
		{1.0, 1.0, 0.0, 30e9, 0.0, 2e-3, 0.05}, {11.9, 1.0, 10.0, 1e8, 0.0, 1e-5, 2e-5},
		// upstream issue #5: emitter at 640 nm in a single layer with the same medium above and below (n = 1.5)
		{2.25, 1.0, 0.0, 4.684e14, 0.0, 0.3e-6, 1.0e-6}, {2.25, 1.0, 0.0, 4.684e14, 0.0, 0.0, 2.0e-6},
	};
	for (auto &p : pts)
	{
		double half = (p.f > 1e13) ? 5e-6 : 0.05;
		LayerManager lm = homogeneous(p.epsr, p.mur, p.sigma, -half, half, p.f);
		MGF mgf;
		MGF_settings s = quiet(MGF_INTEGRATE);
		mgf.Initialize(p.f, lm, s);
		mgf.SetLayers(0, 0);
		std::array<C, 9> G;
		C phi;
		mgf.ComputeMGF(p.rho, 0.0, p.z, p.zp, G, phi);
		double R = std::sqrt(p.rho*p.rho + (p.z - p.zp)*(p.z - p.zp));
		C k = mgf.lm.k[0];
		C gref = g0(k, R);
		double e1 = rel(G[0], p.mur*gref), e2 = rel(G[8], p.mur*gref), e3 = rel(phi, gref*strata::eps0/mgf.lm.eps[0]);
		double e4 = std::abs(G[2])/std::abs(gref) + std::abs(G[6])/std::abs(gref);
		t.check(std::max({e1, e2, e3, e4}) < 1e-6, "eps_r=" + fmt(p.epsr) + " mu_r=" + fmt(p.mur) + " sigma=" + fmt(p.sigma) + " f=" +
		        fmt(p.f) + " rho=" + fmt(p.rho) + " dz=" + fmt(p.z - p.zp) + ": Gxx " + fmt(e1, 3) + " Gzz " + fmt(e2, 3) +
		        " Gphi " + fmt(e3, 3) + " |Gxz|+|Gzx| " + fmt(e4, 3));
	}
}, 300);

Reg qd_nearaxis("QD-nearaxis", "small positive rho with a large vertical separation", [](T &t) {
	const double f = 3e9;
	LayerManager lm = homogeneous(2.0, 1.0, 0.0, -0.01, 0.01, f);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTEGRATE);
	mgf.Initialize(f, lm, s);
	mgf.SetLayers(0, 0);
	for (double rho : {0.0, 1e-14, 1e-10, 1e-6, 2e-3})
	{
		std::array<C, 9> G;
		C phi;
		mgf.ComputeMGF(rho, 0.0, -0.003, 0.006, G, phi);
		double R = std::sqrt(rho*rho + 0.009*0.009);
		C gref = g0(mgf.lm.k[0], R);
		double e = std::max(rel(G[0], gref), rel(phi, gref/2.0));
		t.check(e < 1e-6, "rho=" + fmt(rho) + ": rel err " + fmt(e, 3));
	}
}, 120);

Reg qd_poles("QD-surface-wave-poles", "lossless thick substrate: default tolerance agrees with a converged integration", [](T &t) {
	// Grounded four-layer substrate (eps_r up to 12.5) at 30 GHz: several surface-wave poles on the real axis
	// between k0 and k_max. Cross-layer point of the physics checks where the curl was off by 5.6e-4.
	const double f = 30e9;
	auto stack = [f]() {
		LayerManager lm;
		lm.AddLayer(1.1e-3, 1.8e-3, {2.1, 0.0}, 1.0, 0.0, 0.0);
		lm.AddLayer(0.8e-3, 1.1e-3, {12.5, 0.0}, 1.0, 0.0, 0.0);
		lm.AddLayer(0.3e-3, 0.8e-3, {9.8, 0.0}, 1.0, 0.0, 0.0);
		lm.AddLayer(0.0, 0.3e-3, {8.6, 0.0}, 1.0, 0.0, 0.0);
		lm.SetHalfspaces(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, false, true);
		lm.ProcessLayers(f);
		return lm;
	};
	const double rho = 0.35e-3, phi = 0.3, z = 0.55e-3, zp = 1.38e-3;
	std::array<C, 9> G[2], H[2];
	C p[2];
	for (int pass = 0; pass < 2; ++pass)
	{
		LayerManager lm = stack();
		MGF mgf;
		MGF_settings s = quiet(MGF_INTEGRATE);
		s.compute_curl = true;
		s.curl_type = MGF_GHJ;
		if (pass == 1) s.tol_quadrature = 1e-10;
		mgf.Initialize(f, lm, s);
		mgf.SetLayers(lm.FindLayer(zp), lm.FindLayer(z));
		mgf.ComputeMGF(rho*std::cos(phi), rho*std::sin(phi), z, zp, G[pass], p[pass]);
		mgf.ComputeCurlMGF(rho*std::cos(phi), rho*std::sin(phi), z, zp, H[pass]);
	}
	double gs = 0.0, hs = 0.0, ge = 0.0, he = 0.0;
	for (int n = 0; n < 9; ++n) { gs = std::max(gs, std::abs(G[1][n])); hs = std::max(hs, std::abs(H[1][n])); }
	for (int n = 0; n < 9; ++n) { ge = std::max(ge, std::abs(G[0][n] - G[1][n])/gs); he = std::max(he, std::abs(H[0][n] - H[1][n])/hs); }
	ge = std::max(ge, std::abs(p[0] - p[1])/std::abs(p[1]));
	t.check(ge < 1e-6, "MGF and G_phi: max deviation " + fmt(ge, 3));
	t.check(he < 1e-6, "curl (G_HJ): max deviation " + fmt(he, 3));
}, 300);

// Default settings against a converged integration (tol_quadrature = 1e-10) of the same engine, optionally with the
// switching point of the reference moved to ref_switching_point.
double max_deviation(LayerManager (*stack)(), double f, MGF_settings s, double rho, double z, double zp,
                     double ref_switching_point = -1.0)
{
	std::array<C, 9> G[2], H[2];
	C p[2];
	for (int pass = 0; pass < 2; ++pass)
	{
		LayerManager lm = stack();
		MGF mgf;
		MGF_settings ss = s;
		if (pass == 1)
		{
			ss.tol_quadrature = 1e-10;
			ss.switching_point = ref_switching_point;
		}
		mgf.Initialize(f, lm, ss);
		mgf.SetLayers(lm.FindLayer(zp), lm.FindLayer(z));
		mgf.ComputeMGF(rho*std::cos(0.3), rho*std::sin(0.3), z, zp, G[pass], p[pass]);
		mgf.ComputeCurlMGF(rho*std::cos(0.3), rho*std::sin(0.3), z, zp, H[pass]);
	}
	double gs = std::max(std::abs(G[1][0]), std::abs(G[1][8])), hs = 0.0, e = std::abs(p[0] - p[1])/std::abs(p[1]);
	for (int n = 0; n < 9; ++n) hs = std::max(hs, std::abs(H[1][n]));
	for (int n = 0; n < 9; ++n) e = std::max({e, std::abs(G[0][n] - G[1][n])/gs, std::abs(H[0][n] - H[1][n])/hs});
	return e;
}

LayerManager hi_eps_halfspace()
{
	LayerManager lm;
	lm.AddLayer(0.0, 0.2e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 12.0, 1.0, 0.0, false, false);
	lm.ProcessLayers(10e9);
	return lm;
}

Reg qd_branch("QD-halfspace-branch-point", "a half-space with the largest wave number is enclosed by the near-field path", [](T &t) {
	// eps_r = 12 half-space below an eps_r = 4 layer: its branch point lay beyond the switching point 1.2 |k_max| on
	// the real axis when k_max ignored the half-spaces (physics checks: 3.8e-5, curl 2.7e-4, and 8.4e-5 with extraction
	// for a source in the half-space). The error is systematic (a tighter tolerance does not remove it), so the
	// reference places the switching point well beyond the half-space's branch point.
	MGF_settings s = quiet(MGF_INTEGRATE);
	s.compute_curl = true;
	s.curl_type = MGF_GHJ;
	const double sp = 2.0*std::abs(hi_eps_halfspace().k_bot);
	double e = max_deviation(hi_eps_halfspace, 10e9, s, 0.044968868701575425, 0.12e-3, 0.08e-3, sp);
	t.check(e < 1e-6, "far point (rho = 45 mm), plain integration: max deviation " + fmt(e, 3));
	s.extract_homogeneous = true;
	e = max_deviation(hi_eps_halfspace, 10e9, s, 6e-5, 0.1e-3, -0.04e-3, sp);
	t.check(e < 1e-6, "source in the half-space, extract_homogeneous: max deviation " + fmt(e, 3));
}, 300);

LayerManager thin_nm()
{
	LayerManager lm;
	lm.AddLayer(25e-9, 30e-9, {7.0, 0.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(5e-9, 25e-9, {25.0, 0.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(-1e-6, 5e-9, {3.9, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 11.9, 1.0, 10.0, false, false);
	lm.ProcessLayers(1e9);
	return lm;
}

Reg ex_axis("EX-quasistatic-near-axis", "extract_quasistatic keeps the O(rho) components next to the axis", [](T &t) {
	// rho = 0.5 pm in a nanometre stack: the spatial quasistatic J_1 terms were dropped below rho = 1e-6 x (stack
	// height) = 1.03 pm while the spectral part still subtracted them (G_xz, G_zx and the curl off by up to 8.7e-5).
	auto run = [](bool qs) {
		LayerManager lm = thin_nm();
		MGF mgf;
		MGF_settings s = quiet(MGF_INTEGRATE);
		s.compute_curl = true;
		s.curl_type = MGF_GHJ;
		s.extract_quasistatic = qs;
		mgf.Initialize(1e9, lm, s);
		const double rho = 0.5e-12, z = 29e-9, zp = 26e-9;
		mgf.SetLayers(lm.FindLayer(zp), lm.FindLayer(z));
		std::array<C, 9> G, H;
		C p;
		mgf.ComputeMGF(rho*std::cos(0.3), rho*std::sin(0.3), z, zp, G, p);
		mgf.ComputeCurlMGF(rho*std::cos(0.3), rho*std::sin(0.3), z, zp, H);
		std::array<C, 19> out;
		for (int n = 0; n < 9; ++n) { out[n] = G[n]; out[9 + n] = H[n]; }
		out[18] = p;
		return out;
	};
	auto a = run(false), b = run(true);
	double gs = std::max(std::abs(a[0]), std::abs(a[8])), hs = 0.0, eg = 0.0, eh = 0.0;
	for (int n = 9; n < 18; ++n) hs = std::max(hs, std::abs(a[n]));
	for (int n = 0; n < 9; ++n) eg = std::max(eg, std::abs(a[n] - b[n])/gs);
	for (int n = 9; n < 18; ++n) eh = std::max(eh, std::abs(a[n] - b[n])/hs);
	t.check(eg < 1e-8, "MGF with and without extract_quasistatic: max deviation " + fmt(eg, 3) + " (G_xz " + fmt(a[2]) + " vs " + fmt(b[2]) + ")");
	t.check(eh < 1e-8, "curl: max deviation " + fmt(eh, 3));
}, 300);

Reg qd_farrho("QD-large-rho", "rho beyond the tabulated Bessel zeros (a*rho > 1566.87)", [](T &t) {
	const double f = 30e9;
	LayerManager lm = homogeneous(1.0, 1.0, 0.0, -0.01, 0.01, f);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTEGRATE);
	mgf.Initialize(f, lm, s);
	mgf.SetLayers(0, 0);
	for (double rho : {1.0, 3.0})
	{
		std::array<C, 9> G;
		C phi;
		mgf.ComputeMGF(rho, 0.0, 0.0, 1e-3, G, phi);
		double R = std::sqrt(rho*rho + 1e-6);
		C gref = g0(mgf.lm.k[0], R);
		double e = rel(G[0], gref);
		t.check(e < 1e-5, "rho=" + fmt(rho) + " m (a*rho=" + fmt(1.2*std::abs(mgf.lm.k[0])*rho) + "): rel err " + fmt(e, 3));
	}
}, 600);

Reg ex_homog("EX-homogeneous", "extract_homogeneous adds the extracted term back in the spatial domain", [](T &t) {
	const double f = 1e9;
	LayerManager lm = homogeneous(2.0, 1.0, 0.0, -0.01, 0.01, f);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTEGRATE);
	s.extract_homogeneous = true;
	mgf.Initialize(f, lm, s);
	mgf.SetLayers(0, 0);
	std::array<C, 9> G;
	C phi;
	mgf.ComputeMGF(3e-3, 0.0, 0.0, 2e-3, G, phi);
	double R = std::sqrt(9e-6 + 4e-6);
	C gref = g0(mgf.lm.k[0], R);
	t.check(rel(G[0], gref) < 1e-6, "Gxx rel err " + fmt(rel(G[0], gref), 3));
	t.check(rel(phi, gref/2.0) < 1e-6, "Gphi rel err " + fmt(rel(phi, gref/2.0), 3));
}, 120);

Reg uq_lowfreq("UQ-lowfreq", "extract_quasistatic agrees with plain integration at very low frequency", [](T &t) {
	const double f = 1e6;
	LayerManager lm;
	lm.AddLayer(0.0, 1e-6, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, false, true);
	lm.ProcessLayers(f);
	std::array<C, 9> Ga, Gb;
	C pa, pb;
	for (int qs = 0; qs < 2; ++qs)
	{
		MGF mgf;
		MGF_settings s = quiet(MGF_INTEGRATE);
		s.extract_quasistatic = qs;
		mgf.Initialize(f, lm, s);
		mgf.SetLayers(0, 0);
		mgf.ComputeMGF(5e-6, 0.0, 0.5e-6, 0.5e-6, qs ? Gb : Ga, qs ? pb : pa);
	}
	t.check(rel(Gb[0], Ga[0]) < 1e-4, "Gxx: plain " + fmt(Ga[0]) + " vs qs " + fmt(Gb[0]) + " rel " + fmt(rel(Gb[0], Ga[0]), 3));
	t.check(rel(pb, pa) < 1e-4, "Gphi: plain " + fmt(pa) + " vs qs " + fmt(pb) + " rel " + fmt(rel(pb, pa), 3));
}, 300);

Reg hs_obs("HS-observation", "observation point in a lossless half-space gives a finite MGF", [](T &t) {
	const double f = 1e9;
	LayerManager lm = two_layer(f);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTEGRATE);
	mgf.Initialize(f, lm, s);
	for (auto zz : std::vector<std::array<double, 2>>{{2e-3, 0.5e-3}, {0.5e-3, 2e-3}, {-2e-3, 0.5e-3}, {2e-3, 3e-3}})
	{
		mgf.SetLayers(mgf.lm.FindLayer(zz[1]), mgf.lm.FindLayer(zz[0]));
		std::array<C, 9> G;
		C phi;
		std::string err;
		try { mgf.ComputeMGF(1e-3, 0.0, zz[0], zz[1], G, phi); }
		catch (const std::exception &e) { err = e.what(); }
		bool finite = err.empty() && std::isfinite(std::abs(G[0])) && std::isfinite(std::abs(G[2])) &&
		              std::isfinite(std::abs(G[6])) && std::isfinite(std::abs(G[8])) && std::isfinite(std::abs(phi));
		t.check(finite, "z=" + fmt(zz[0]) + " zp=" + fmt(zz[1]) + ": finite (Gxx=" + fmt(G[0]) + ")" + (err.empty() ? "" : " err: " + err));
	}
}, 300);

// ================================================================================================
// Spectral API state
// ================================================================================================

void setup_spec(SpectralMGF &s, LayerManager &lm, int i, int m, double zp, double z, C kr, std::string order = "SOK")
{
	s.Initialize(&lm, 1e9);
	s.SetLayers(i, m);
	for (char c : order)
	{
		if (c == 'S') s.SetSourcePoint(zp);
		if (c == 'O') s.SetObservationPoint(z);
		if (c == 'K') s.SetRadialWaveNumber(kr);
	}
}

std::array<C, 5> potentials(SpectralMGF &s)
{
	s.ComputeSpectralMGF();
	std::array<C, 5> K;
	for (int c = 0; c < 5; ++c) K[c] = s.GetResult(c);
	return K;
}

double maxrel5(const std::array<C, 5> &a, const std::array<C, 5> &b)
{
	double e = 0;
	for (int c = 0; c < 5; ++c) e = std::max(e, std::abs(a[c] - b[c])/std::max(std::abs(b[c]), 1e-300));
	return e;
}

Reg sp_state("SP-state", "SpectralMGF results do not depend on call history or setter order", [](T &t) {
	LayerManager lm = two_layer();
	const C kr(80.0, 10.0);
	struct Cfg { int i, m; double zp, z; };
	for (Cfg c : {Cfg{0, 1, 0.7e-3, -0.3e-3}, Cfg{1, 0, -0.3e-3, 0.7e-3}, Cfg{0, 0, 0.7e-3, 0.3e-3}, Cfg{0, -1, 0.7e-3, 2e-3}})
	{
		std::string tag = "i=" + std::to_string(c.i) + " m=" + std::to_string(c.m) + ": ";
		SpectralMGF fresh;
		setup_spec(fresh, lm, c.i, c.m, c.zp, c.z, kr);
		auto ref = potentials(fresh);
		auto again = potentials(fresh);
		t.check(maxrel5(again, ref) < 1e-12, tag + "repeated ComputeSpectralMGF (" + fmt(maxrel5(again, ref), 3) + ")");
		for (std::string order : {"KSO", "KOS", "SKO", "OKS", "OSK"})
		{
			SpectralMGF s;
			setup_spec(s, lm, c.i, c.m, c.zp, c.z, kr, order);
			double e = maxrel5(potentials(s), ref);
			t.check(e < 1e-12, tag + "setter order " + order + " (" + fmt(e, 3) + ")");
		}
		for (int ghj = 0; ghj < 2; ++ghj)
		{
			SpectralMGF s;
			setup_spec(s, lm, c.i, c.m, c.zp, c.z, kr);
			potentials(s);
			if (ghj) s.SetCurlToGHJ(); else s.SetCurlToGEM();
			s.ComputeCurlSpectralMGF();
			double e = maxrel5(potentials(s), ref);
			t.check(e < 1e-12, tag + "potentials after a " + (ghj ? std::string("GHJ") : std::string("GEM")) + " curl call (" + fmt(e, 3) + ")");
		}
		// A curl call must not depend on a preceding curl of the other type either.
		SpectralMGF h1, h2;
		setup_spec(h1, lm, c.i, c.m, c.zp, c.z, kr);
		h1.SetCurlToGHJ();
		h1.ComputeCurlSpectralMGF();
		setup_spec(h2, lm, c.i, c.m, c.zp, c.z, kr);
		h2.SetCurlToGEM();
		h2.ComputeCurlSpectralMGF();
		h2.SetCurlToGHJ();
		h2.ComputeCurlSpectralMGF();
		double e = 0;
		for (int n = 0; n < 4; ++n) e = std::max(e, rel(h2.GetResultCurl(n), h1.GetResultCurl(n)));
		t.check(e < 1e-12, tag + "GHJ after GEM on the same object (" + fmt(e, 3) + ")");
	}
});

Reg sp_recip("SP-curl-reciprocity", "G_HJ(r,r') and G_EM(r',r) satisfy the reciprocity mapping", [](T &t) {
	LayerManager lm = two_layer();
	const C kr(80.0, 10.0);
	struct Cfg { int i, m; double zp, z; };
	const int rev[] = {0, 1, 3, 2};
	for (Cfg c : {Cfg{0, 0, 0.7e-3, 0.3e-3}, Cfg{0, 1, 0.7e-3, -0.3e-3}, Cfg{1, 0, -0.3e-3, 0.7e-3}, Cfg{0, -1, 0.7e-3, 2e-3},
	              Cfg{0, 2, 0.7e-3, -2e-3}, Cfg{-1, 2, 2e-3, -2e-3}})
	{
		SpectralMGF h, e;
		setup_spec(h, lm, c.i, c.m, c.zp, c.z, kr);
		h.SetCurlToGHJ();
		h.ComputeCurlSpectralMGF();
		setup_spec(e, lm, c.m, c.i, c.z, c.zp, kr);
		e.SetCurlToGEM();
		e.ComputeCurlSpectralMGF();
		double err = 0;
		for (int n = 0; n < 4; ++n)
			err = std::max(err, rel(h.GetResultCurl(n), e.GetResultCurl(rev[n])*(n == 1 ? -1.0 : 1.0)));
		t.check(err < 1e-10, "i=" + std::to_string(c.i) + " m=" + std::to_string(c.m) + ": max rel " + fmt(err, 3));
	}
});

// ================================================================================================
// Quasistatic module
// ================================================================================================

Reg qs_state("QS-state", "quasistatic results do not depend on a preceding curl call", [](T &t) {
	LayerManager lm = two_layer();
	for (double zz : {0.0, 0.3e-3})
	{
		QuasistaticMGF q;
		q.Initialize(&lm, 1e9);
		q.SetLayers(0, 1);
		double z = zz > 0 ? -zz : 0.0, zp = zz;
		q.ComputeQMGF_Spatial(z, zp, 5e-4);
		C k1 = q.GetResult_Spatial(1), k2 = q.GetResult_Spatial(2);
		q.ComputeCurlQMGF_Spatial(z, zp, 5e-4);
		q.ComputeQMGF_Spatial(z, zp, 5e-4);
		t.check(rel(q.GetResult_Spatial(1), k1) < 1e-14 && rel(q.GetResult_Spatial(2), k2) < 1e-14,
		        "spatial z=" + fmt(z) + " zp=" + fmt(zp) + ": K1 " + fmt(k1) + " -> " + fmt(q.GetResult_Spatial(1)));
		QuasistaticMGF r;
		r.Initialize(&lm, 1e9);
		r.SetLayers(0, 1);
		r.ComputeQMGF_Spectral(z, zp, C(80.0, 10.0));
		C s1 = r.GetResult_Spectral(1);
		r.ComputeCurlQMGF_Spectral(z, zp, C(80.0, 10.0));
		r.ComputeQMGF_Spectral(z, zp, C(80.0, 10.0));
		t.check(rel(r.GetResult_Spectral(1), s1) < 1e-14, "spectral z=" + fmt(z) + ": K1 " + fmt(s1) + " -> " + fmt(r.GetResult_Spectral(1)));
	}
});

Reg qs_fresnel("QS-halfspace-fresnel", "Fresnel coefficients seen from a half-space source layer", [](T &t) {
	LayerManager lm = two_layer();
	QuasistaticMGF q;
	q.Initialize(&lm, 1e9);
	int n = (int)lm.layers.size();
	Fresnel top = q.GetFresnelCoefficient_Downward(-1);
	Fresnel bot = q.GetFresnelCoefficient_Upward(n - 1);
	C fe_top = (lm.eps_top - lm.eps[0])/(lm.eps_top + lm.eps[0]);
	C fe_bot = (lm.eps_bot - lm.eps[n-1])/(lm.eps_bot + lm.eps[n-1]);
	t.check(std::abs(top.Fe - fe_top) < 1e-14 && std::abs(top.Fh) < 1e-14, "downward from the top half-space: Fe " + fmt(top.Fe) + " expected " + fmt(fe_top));
	t.check(std::abs(bot.Fe - fe_bot) < 1e-14 && std::abs(bot.Fh) < 1e-14, "upward from the bottom half-space: Fe " + fmt(bot.Fe) + " expected " + fmt(fe_bot));
});

Reg qs_curl_transform("QS-curl-transform", "spatial quasistatic curl = Sommerfeld transform of the spectral one, across layers", [](T &t) {
	// Lossy source layers keep the branch point of the quasistatic terms off the real axis, so the transform can be
	// integrated along it: (1/2pi) int curlK(krho) J_nu(krho rho) krho^(nu+1) dkrho, nu = 2, 0, 1, 1.
	const double f = 10e9;
	LayerManager lm;
	lm.AddLayer(0.0, 1e-3, {4.0, -1.0}, 1.0, 0.0, 0.0);
	lm.AddLayer(-1e-3, 0.0, {2.0, -0.5}, 1.5, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 6.0, 1.0, 0.0, false, false);
	lm.ProcessLayers(f);
	const int nu[4] = {2, 0, 1, 1};
	struct Cfg { double z, zp; const char *name; };
	for (Cfg c : {Cfg{-0.5e-3, 0.5e-3, "layer 0 -> layer 1"}, Cfg{0.5e-3, -0.5e-3, "layer 1 -> layer 0"},
	              Cfg{2.0e-3, 0.5e-3, "layer 0 -> top half-space"}, Cfg{-2.0e-3, -0.5e-3, "layer 1 -> bottom half-space"}})
		for (bool gem : {true, false})
		{
			const double rho = 0.7e-3;
			QuasistaticMGF q;
			q.Initialize(&lm, f);
			if (gem) q.SetCurlToGEM(); else q.SetCurlToGHJ();
			q.SetLayers(lm.FindLayer(c.zp), lm.FindLayer(c.z));
			q.ComputeCurlQMGF_Spatial(c.z, c.zp, rho);
			std::array<C, 4> spatial, transform;
			for (int n = 0; n < 4; ++n) spatial[n] = q.GetResultCurl_Spatial(n);
			for (int n = 0; n < 4; ++n)
			{
				auto g = [&](double kr) {
					q.ComputeCurlQMGF_Spectral(c.z, c.zp, C(kr, 0.0));
					return q.GetResultCurl_Spectral(n)*boost::math::cyl_bessel_j(nu[n], kr*rho)*std::pow(kr, nu[n] + 1)/(2.0*M_PI);
				};
				boost::math::quadrature::gauss_kronrod<double, 61> rule;
				C sum = 0.0;
				const double w = M_PI/rho;
				for (int k = 0, small = 0; k < 2000 && small < 3; ++k)
				{
					C part = rule.integrate(g, k*w, (k + 1)*w, 15, 1e-13);
					sum += part;
					small = (std::abs(part) < 1e-15*std::abs(sum)) ? small + 1 : 0;
				}
				transform[n] = sum;
			}
			double scale = 0.0, err = 0.0;
			for (int n = 0; n < 4; ++n) scale = std::max(scale, std::abs(transform[n]));
			for (int n = 0; n < 4; ++n) err = std::max(err, std::abs(spatial[n] - transform[n])/scale);
			// (the numerical transform itself is good to about 1e-9; the factor-2 error was >= 3e-2)
			t.check(err < 1e-7, std::string(c.name) + (gem ? " GEM" : " GHJ") + ": max error " + fmt(err, 3) +
			        " (curl 0: " + fmt(spatial[0]) + " vs " + fmt(transform[0]) + ")");
		}
}, 300);

// ================================================================================================
// DCIM
// ================================================================================================

Reg dc_zero("DC-zero-component", "DCIM with an identically zero spectral component stays finite", [](T &t) {
	const double f = 10e9;
	LayerManager lm;
	lm.AddLayer(-0.5e-3, 0.5e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, false, false);
	lm.ProcessLayers(f);
	std::vector<double> zn = {0.0};
	lm.InsertNodes_z(zn);
	MGF mgf;
	MGF_settings s = quiet(MGF_DCIM);
	mgf.Initialize(f, lm, s);
	mgf.SetLayers(0, 0);
	std::array<C, 9> G;
	C phi;
	mgf.ComputeMGF(1e-3, 0.0, 0.0, 0.0, G, phi);
	bool finite = true;
	for (auto g : G) finite = finite && std::isfinite(g.real()) && std::isfinite(g.imag());
	finite = finite && std::isfinite(std::abs(phi));
	t.check(finite, "all components finite (Gxx=" + fmt(G[0]) + ", Gxz=" + fmt(G[2]) + ")");
});

Reg dc_qs("DC-quasistatic-levels", "DCIM with extract_quasistatic agrees with direct integration at every level", [](T &t) {
	// With extract_quasistatic the samples far from the real axis are rounding noise; GPOF images fitted to them blew
	// up on the lower paths (three-level DCIM: errors of 1e3-1e4 and inf in the physics checks).
	const double f = 1e9, rho = 4e-3, z = 0.7e-3, zp = 0.3e-3;
	auto run = [&](int method, int level, bool qs) {
		LayerManager lm = two_layer(f);
		std::vector<double> zn = {zp, z};
		lm.InsertNodes_z(zn);
		MGF mgf;
		MGF_settings s = quiet(method);
		s.DCIM_method = level;
		s.extract_quasistatic = qs;
		s.compute_curl = true;
		mgf.Initialize(f, lm, s);
		mgf.SetLayers(0, 0);
		std::array<C, 9> G, H;
		C p;
		mgf.ComputeMGF(rho*std::cos(0.3), rho*std::sin(0.3), z, zp, G, p);
		mgf.ComputeCurlMGF(rho*std::cos(0.3), rho*std::sin(0.3), z, zp, H);
		std::array<C, 19> out;
		for (int n = 0; n < 9; ++n) { out[n] = G[n]; out[10 + n] = H[n]; }
		out[9] = p;
		return out;
	};
	auto ref = run(MGF_INTEGRATE, 0, false);
	double gs = std::max(std::abs(ref[0]), std::abs(ref[8])), hs = 0.0;
	for (int n = 10; n < 19; ++n) hs = std::max(hs, std::abs(ref[n]));
	for (int level : {2, 3})
		for (bool qs : {false, true})
		{
			auto d = run(MGF_DCIM, level, qs);
			double e = std::abs(d[9] - ref[9])/std::abs(ref[9]), eh = 0.0;
			for (int n = 0; n < 9; ++n) e = std::max(e, std::abs(d[n] - ref[n])/gs);
			for (int n = 10; n < 19; ++n) eh = std::max(eh, std::abs(d[n] - ref[n])/hs);
			t.check(std::isfinite(e) && e < 1e-2 && std::isfinite(eh) && eh < 1e-2, "level " + std::to_string(level) +
			        (qs ? " with" : " without") + " extract_quasistatic: max deviation " + fmt(e, 3) + ", curl " + fmt(eh, 3));
		}
}, 300);

Reg dc_gpof("DC-gpof", "GPOF recovers exact exponentials for any pencil parameter", [](T &t) {
	const int N = 100;
	const double dt = 0.05;
	std::vector<C> amp = {C(1.0, 0.5), C(-0.3, 0.2), C(0.05, 0.0)};
	std::vector<C> s = {C(-0.2, 1.0), C(-1.0, -3.0), C(-0.05, 0.3)};
	std::vector<C> y(N);
	for (int k = 0; k < N; ++k)
		for (size_t j = 0; j < amp.size(); ++j) y[k] += amp[j]*std::exp(s[j]*(k*dt));
	DCIM dcim;
	for (int L : {-1, 30, 60, 80})
	{
		std::vector<C> a, alpha, yy = y;
		dcim.RunGPOF(yy, dt, a, alpha, 1e-10, 1e-16, L);
		double err = 0, ymax = 0;
		for (int k = 0; k < N; ++k)
		{
			C v = 0.0;
			for (size_t j = 0; j < a.size(); ++j) v += a[j]*std::exp(alpha[j]*(k*dt));
			err = std::max(err, std::abs(v - y[k]));
			ymax = std::max(ymax, std::abs(y[k]));
		}
		t.check(err/ymax < 1e-8, "pencil " + std::to_string(L) + ": " + std::to_string(a.size()) + " images, max rel error " + fmt(err/ymax, 3));
	}
	std::vector<C> zero(N, 0.0), a, alpha;
	dcim.RunGPOF(zero, dt, a, alpha);
	bool finite = true;
	for (auto v : a) finite = finite && std::isfinite(std::abs(v));
	for (auto v : alpha) finite = finite && std::isfinite(std::abs(v));
	t.check(finite, "all-zero samples give finite (or no) images: " + std::to_string(a.size()) + " images");
});

Reg dc_curl("DC-curl-not-generated", "ComputeCurlMGF with DCIM but compute_curl=false fails cleanly", [](T &t) {
	const double f = 10e9;
	LayerManager lm = two_layer(f);
	std::vector<double> zn = {0.5e-3};
	lm.InsertNodes_z(zn);
	MGF mgf;
	MGF_settings s = quiet(MGF_DCIM);
	mgf.Initialize(f, lm, s);
	mgf.SetLayers(0, 0);
	std::array<C, 9> G;
	bool thrown = false;
	try { mgf.ComputeCurlMGF(1e-3, 0.0, 0.5e-3, 0.5e-3, G); }
	catch (const std::logic_error &) { thrown = true; }
	t.check(thrown, "logic_error instead of an out-of-range access");
}, 60);

// ================================================================================================
// Interpolation
// ================================================================================================

MGF interp_engine(LayerManager &lm, double f, std::vector<double> z, std::vector<double> rho, bool curl = false, bool interpolate_z = false,
                  int order = 3, int order_z = 3)
{
	lm.InsertNodes_z(z);
	lm.InsertNodes_rho(rho);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTERPOLATE);
	s.compute_curl = curl;
	s.interpolate_z = interpolate_z;
	s.order = order;
	s.order_z = order_z;
	mgf.Initialize(f, lm, s);
	return mgf;
}

Reg in_range("IN-out-of-range", "interpolation outside the tabulated rho / z range does not read out of bounds", [](T &t) {
	const double f = 10e9;
	LayerManager lm;
	lm.AddLayer(0.0, 1e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, false, true);
	lm.ProcessLayers(f);
	std::vector<double> rho;
	strata::linspace(1e-3, 1e-2, 10, rho);
	MGF mgf = interp_engine(lm, f, {0.5e-3}, rho);
	mgf.SetLayers(0, 0);
	std::array<C, 9> G, Gref;
	C phi, phiref;
	mgf.ComputeMGF(1e-2, 0.0, 0.5e-3, 0.5e-3, Gref, phiref);
	for (double r : {2e-2, 5e-4})
	{
		bool thrown = false;
		try { mgf.ComputeMGF(r, 0.0, 0.5e-3, 0.5e-3, G, phi); }
		catch (const std::out_of_range &) { thrown = true; }
		t.check(thrown, "rho=" + fmt(r) + " outside [1e-3, 1e-2] raises out_of_range");
	}
	mgf.ComputeMGF(1e-2*(1.0 + 1e-13), 0.0, 0.5e-3, 0.5e-3, G, phi);
	t.check(rel(G[0], Gref[0]) < 1e-9, "rho a rounding error above the last node is accepted");
	// z below the only node of layer 0 must map to that node (nearest-node rule), not read before the map.
	mgf.ComputeMGF(5e-3, 0.0, 0.2e-3, 0.5e-3, G, phi);
	std::array<C, 9> Gn;
	C pn;
	mgf.ComputeMGF(5e-3, 0.0, 0.5e-3, 0.5e-3, Gn, pn);
	t.check(rel(G[0], Gn[0]) < 1e-12, "z below the lowest node uses the nearest node");
}, 300);

Reg in_zid("IN-adaptive-z-ids", "adding a z-node keeps z-node ids unique and rows consistent", [](T &t) {
	const double f = 10e9;
	LayerManager lm;
	for (int l = 0; l < 5; ++l)
		lm.AddLayer(-(l + 1)*0.2e-3, -l*0.2e-3, {2.0 + l, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, false, false);
	lm.ProcessLayers(f);
	lm.z_nodes.assign(5, {});
	for (int l = 0; l < 5; ++l) lm.z_nodes[l] = {-(l + 1)*0.2e-3 + 0.05e-3, -l*0.2e-3 - 0.05e-3};
	std::vector<double> rho = {1e-3, 2e-3, 3e-3, 4e-3};
	lm.InsertNodes_rho(rho);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTERPOLATE);
	s.interpolate_z = true;
	s.order = 1;
	s.order_z = 1;
	mgf.Initialize(f, lm, s);
	double znew = -0.1e-3;
	mgf.AddZTable(znew, 0);
	std::set<int> ids;
	for (auto &kv : mgf.z_to_idx) ids.insert(kv.second);
	t.check(ids.size() == mgf.z_to_idx.size(), "z-node ids unique (" + std::to_string(ids.size()) + " of " + std::to_string(mgf.z_to_idx.size()) + ")");
	// Table entry used for (z = znew, zp = a node of the last layer) equals a direct integration.
	double zp = mgf.lm.z_nodes[4][1];
	mgf.SetLayers(4, 0);
	std::array<C, 9> G;
	C phi;
	mgf.ComputeMGF(2e-3, 0.0, znew, zp, G, phi);
	std::array<C, 5> direct;
	mgf.smgf.SetLayers(4, 0);
	mgf.ComputeMGF_Integration(2e-3, znew, zp, direct);
	t.check(rel(G[0], direct[0]) < 1e-9 && rel(phi, direct[4]) < 1e-9, "row for (znew, zp) holds that pair: Gxx " + fmt(G[0]) + " vs " + fmt(direct[0]));
}, 600);

Reg in_curl_sync("IN-curl-table-sync", "adaptive rho refinement keeps the curl table in sync", [](T &t) {
	const double f = 10e9;
	LayerManager lm = two_layer(f);
	std::vector<double> rho = {1e-3, 2e-3, 3e-3, 4e-3, 5e-3};
	MGF mgf = interp_engine(lm, f, {0.5e-3}, rho, true);
	mgf.AddRhoTable(4.5e-3);
	t.check(mgf.MGF_table[0].size() == mgf.lm.rho_nodes.size(), "MGF table has one column per rho node");
	t.check(mgf.CurlMGF_table[0].size() == mgf.lm.rho_nodes.size(), "curl table has one column per rho node (" +
	        std::to_string(mgf.CurlMGF_table[0].size()) + " vs " + std::to_string(mgf.lm.rho_nodes.size()) + ")");
}, 600);

Reg in_curl_z("IN-curl-interpolate-z", "ComputeCurlMGF honours interpolate_z", [](T &t) {
	const double f = 10e9;
	LayerManager lm;
	lm.AddLayer(0.0, 1e-3, {4.0, 0.0}, 1.0, 0.0, 0.0);
	lm.SetHalfspaces(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, false, true);
	lm.ProcessLayers(f);
	std::vector<double> rho = {1e-3, 2e-3, 3e-3, 4e-3};
	MGF mgf = interp_engine(lm, f, {0.2e-3, 0.8e-3}, rho, true, true, 1, 1);
	mgf.SetLayers(0, 0);
	std::array<C, 9> a, b, mid;
	mgf.ComputeCurlMGF(2e-3, 0.0, 0.2e-3, 0.2e-3, a);
	mgf.ComputeCurlMGF(2e-3, 0.0, 0.8e-3, 0.2e-3, b);
	mgf.ComputeCurlMGF(2e-3, 0.0, 0.5e-3, 0.2e-3, mid);
	double e = 0;
	for (int n = 0; n < 9; ++n)
		if (std::abs(a[n]) + std::abs(b[n]) > 0) e = std::max(e, std::abs(mid[n] - 0.5*(a[n] + b[n]))/std::max(std::abs(a[n]), std::abs(b[n])));
	t.check(e < 1e-12, "midpoint equals the linear interpolation of the two nodes (dev " + fmt(e, 3) + ")");
}, 600);

Reg in_updatez("IN-update-z-empty-layer", "adaptive z refinement with a layer that has no z-nodes", [](T &t) {
	const double f = 10e9;
	LayerManager lm = two_layer(f);
	lm.z_nodes.assign(lm.layers.size(), {});
	lm.z_nodes[0] = {0.2e-3, 0.5e-3, 0.8e-3};
	std::vector<double> rho = {1e-3, 2e-3, 3e-3};
	lm.InsertNodes_rho(rho);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTERPOLATE);
	s.interpolate_z = true;
	s.update_z_nodes = true;
	s.order = 1;
	s.order_z = 1;
	s.adaptive_threshold = 1e30;
	mgf.Initialize(f, lm, s);
	bool ok = true;
	try { mgf.AdaptiveInterpolation(); }
	catch (const std::exception &) { ok = true; }
	t.check(ok, "AdaptiveInterpolation returns");
}, 300);

// ================================================================================================
// Floating-point hygiene
// ================================================================================================

Reg fe_flags("FE-halfspace", "half-space spectral kernels do not raise FE_INVALID / FE_OVERFLOW", [](T &t) {
	LayerManager lm = two_layer();
	struct Cfg { int i, m; double zp, z; C kr; };
	for (Cfg c : {Cfg{0, -1, 0.5e-3, 2e-3, C(0.0, 5.0)}, Cfg{0, -1, 0.5e-3, 1.0, C(1e4, 0.0)}, Cfg{0, 2, 0.5e-3, -1.0, C(1e4, 0.0)},
	              Cfg{-1, -1, 1.0, 1.1, C(80.0, 0.0)}, Cfg{2, 2, -2e-3, -3e-3, C(80.0, 0.0)}})
	{
		SpectralMGF s;
		setup_spec(s, lm, c.i, c.m, c.zp, c.z, c.kr);
		std::feclearexcept(FE_ALL_EXCEPT);
		s.ComputeSpectralMGF();
		bool inv = std::fetestexcept(FE_INVALID), ovf = std::fetestexcept(FE_OVERFLOW);
		bool finite = true;
		for (int n = 0; n < 5; ++n) finite = finite && std::isfinite(std::abs(s.GetResult(n)));
		t.check(!inv && !ovf && finite, "i=" + std::to_string(c.i) + " m=" + std::to_string(c.m) + " krho=" + fmt(c.kr) +
		        ": FE_INVALID=" + std::to_string(inv) + " FE_OVERFLOW=" + std::to_string(ovf) + " finite=" + std::to_string(finite));
	}
});

// ================================================================================================
// Further checks (including findings of the Codex review of the same revision)
// ================================================================================================

// Strata's curl in a homogeneous medium (G_HJ = G_EM in Strata's sign convention, the negative of the H-field kernel):
// scalars G1 = 2 dz (1 + jkR) e^{-jkR}/(4 pi R^3), G2 = G3 = rho (1 + jkR) e^{-jkR}/(4 pi R^3), G0 = 0.
std::array<C, 9> homogeneous_curl(C k, double x, double y, double dz)
{
	double rho = std::hypot(x, y), R = std::sqrt(rho*rho + dz*dz), phi = std::atan2(y, x);
	C g = (1.0 + strata::J*k*R)*std::exp(-strata::J*k*R)/(4.0*M_PI*std::pow(R, 3));
	C G1 = 2.0*dz*g, G2 = rho*g, G3 = rho*g;
	return {0.0, -G1/2.0, G2*std::sin(phi), G1/2.0, 0.0, -G2*std::cos(phi), -G3*std::sin(phi), G3*std::cos(phi), 0.0};
}

Reg ex_curl("EX-curl-homogeneous", "curl in a homogeneous medium with and without extraction", [](T &t) {
	const double f = 2e9;
	LayerManager lm = homogeneous(3.0, 1.5, 0.0, -0.02, 0.02, f);
	for (int variant = 0; variant < 3; ++variant)
		for (int ghj = 0; ghj < 2; ++ghj)
		{
			MGF mgf;
			MGF_settings s = quiet(MGF_INTEGRATE);
			s.compute_curl = true;
			s.curl_type = ghj ? MGF_GHJ : MGF_GEM;
			s.extract_quasistatic = (variant == 1);
			s.extract_homogeneous = (variant == 2);
			mgf.Initialize(f, lm, s);
			mgf.SetLayers(0, 0);
			std::array<C, 9> G;
			double x = 3e-3, y = 2e-3, z = 1e-3, zp = -2e-3;
			mgf.ComputeCurlMGF(x, y, z, zp, G);
			auto ref = homogeneous_curl(mgf.lm.k[0], x, y, z - zp);
			double scale = 0, err = 0;
			for (int n = 0; n < 9; ++n) scale = std::max(scale, std::abs(ref[n]));
			for (int n = 0; n < 9; ++n) err = std::max(err, std::abs(G[n] - ref[n])/scale);
			std::string name = std::string(variant == 0 ? "plain" : variant == 1 ? "extract_quasistatic" : "extract_homogeneous") + (ghj ? " G_HJ" : " G_EM");
			t.check(err < 1e-6, name + ": max rel error " + fmt(err, 3));
		}
}, 300);

Reg ex_mask("EX-components-quasistatic", "SetComponents() with extract_quasistatic keeps the enabled components exact", [](T &t) {
	const double f = 1e9;
	LayerManager lm = two_layer(f);
	MGF mgf;
	MGF_settings s = quiet(MGF_INTEGRATE);
	s.extract_quasistatic = true;
	mgf.Initialize(f, lm, s);
	mgf.SetLayers(0, 0);
	std::array<C, 9> ref, G;
	C pref, p;
	mgf.ComputeMGF(1e-3, 0.5e-3, 0.7e-3, 0.3e-3, ref, pref);
	mgf.SetComponents({1, 0, 1, 0, 1});
	mgf.ComputeMGF(1e-3, 0.5e-3, 0.7e-3, 0.3e-3, G, p);
	t.check(rel(G[0], ref[0]) < 1e-9 && rel(G[6], ref[6]) < 1e-9 && rel(p, pref) < 1e-9,
	        "mask {1,0,1,0,1}: Gxx " + fmt(rel(G[0], ref[0]), 3) + " Gzx " + fmt(rel(G[6], ref[6]), 3) + " Gphi " + fmt(rel(p, pref), 3));
	mgf.SetComponents({1, 1, 1, 1, 1});
	mgf.ComputeMGF(1e-3, 0.5e-3, 0.7e-3, 0.3e-3, G, p);
	double e = 0;
	for (int n : {0, 2, 6, 8}) e = std::max(e, rel(G[n], ref[n]));
	e = std::max(e, rel(p, pref));
	t.check(e < 1e-9, "all components re-enabled: max rel " + fmt(e, 3));
}, 300);

Reg sp_reinit("SP-reinit-curl-type", "SpectralMGF::Initialize() keeps the nested quasistatic engine's curl type", [](T &t) {
	LayerManager lm = two_layer();
	const C kr(80.0, 10.0);
	SpectralMGF a, b;
	a.Initialize(&lm, 1e9, {1, 1, 1, 1, 1}, true);
	a.SetCurlToGHJ();
	b.SetCurlToGHJ();
	b.Initialize(&lm, 1e9, {1, 1, 1, 1, 1}, true);
	for (SpectralMGF *x : {&a, &b})
	{
		x->SetLayers(0, 0);
		x->SetSourcePoint(0.3e-3);
		x->SetObservationPoint(0.7e-3);
		x->SetRadialWaveNumber(kr);
		x->ComputeCurlSpectralMGF();
	}
	double e = 0;
	for (int n = 0; n < 4; ++n) e = std::max(e, rel(b.GetResultCurl(n), a.GetResultCurl(n)));
	t.check(e < 1e-12, "curl type set before Initialize(): max rel " + fmt(e, 3));
});

Reg lm_sigmamu("LM-sigmamu", "the magnetic conductivity sigmamu enters the permeability", [](T &t) {
	LayerManager lm;
	lm.AddLayer(0.0, 1e-3, {4.0, 0.0}, 2.0, 0.0, 100.0);
	lm.ProcessLayers(1e9);
	double expected = -100.0/(2.0*M_PI*1e9);
	t.check(std::abs(lm.mu[0].imag() - expected) < 1e-12*std::abs(expected) && std::abs(lm.mu[0].real() - 2.0*strata::mu0) < 1e-20,
	        "mu = " + fmt(lm.mu[0], 8) + ", expected imaginary part " + fmt(expected, 8));
});

Reg lm_ds("LM-ds-dc-permittivity", "relative_permittivity_at_dc of the Djordjevic-Sarkar model is honoured", [](T &t) {
	std::vector<double> eps_low;
	for (double dc : {-1.0, 4.8, 6.0})
	{
		DjordjevicSarkarParams ds;
		ds.relative_permittivity = 4.4;
		ds.loss_tangent = 0.02;
		ds.at_frequency = 1e9;
		ds.relative_permittivity_at_dc = dc;
		LayerManager lm;
		lm.AddLayer(0.0, 1e-3, {4.4, 0.0}, 1.0, 0.0, 0.0, DielectricModelType::DJORDJEVIC_SARKAR, ds);
		lm.ProcessLayers(1e9);
		C e = lm.eps[0]/strata::eps0;
		t.check(std::abs(e.real() - 4.4) < 1e-9 && std::abs(-e.imag()/e.real() - 0.02) < 1e-9,
		        "dc=" + fmt(dc) + ": anchors at 1 GHz kept (" + fmt(e, 8) + ")");
		// The lower corner frequency that reaches dc = 6 lies near 0.4 mHz (tan d = 0.02 gives about 0.056 per neper
		// of frequency), so the DC limit is checked far below it.
		lm.ProcessLayers(1e-12);
		eps_low.push_back((lm.eps[0]/strata::eps0).real());
		if (dc > 0)
			t.check(std::abs(eps_low.back() - dc) < 1e-6*dc, "dc=" + fmt(dc) + ": eps(1e-12 Hz) = " + fmt(eps_low.back(), 8));
	}
	t.check(std::abs(eps_low[1] - eps_low[2]) > 0.5, "different DC permittivities give different models");
});

Reg lm_stale("LM-stale-frequency", "low-level engines refuse layer data processed at another frequency", [](T &t) {
	LayerManager lm = two_layer(1e9);
	bool thrown = false;
	try { SpectralMGF s; s.Initialize(&lm, 2e9); }
	catch (const std::logic_error &) { thrown = true; }
	t.check(thrown, "SpectralMGF::Initialize at 2 GHz with layers processed at 1 GHz");
	thrown = false;
	try { QuasistaticMGF q; q.Initialize(&lm, 2e9); }
	catch (const std::logic_error &) { thrown = true; }
	t.check(thrown, "QuasistaticMGF::Initialize likewise");
	bool ok = true;
	try { SpectralMGF s; s.Initialize(&lm, 1e9); }
	catch (const std::exception &) { ok = false; }
	t.check(ok, "same frequency accepted");
});

Reg in_file("IN-table-file", "a table file is only reused for the configuration it was written for", [](T &t) {
	std::string file = "/tmp/strata_unit_table_" + std::to_string(getpid()) + ".bin";
	std::vector<double> rho = {1e-3, 2e-3, 3e-3, 4e-3};
	auto make = [&](double f, bool exp, bool load) {
		LayerManager lm = two_layer(f);
		std::vector<double> zn = {0.5e-3};
		lm.InsertNodes_z(zn);
		lm.InsertNodes_rho(rho);
		MGF mgf;
		MGF_settings s = quiet(MGF_INTERPOLATE);
		s.order = 1;
		s.filename = file;
		s.export_table = exp;
		s.load_table = load;
		mgf.Initialize(f, lm, s);
		mgf.SetLayers(0, 0);
		std::array<C, 9> G;
		C p;
		mgf.ComputeMGF(2e-3, 0.0, 0.5e-3, 0.5e-3, G, p);
		return G[0];
	};
	C g1 = make(1e9, true, false);
	C g1_loaded = make(1e9, false, true);
	C g2_fresh = make(3e9, false, false);
	C g2_loaded = make(3e9, false, true);
	std::remove(file.c_str());
	t.check(rel(g1_loaded, g1) < 1e-14, "same configuration: table reused");
	t.check(rel(g2_loaded, g2_fresh) < 1e-12 && rel(g2_loaded, g1) > 1e-3, "other frequency: table regenerated (" + fmt(g2_loaded) + " vs " + fmt(g2_fresh) + ")");
}, 300);

Reg in_curl_missing("IN-curl-not-generated", "ComputeCurlMGF with interpolation but compute_curl=false fails cleanly", [](T &t) {
	const double f = 10e9;
	LayerManager lm = two_layer(f);
	std::vector<double> rho = {1e-3, 2e-3, 3e-3, 4e-3};
	MGF mgf = interp_engine(lm, f, {0.5e-3}, rho, false);
	mgf.SetLayers(0, 0);
	std::array<C, 9> G;
	bool thrown = false;
	try { mgf.ComputeCurlMGF(2e-3, 0.0, 0.5e-3, 0.5e-3, G); }
	catch (const std::logic_error &) { thrown = true; }
	t.check(thrown, "logic_error instead of reading an empty table");
}, 120);

Reg in_mask("IN-components-not-precomputed", "SetComponents() cannot enable a component without DCIM images or table entries", [](T &t) {
	const double f = 10e9;
	auto throws = [](MGF &mgf, std::vector<bool> c, bool curl) -> int {
		try { curl ? mgf.SetCurlComponents(c) : mgf.SetComponents(c); }
		catch (const std::invalid_argument &) { return 2; }
		catch (const std::logic_error &) { return 1; }
		return 0;
	};
	// DCIM: images for components 1 and 3 do not exist (they used to evaluate to zero after being re-enabled)
	{
		LayerManager lm = two_layer(f);
		std::vector<double> zn = {0.5e-3};
		lm.InsertNodes_z(zn);
		MGF mgf;
		MGF_settings s = quiet(MGF_DCIM);
		s.components = {1, 0, 1, 0, 1};
		mgf.Initialize(f, lm, s);
		t.check(throws(mgf, {1, 1, 1, 1, 1}, false) == 1, "DCIM: enabling a component without images throws");
		t.check(throws(mgf, {1, 0, 0, 0, 1}, false) == 0 && throws(mgf, {1, 0, 1, 0, 1}, false) == 0,
		        "DCIM: disabling and re-enabling a component with images is allowed");
		t.check(throws(mgf, {1, 1}, false) == 2, "DCIM: a mask of the wrong length is rejected");
	}
	// Interpolation: rows appended while a component is disabled lack it
	{
		LayerManager lm = two_layer(f);
		MGF mgf = interp_engine(lm, f, {0.5e-3}, {1e-3, 2e-3, 3e-3, 4e-3});
		t.check(throws(mgf, {1, 0, 1, 1, 1}, false) == 0, "interpolation: disabling a component is allowed");
		mgf.AddRhoTable(2.5e-3);
		t.check(throws(mgf, {1, 1, 1, 1, 1}, false) == 1, "interpolation: after adaptive refinement without component 1 it cannot be re-enabled");
	}
	// Direct integration computes everything on demand
	{
		LayerManager lm = two_layer(f);
		MGF mgf;
		MGF_settings s = quiet(MGF_INTEGRATE);
		s.components = {1, 0, 0, 0, 0};
		s.components_curl = {0, 0, 0, 0};
		mgf.Initialize(f, lm, s);
		t.check(throws(mgf, {1, 1, 1, 1, 1}, false) == 0 && throws(mgf, {1, 1, 1, 1}, true) == 0, "integration: any mask");
	}
}, 300);

Reg qs_hs("QS-halfspace-finite", "quasistatic terms with the source in a half-space are finite", [](T &t) {
	LayerManager lm = two_layer();
	QuasistaticMGF q;
	q.Initialize(&lm, 1e9);
	for (int src : {-1, 2})
		for (int obs : {src, 0, 1})
		{
			q.SetLayers(src, obs);
			double zp = src == -1 ? 2e-3 : -2e-3;
			double z = obs == src ? (src == -1 ? 3e-3 : -3e-3) : (obs == 0 ? 0.5e-3 : -0.5e-3);
			bool finite = true;
			for (C kr : {C(0.0, 5.0), C(30.0, 0.0), C(1e4, 0.0)})
			{
				q.ComputeQMGF_Spectral(z, zp, kr);
				for (int n = 0; n < 5; ++n) finite = finite && std::isfinite(std::abs(q.GetResult_Spectral(n)));
				q.ComputeCurlQMGF_Spectral(z, zp, kr);
				for (int n = 0; n < 4; ++n) finite = finite && std::isfinite(std::abs(q.GetResultCurl_Spectral(n)));
			}
			q.ComputeQMGF_Spatial(z, zp, 1e-3);
			for (int n = 0; n < 5; ++n) finite = finite && std::isfinite(std::abs(q.GetResult_Spatial(n)));
			q.ComputeCurlQMGF_Spatial(z, zp, 1e-3);
			for (int n = 0; n < 4; ++n) finite = finite && std::isfinite(std::abs(q.GetResultCurl_Spatial(n)));
			t.check(finite, "source layer " + std::to_string(src) + ", observation layer " + std::to_string(obs));
		}
});

} // namespace


int main(int argc, char **argv)
{
	std::string filter = argc > 1 ? argv[1] : "";
	// Time limits can be stretched for slow (e.g. sanitizer) builds: STRATA_TEST_TIMEOUT_SCALE=10
	double scale = std::getenv("STRATA_TEST_TIMEOUT_SCALE") ? std::atof(std::getenv("STRATA_TEST_TIMEOUT_SCALE")) : 1.0;
	if (!(scale > 0.0))
		scale = 1.0;
	int failed = 0, run = 0;
	for (const auto &c : registry())
	{
		if (!filter.empty() && c.id.find(filter) == std::string::npos)
			continue;
		++run;
		int fd[2];
		if (pipe(fd) != 0) { perror("pipe"); return 255; }
		std::cout.flush();
		pid_t pid = fork();
		if (pid == 0)
		{
			close(fd[0]);
			alarm((unsigned)std::ceil(c.timeout*scale));
			T t;
			std::string status;
			try
			{
				c.fn(t);
			}
			catch (const std::exception &e)
			{
				t.check(false, std::string("unexpected exception: ") + e.what());
			}
			std::string out = (t.ok ? "PASS" : "FAIL") + std::string("\n") + t.log.str();
			ssize_t w = write(fd[1], out.data(), out.size());
			(void)w;
			close(fd[1]);
			_exit(t.ok ? 0 : 1);
		}
		close(fd[1]);
		std::string out;
		char buf[4096];
		ssize_t r;
		while ((r = read(fd[0], buf, sizeof(buf))) > 0) out.append(buf, r);
		close(fd[0]);
		int st = 0;
		waitpid(pid, &st, 0);
		std::string verdict, details;
		if (WIFSIGNALED(st))
		{
			int sig = WTERMSIG(st);
			verdict = sig == SIGALRM ? "TIMEOUT" : "CRASH";
			details = "      " + std::string(sig == SIGALRM ? "time limit " + std::to_string((unsigned)std::ceil(c.timeout*scale)) + " s exceeded" : "signal " + std::to_string(sig) + " (" + strsignal(sig) + ")") + "\n";
			auto nl = out.find('\n');
			if (nl != std::string::npos) details += out.substr(nl + 1);
		}
		else
		{
			auto nl = out.find('\n');
			verdict = out.substr(0, nl);
			details = nl == std::string::npos ? "" : out.substr(nl + 1);
			if (verdict.empty()) verdict = "CRASH";
		}
		if (verdict != "PASS") ++failed;
		std::cout << "[" << verdict << "] " << c.id << "  " << c.title << "\n" << details << std::flush;
	}
	std::cout << "\n" << run - failed << " of " << run << " tests passed." << std::endl;
	return std::min(failed, 255);
}
