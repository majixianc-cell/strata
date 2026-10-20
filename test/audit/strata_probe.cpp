// Strata audit probe.
//
// Reads a job file (JSON or YAML), builds the stack-up and the MGF engine, evaluates the requested
// queries and prints one JSON object per query on stdout. Used by the Python checks in test/audit/.
//
//   strata_probe JOB
//
// Job keys:
//   stack:    {file: tech.yaml}  or  {layers: [[zmin, zmax, epsr, epsr_im, mur, sigma], ...],
//                                     top: [epsr, mur, sigma, pec], bot: [epsr, mur, sigma, pec]}
//   freq:     frequency (Hz)
//   settings: MGF_settings fields (method: INTEGRATE | DCIM | INTERPOLATE | QUASISTATIC, curl_type: GEM | GHJ,
//             sampling_method: INTEGRATE | DCIM, DCIM_method: 1 | 2 | 3, booleans as 0/1, ...)
//   znodes:   [z, ...]                         (LayerManager::InsertNodes_z, after ProcessLayers)
//   rhonodes: [rho, ...]                       (LayerManager::InsertNodes_rho)
//   init:     0 to skip MGF::Initialize (only spectral / quasistatic / refint queries then)
//   queries:  list of [kind, args...]:
//     [mgf, x, y, z, zp, i, m]          MGF::ComputeMGF (+ ComputeCurlMGF if compute_curl); i, m = -999: FindLayer
//     [qmgf, x, y, z, zp, i, m]         MGF::ComputeQMGF (+ ComputeCurlQMGF if compute_curl)
//     [spec, kre, kim, z, zp, i, m]     fresh SpectralMGF objects: K[0..4], curl GEM[0..3], curl GHJ[0..3]
//     [qspec, kre, kim, z, zp, i, m]    fresh QuasistaticMGF: spectral K[0..4], curl GEM/GHJ
//     [qspat, rho, z, zp, i, m]         fresh QuasistaticMGF: spatial K[0..4], curl GEM/GHJ
//     [refint, rho, z, zp, i, m, kind, comp]
//                                       independent Sommerfeld integration (elliptic path + Bessel-zero
//                                       partitions + Levin-Sidi) of Strata's own spectral kernel:
//                                       kind 0 = MGF component comp (0..4), 1 = GEM curl, 2 = GHJ curl
//     [findlayer, z]                    LayerManager::FindLayer on the processed stack-up
//
// Every output line carries "q" (query index) and either the values or "err".

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/special_functions/bessel.hpp>

#include "yaml-cpp/yaml.h"
#include "libAmosBessel.h"

#include "MGF.hpp"
#include "constants.hpp"

using C = std::complex<double>;

namespace
{

std::string num(double v)
{
	if (std::isnan(v))
		return "NaN";
	if (std::isinf(v))
		return v > 0 ? "Infinity" : "-Infinity";
	std::ostringstream o;
	o << std::setprecision(17) << v;
	return o.str();
}

std::string cnum(C v) { return "[" + num(v.real()) + "," + num(v.imag()) + "]"; }

template<class Seq>
std::string carr(const Seq &v)
{
	std::string s = "[";
	bool first = true;
	for (const auto &x : v)
	{
		s += (first ? "" : ",") + cnum(x);
		first = false;
	}
	return s + "]";
}

std::string quote(const std::string &s)
{
	std::string out = "\"";
	for (char c : s)
	{
		if (c == '"' || c == '\\')
			out += '\\';
		if (c == '\n')
		{
			out += "\\n";
			continue;
		}
		out += c;
	}
	return out + "\"";
}

int method_of(const std::string &m)
{
	if (m == "INTEGRATE") return MGF_INTEGRATE;
	if (m == "DCIM") return MGF_DCIM;
	if (m == "INTERPOLATE") return MGF_INTERPOLATE;
	if (m == "QUASISTATIC") return MGF_QUASISTATIC;
	throw std::invalid_argument("unknown method " + m);
}

void read_settings(const YAML::Node &n, MGF_settings &s)
{
	s.verbose = false;
	if (!n)
		return;
	for (auto it = n.begin(); it != n.end(); ++it)
	{
		const std::string k = it->first.as<std::string>();
		const YAML::Node &v = it->second;
		if (k == "method") s.method = method_of(v.as<std::string>());
		else if (k == "sampling_method") s.sampling_method = method_of(v.as<std::string>());
		else if (k == "curl_type") s.curl_type = (v.as<std::string>() == "GHJ") ? MGF_GHJ : MGF_GEM;
		else if (k == "interpolate_z") s.interpolate_z = v.as<int>();
		else if (k == "extract_quasistatic") s.extract_quasistatic = v.as<int>();
		else if (k == "extract_singularities") s.extract_singularities = v.as<int>();
		else if (k == "extract_homogeneous") s.extract_homogeneous = v.as<int>();
		else if (k == "verbose") s.verbose = v.as<int>();
		else if (k == "compute_curl") s.compute_curl = v.as<int>();
		else if (k == "DCIM_method") s.DCIM_method = v.as<int>();
		else if (k == "order") s.order = v.as<int>();
		else if (k == "order_z") s.order_z = v.as<int>();
		else if (k == "N_lambda") s.N_lambda = v.as<double>();
		else if (k == "adaptive_threshold") s.adaptive_threshold = v.as<double>();
		else if (k == "update_z_nodes") s.update_z_nodes = v.as<int>();
		else if (k == "filename") s.filename = v.as<std::string>();
		else if (k == "filename_curl") s.filename_curl = v.as<std::string>();
		else if (k == "load_table") s.load_table = v.as<int>();
		else if (k == "export_table") s.export_table = v.as<int>();
		else if (k == "tol_qse") s.tol_qse = v.as<double>();
		else if (k == "switching_point") s.switching_point = v.as<double>();
		else if (k == "tol_quadrature") s.tol_quadrature = v.as<double>();
		else if (k == "tol_svd") s.tol_svd = v.as<double>();
		else if (k == "tol_eig") s.tol_eig = v.as<double>();
		else if (k == "max_num_images") s.max_num_images = v.as<int>();
		else if (k == "components")
		{
			s.components.clear();
			for (auto c : v) s.components.push_back(c.as<int>());
		}
		else if (k == "components_curl")
		{
			s.components_curl.clear();
			for (auto c : v) s.components_curl.push_back(c.as<int>());
		}
		else
			throw std::invalid_argument("unknown setting " + k);
	}
}

void build_stack(const YAML::Node &st, LayerManager &lm)
{
	if (st["file"])
	{
		lm.ProcessTechFile(st["file"].as<std::string>());
		return;
	}
	for (auto l : st["layers"])
		lm.AddLayer(l[0].as<double>(), l[1].as<double>(), C(l[2].as<double>(), l[3].as<double>()),
		            l[4].as<double>(), l[5].as<double>(), 0.0);
	auto hs = [](const YAML::Node &n, double &e, double &mu, double &sig, bool &pec)
	{
		e = 1.0; mu = 1.0; sig = 0.0; pec = false;
		if (!n) return;
		e = n[0].as<double>(); mu = n[1].as<double>(); sig = n[2].as<double>(); pec = n[3].as<int>();
	};
	double et, mt, st_, eb, mb, sb;
	bool pt, pb;
	hs(st["top"], et, mt, st_, pt);
	hs(st["bot"], eb, mb, sb, pb);
	lm.SetHalfspaces(et, mt, st_, eb, mb, sb, pt, pb);
}

// ------------------------------------------------------------------------------------------------
// Independent reference integration of Strata's spectral kernels.
// Path: half ellipse kρ = (A/2)(1 - cos θ) + i H sin θ, θ in [0, π] (first quadrant, above every pole and
// branch point), then the real axis from A: geometric panels up to the first J_ν zero beyond max(A, 40/d), then
// partitions between consecutive exact Bessel zeros (Boost) with a Levin-Sidi transformation of the partial sums.
// Every kernel value comes from a freshly configured SpectralMGF (canonical setter order), so no cached state
// is shared with Strata's own integration.
// ------------------------------------------------------------------------------------------------

struct RefIntegral
{
	C value = 0.0;
	double err = 0.0;
	int partitions = 0;
	bool converged = false;
};

C BesselJ(double nu, C x)
{
	__complex__ double in;
	__real__ in = x.real();
	__imag__ in = x.imag();
	C out;
	int info = AmosBessel('J', in, nu, 1, 0, (__complex__ double *)&out);
	if (info != 0)
		throw std::runtime_error("AmosBessel failed");
	return out;
}

struct RefKernel
{
	LayerManager *lm;
	double f;
	MGF_settings s;
	int i, m;
	double z, zp;
	int kind, comp;

	C operator()(C krho) const
	{
		SpectralMGF sm;
		sm.Initialize(lm, f, s.components, s.extract_quasistatic, s.extract_homogeneous);
		if (kind == 1) sm.SetCurlToGEM();
		if (kind == 2) sm.SetCurlToGHJ();
		sm.SetLayers(i, m);
		sm.SetSourcePoint(zp);
		sm.SetObservationPoint(z);
		sm.SetRadialWaveNumber(krho);
		if (kind == 0)
		{
			sm.ComputeSpectralMGF();
			return sm.GetResult(comp);
		}
		sm.ComputeCurlSpectralMGF();
		return sm.GetResultCurl(comp);
	}
};

template<class F>
C gk(const F &f, double a, double b, double tol, double &err)
{
	boost::math::quadrature::gauss_kronrod<double, 31> rule;
	double e = 0.0;
	C v = rule.integrate(f, a, b, 20, tol, &e);
	err += e;
	return v;
}

RefIntegral ref_integrate(const RefKernel &K, double rho, int order)
{
	RefIntegral out;
	const double tol = 1e-12;
	double kmax = 0.0;
	for (auto k : K.lm->k) kmax = std::max(kmax, std::abs(k));
	kmax = std::max(kmax, std::abs(K.lm->k_top));
	kmax = std::max(kmax, std::abs(K.lm->k_bot));
	double d = std::abs(K.z - K.zp);
	const double A = 1.5*kmax;
	double H = 0.4*A;
	if (rho > 0.0) H = std::min(H, 1.0/rho);

	auto integrand = [&](C krho) { return K(krho)*BesselJ(order, krho*rho)*std::pow(krho, order + 1)/(2.0*M_PI); };

	// Half ellipse.
	auto fe = [&](double th) {
		C kr(0.5*A*(1.0 - std::cos(th)), H*std::sin(th));
		C dk(0.5*A*std::sin(th), H*std::cos(th));
		return integrand(kr)*dk;
	};
	double err = 0.0;
	C total = 0.0;
	const int nseg = 8;
	for (int s = 0; s < nseg; ++s)
		total += gk(fe, M_PI*s/nseg, M_PI*(s + 1)/nseg, tol, err);

	auto fr = [&](double x) { return integrand(C(x, 0.0)); };

	// Real axis. Geometric panels until the first Bessel zero beyond a point where the exponential decay
	// (if any) has killed the integrand, unless that point is many oscillations away; then Bessel-zero
	// partitions with Levin-Sidi extrapolation.
	double left = A;
	double target = A;
	if (d > 0.0 && (rho == 0.0 || 40.0/d < A + 200.0*M_PI/rho)) target = std::max(target, 40.0/d);
	double first_zero = target;
	if (rho > 0.0)
	{
		int n = std::max(1, (int)std::floor(target*rho/M_PI - 0.5*order));
		double zz = boost::math::cyl_bessel_j_zero((double)order, n)/rho;
		while (zz <= target) zz = boost::math::cyl_bessel_j_zero((double)order, ++n)/rho;
		first_zero = zz;
	}
	double width = std::max(A, d > 0.0 ? 1.0/d : A);
	if (rho > 0.0) width = std::min(width, M_PI/rho);
	while (left < first_zero)
	{
		double right = std::min(first_zero, left + width);
		total += gk(fr, left, right, tol, err);
		left = right;
		if (rho == 0.0 || width < 0.5*M_PI/rho) width *= 1.5;
		if (++out.partitions > 100000) throw std::runtime_error("refint: too many panels");
	}
	if (rho == 0.0)
	{
		// No oscillation: continue until negligible (requires d > 0).
		if (d == 0.0) throw std::runtime_error("refint: rho = 0 and z = zp is singular");
		out.value = total;
		out.err = err;
		out.converged = true;
		return out;
	}

	// Partition-extrapolation over exact zeros (Levin-Sidi, remainder estimates w_k = u_k).
	const int kmax_part = 400;
	std::vector<double> X(kmax_part + 2);
	std::vector<C> Aa(kmax_part + 1), Bb(kmax_part + 1);
	int n0 = 1;
	while (boost::math::cyl_bessel_j_zero((double)order, n0)/rho < left*(1.0 - 1e-14)) ++n0;
	X[0] = left;
	C s = 0.0, old = 1e300, val = 0.0;
	double perr = 0.0;
	for (int k = 1; k <= kmax_part; ++k)
	{
		X[k] = boost::math::cyl_bessel_j_zero((double)order, n0 + k)/rho;
		double e = 0.0;
		C u = gk(fr, X[k-1], X[k], tol, e);
		perr += e;
		if (std::abs(u) == 0.0)
		{
			val = s;
			out.converged = true;
			break;
		}
		s += u;
		Bb[k-1] = 1.0/u;
		Aa[k-1] = s*Bb[k-1];
		for (int j = 1; j <= k-1; ++j)
		{
			double dd = 1.0/X[k] - 1.0/X[k-j];
			Aa[k-1-j] = (Aa[k-j] - Aa[k-1-j])/dd;
			Bb[k-1-j] = (Bb[k-j] - Bb[k-1-j])/dd;
		}
		val = Aa[0]/Bb[0];
		out.partitions = k;
		if (k > 3 && std::abs(val - old) < 1e-13*std::abs(val))
		{
			out.converged = true;
			break;
		}
		if (k > 3 && std::abs(u) < 1e-17*std::abs(s))
		{
			val = s;
			out.converged = true;
			break;
		}
		old = val;
	}
	out.value = total + val;
	out.err = err + perr + std::abs(val - old);
	return out;
}

int curl_order(int comp) { return comp == 0 ? 2 : (comp == 1 ? 0 : 1); }
int mgf_order(int comp) { return (comp == 1 || comp == 2) ? 1 : 0; }

} // namespace


int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "usage: strata_probe JOB" << std::endl;
		return 2;
	}

	YAML::Node job = YAML::LoadFile(argv[1]);
	const double f = job["freq"].as<double>();

	LayerManager lm;
	build_stack(job["stack"], lm);
	lm.ProcessLayers(f);

	if (job["znodes"])
	{
		std::vector<double> zn;
		for (auto z : job["znodes"]) zn.push_back(z.as<double>());
		lm.InsertNodes_z(zn);
	}
	if (job["rhonodes"])
	{
		std::vector<double> rn;
		for (auto r : job["rhonodes"]) rn.push_back(r.as<double>());
		lm.InsertNodes_rho(rn);
	}

	MGF_settings s;
	read_settings(job["settings"], s);

	MGF mgf;
	bool do_init = !job["init"] || job["init"].as<int>();
	std::string init_err;
	auto t0 = std::chrono::steady_clock::now();
	if (do_init)
	{
		try { mgf.Initialize(f, lm, s); }
		catch (const std::exception &e) { init_err = e.what(); }
	}
	double init_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
	std::cout << "{\"q\":-1,\"init_ms\":" << num(init_ms) << ",\"nlayers\":" << lm.layers.size()
	          << ",\"mgf_nlayers\":" << mgf.lm.layers.size();
	if (!init_err.empty()) std::cout << ",\"err\":" << quote(init_err);
	std::cout << "}" << std::endl;

	int qi = 0;
	for (auto q : job["queries"])
	{
		std::ostringstream o;
		o << "{\"q\":" << qi;
		auto tq = std::chrono::steady_clock::now();
		try
		{
			const std::string kind = q[0].as<std::string>();
			LayerManager &L = do_init ? mgf.lm : lm;
			auto layer = [&](int idx, double zz) { return idx == -999 ? L.FindLayer(zz) : idx; };
			if (kind == "mgf" || kind == "qmgf")
			{
				double x = q[1].as<double>(), y = q[2].as<double>(), z = q[3].as<double>(), zp = q[4].as<double>();
				int i = layer(q[5].as<int>(), zp), m = layer(q[6].as<int>(), z);
				mgf.SetLayers(i, m);
				std::array<C, 9> G, Gc;
				C phi;
				if (kind == "mgf") mgf.ComputeMGF(x, y, z, zp, G, phi);
				else mgf.ComputeQMGF(x, y, z, zp, G, phi);
				o << ",\"i\":" << i << ",\"m\":" << m << ",\"G\":" << carr(G) << ",\"phi\":" << cnum(phi);
				if (mgf.singularity_factors_computed)
				{
					std::array<C, 10> F;
					for (int jj = 0; jj < 9; ++jj) F[jj] = mgf.GetSingularityFactor(jj);
					F[9] = mgf.GetSingularityFactor(-1);
					o << ",\"F\":" << carr(F);
				}
				if (s.compute_curl)
				{
					if (kind == "mgf") mgf.ComputeCurlMGF(x, y, z, zp, Gc);
					else mgf.ComputeCurlQMGF(x, y, z, zp, Gc);
					o << ",\"curl\":" << carr(Gc);
				}
			}
			else if (kind == "spec")
			{
				C kr(q[1].as<double>(), q[2].as<double>());
				double z = q[3].as<double>(), zp = q[4].as<double>();
				int i = layer(q[5].as<int>(), zp), m = layer(q[6].as<int>(), z);
				std::array<C, 5> K;
				std::array<C, 4> KE, KH;
				for (int pass = 0; pass < 3; ++pass)
				{
					SpectralMGF sm;
					sm.Initialize(&L, f, s.components, s.extract_quasistatic, s.extract_homogeneous);
					if (pass == 1) sm.SetCurlToGEM();
					if (pass == 2) sm.SetCurlToGHJ();
					sm.SetLayers(i, m);
					sm.SetSourcePoint(zp);
					sm.SetObservationPoint(z);
					sm.SetRadialWaveNumber(kr);
					if (pass == 0)
					{
						sm.ComputeSpectralMGF();
						for (int c = 0; c < 5; ++c) K[c] = sm.GetResult(c);
					}
					else
					{
						sm.ComputeCurlSpectralMGF();
						for (int c = 0; c < 4; ++c) (pass == 1 ? KE : KH)[c] = sm.GetResultCurl(c);
					}
				}
				o << ",\"i\":" << i << ",\"m\":" << m << ",\"K\":" << carr(K) << ",\"GEM\":" << carr(KE) << ",\"GHJ\":" << carr(KH);
			}
			else if (kind == "qspec" || kind == "qspat")
			{
				int off = (kind == "qspec") ? 1 : 0;
				C kr = (kind == "qspec") ? C(q[1].as<double>(), q[2].as<double>()) : C(0.0);
				double rho = (kind == "qspat") ? q[1].as<double>() : 0.0;
				double z = q[2 + off].as<double>(), zp = q[3 + off].as<double>();
				int i = layer(q[4 + off].as<int>(), zp), m = layer(q[5 + off].as<int>(), z);
				std::array<C, 5> K;
				std::array<C, 4> KE, KH;
				for (int pass = 0; pass < 3; ++pass)
				{
					QuasistaticMGF qm;
					qm.Initialize(&L, f, s.components, s.extract_singularities);
					if (pass == 1) qm.SetCurlToGEM();
					if (pass == 2) qm.SetCurlToGHJ();
					qm.SetLayers(i, m);
					if (pass == 0)
					{
						if (off) qm.ComputeQMGF_Spectral(z, zp, kr);
						else qm.ComputeQMGF_Spatial(z, zp, rho);
						for (int c = 0; c < 5; ++c) K[c] = off ? qm.GetResult_Spectral(c) : qm.GetResult_Spatial(c);
					}
					else
					{
						if (off) qm.ComputeCurlQMGF_Spectral(z, zp, kr);
						else qm.ComputeCurlQMGF_Spatial(z, zp, rho);
						for (int c = 0; c < 4; ++c)
							(pass == 1 ? KE : KH)[c] = off ? qm.GetResultCurl_Spectral(c) : qm.GetResultCurl_Spatial(c);
					}
				}
				o << ",\"i\":" << i << ",\"m\":" << m << ",\"K\":" << carr(K) << ",\"GEM\":" << carr(KE) << ",\"GHJ\":" << carr(KH);
			}
			else if (kind == "refint")
			{
				double rho = q[1].as<double>(), z = q[2].as<double>(), zp = q[3].as<double>();
				int i = layer(q[4].as<int>(), zp), m = layer(q[5].as<int>(), z);
				RefKernel K{&L, f, s, i, m, z, zp, q[6].as<int>(), q[7].as<int>()};
				int order = K.kind == 0 ? mgf_order(K.comp) : curl_order(K.comp);
				RefIntegral r = ref_integrate(K, rho, order);
				o << ",\"i\":" << i << ",\"m\":" << m << ",\"value\":" << cnum(r.value) << ",\"err_est\":" << num(r.err)
				  << ",\"partitions\":" << r.partitions << ",\"converged\":" << (r.converged ? 1 : 0);
			}
			else if (kind == "findlayer")
			{
				o << ",\"layer\":" << L.FindLayer(q[1].as<double>());
			}
			else
				throw std::invalid_argument("unknown query kind " + kind);
		}
		catch (const std::exception &e)
		{
			o << ",\"err\":" << quote(e.what());
		}
		double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tq).count();
		o << ",\"ms\":" << num(ms) << "}";
		std::cout << o.str() << std::endl;
		++qi;
	}
	return 0;
}
