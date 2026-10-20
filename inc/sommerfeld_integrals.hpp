// Author: Shashwat Sharma

// Copyright 2021 Shashwat Sharma and Piero Triverio

// This file is part of Strata.

// Strata is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// Strata is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with Strata.  If not, see <https://www.gnu.org/licenses/>.

/************************ sommerfeld_integrals.hpp ************************

 * Toolkit for computing Sommerfeld integrals in the complex krho plane.
 * Includes straightforward numerical integration, as well as the
 * Levin-Sidi approach with partition-extrapolation. This is a header-only
 * library.

 * References: 
 * Michalski, Mosig, "Efficient Computation of Sommerfeld Integral Tails -
 * Methods and Algorithms", Jnl. EM Waves & Appl., 2016.
 *
 * Author: Shashwat Sharma
 * Created on: Mar 14, 2020

 *************************************************************************/


#ifndef SI_H
#define SI_H


#include <algorithm>
#include <atomic>
#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp> 
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include "libAmosBessel.h"

#include "spectral_MGF.hpp"
#include "bessel_zeros.hpp"


// ==================================================================================
// Declarations
// ==================================================================================

// ====== Computational drivers ======

std::complex<double> SommerfeldIntegrand(SpectralMGF &smgf, double rho, std::complex<double> krho, int component, int order, bool curl = false);
std::complex<double> IntegrateSpectralFarField(SpectralMGF &smgf, double rho, int component, int order, double a, bool curl = false, double tol = 1.0e-6);
std::complex<double> IntegrateSpectralNearField(SpectralMGF &smgf, double rho, int component, int order, double a, bool curl = false, double tol = 1.0e-6);


// ====== Partition-extrapolation integration routines ======

template<class F>
void PartExtrap(const F f, double a, double q, double tol, std::complex<double> &val);

std::complex<double> LevinSidi(int k, std::complex<double> Sk, std::complex<double> wk, double *X, std::complex<double> *A, std::complex<double> *B);

template<class F>
void TanhSinh(const F f0, double a, double b, double eps, std::complex<double> &val);

template<class F>
void TruncIndex(const F f, double eh, int &n, std::complex<double> &s, double sigma, double eta, double a, double b);

template<class F>
void PartSum(const F f, double eh, double e2h, int &n, std::complex<double> &s, double sigma, double eta, double a, double b);

template<class F>
std::complex<double> Term(const F f, double ekh, double sigma, double eta, double a, double b);


// ====== Wrappers for computing the Bessel function ======

void BesselJ_Amos(double v, std::complex<double> x, std::complex<double> a, std::complex<double> &result);

double BesselJ_NextZero(double v, double x);


// ====== Numerical integration wrapper functions via Boost ======

template<class F>
void GaussKronrodBoost(const F f, double a, double b, double tol, std::complex<double> &result);

template<class F>
void TanhSinhBoost(const F f, double a, double b, double tol, std::complex<double> &result);

template<class F>
void ExpSinhBoost(const F f, double a, double tol, std::complex<double> &result);


// ====== Diagnostics ======

#define STRATA_SOMMERFELD_DIAGNOSTICS 1

/*! \brief Number of Sommerfeld tails (all threads) whose partition extrapolation ended without meeting its tolerance.
PartExtrap() then returns its last estimate; a warning is printed the first time this happens.*/
inline std::atomic<long> &SommerfeldTailNonConvergedCounter()
{
	static std::atomic<long> count(0);
	return count;
}

inline long SommerfeldTailNonConvergedCount() { return SommerfeldTailNonConvergedCounter().load(); }


// ==================================================================================
// Definitions
// ==================================================================================
// Computational drivers
// ==================================================================================
inline void CheckComplexResultForInfOrNaN(const std::complex<double> &value, const std::string &calculation_stage)
{
	if (std::isnan(value.real()) || std::isnan(value.imag()))
		throw std::runtime_error("NaN values detected in " + calculation_stage);
	if (std::isinf(std::abs(value)))
		throw std::runtime_error("Inf values detected in the magnitude of " + calculation_stage);
}


/*! \brief Function to assemble the integrand for a given component of the MGF.*/
inline std::complex<double> SommerfeldIntegrand(SpectralMGF &smgf, double rho, std::complex<double> krho, int component, int order, bool curl)
{

	// Compute spectral MGF or its curl product
	std::complex<double> K;
	smgf.SetRadialWaveNumber(krho);
	if (!curl)
	{		
		smgf.ComputeSpectralMGF();
		K = smgf.GetResult(component);
	}
	else
	{
		smgf.ComputeCurlSpectralMGF();
		K = smgf.GetResultCurl(component);
	}

	// Compute Bessel function
	std::complex<double> Jv;
	BesselJ_Amos(order, krho, rho, Jv);

	return Jv*K*std::pow(krho, order+1)/2.0/M_PI;

}


/*! \brief Function to integrate the spectral MGF for values of krho beyond the spectral near-field, using partition-extrapolation routines.*/
inline std::complex<double> IntegrateSpectralFarField(SpectralMGF &smgf, double rho, int component, int order, double a, bool curl, double tol)
{

	
	// According to Michalski, Mosig, 2016, the partition-extrapolation procedure should begin at the first zero of the Bessel function that occurs after a - call this point b. A separate finite-domain integration has to be performed to "bridge the gap" between a and b. Pre-tabulated values of Bessel zeros from Abramowitz & Stegun 1972 are used to pick b, available through the Boost library.

	double b = BesselJ_NextZero(order, a*rho)/rho;
	double q = M_PI/rho;

	// MGF integrand functor
	auto f = [&rho, &smgf, &component, &order, &curl](std::complex<double> krho) -> std::complex<double>
		{
			return SommerfeldIntegrand(smgf, rho, krho, component, order, curl);
		};

	// On the axis the integrand does not oscillate, and a semi-infinite rule is used. A single mapped rule on
	// [a, inf) only resolves features on the scale of a, however, while the integrand decays on the scale
	// 1/|z - zp|, which on chip can be 1e4 times larger (8 % error for rho = 0 in a 6 um stack at 1 GHz). Panels whose
	// widths start at a and double cover all scales; the semi-infinite rule is applied once they have become
	// negligible.
	if (rho < 1.0e-15)
	{
		std::complex<double> result = 0.0;
		double left = a, width = (a > 0.0) ? a : 1.0;
		for (int panel = 0, negligible = 0; panel < 200 && negligible < 3; panel++)
		{
			std::complex<double> part = 0.0;
			GaussKronrodBoost(f, left, left + width, tol, part);
			result += part;
			negligible = (std::abs(part) <= tol*std::abs(result)) ? negligible + 1 : 0;
			left += width;
			width *= 2.0;
		}
		std::complex<double> tail = 0.0;
		GaussKronrodBoost(f, left, std::numeric_limits<double>::infinity(), tol, tail);
		result += tail;
		CheckComplexResultForInfOrNaN(result, "the direct Gauss-Kronrod integration result");
		return result;
	};
	
	// Bridge the gap [a, b]. For rho << 1/a, b = j_{nu,1}/rho lies many orders of magnitude beyond a, while the
	// integrand varies (and, for z != zp, decays) on the scale of a. A single adaptive rule on [a, b] then only samples
	// the far part of the interval and can return zero. Panels whose widths start at a and double resolve every
	// scale between a and b; for ordinary rho this is one or two panels.
	std::complex<double> bridge = 0.0;
	double left = a, width = std::max(a, std::ldexp(b, -60));
	while (left < b)
	{
		double right = std::min(b, left + width);
		std::complex<double> part = 0.0;
		GaussKronrodBoost(f, left, right, tol, part);
		bridge += part;
		left = right;
		width *= 2.0;
	}

	std::complex<double> result = 0.0;
	PartExtrap(f, b, q, tol, result);

	
	// DEBUGGING

	// if (curl && component == 2)
	// std::cout << result << std::endl;

	// if (!std::isfinite(std::abs(bridge)))
	// 	std::cout << "bridge: " << bridge << "; rho: " << rho << "; component: " << component << std::endl;
	// if (!std::isfinite(std::abs(result)))
	// 	std::cout << "result: " << result << "; rho: " << rho << "; component: " << component << std::endl;

	// \todo Extremely small distances give overflow / underflow issues - needs to be addressed
	// if (!std::isfinite(std::abs(bridge)))
	// 	bridge = 0.0;
	// if (!std::isfinite(std::abs(result)))
		// result = 0.0;

	CheckComplexResultForInfOrNaN(result, "the partition-extrapolated tail result");
	CheckComplexResultForInfOrNaN(bridge, "the Gauss-Kronrod bridge integration result");
	return result + bridge;

}


/*! \brief Function to integrate the spectral MGF for values of krho in the spectral near-field, using a rectangular path deformation to avoid singularities.*/
inline std::complex<double> IntegrateSpectralNearField(SpectralMGF &smgf, double rho, int component, int order, double a, bool curl, double tol)
{

	// if (std::abs(smgf.lm->k_max) < 1.0e-15)
	// 	return 0.0;

	// Integrate along three paths in the complex plane: first, upwards along the imaginary axis, then right-wards until all poles are left behind, and then back down towards the real axis.

	const std::complex<double> J (0.0, 1.0);
	
	// Different problems seem to require different settings for h. A more robust way to pick it would be useful.
	
	// double h = 1.0e-2*std::abs(smgf.lm->k_max);
	double h = 1.0e-3*std::abs(smgf.lm->k_max);
	// double h = 1.0e-1*std::abs(smgf.lm->k_max);
	double krho_real, krho_imag;
	
	// MGF integrand functor for a fixed imaginary part in the complex plane
	auto f_real = [&rho, &smgf, &component, &order, &curl, &krho_imag, &J](double _krho_real) -> std::complex<double>
		{
			std::complex<double> krho = _krho_real + J*krho_imag;
			return SommerfeldIntegrand(smgf, rho, krho, component, order, curl);
		};

	// MGF integrand functor for a fixed real part in the complex plane
	auto f_imag = [&rho, &smgf, &component, &order, &curl, &krho_real, &J](double _krho_imag) -> std::complex<double>
		{
			std::complex<double> krho = krho_real + J*_krho_imag;
			return SommerfeldIntegrand(smgf, rho, krho, component, order, curl);
		};

	double zero_offset = 0.0e-3;
	
	// First path
	std::complex<double> first_path = 0.0;
	krho_real = zero_offset;
	GaussKronrodBoost(f_imag, 0.0, h, tol, first_path);

	// Second path
	std::complex<double> second_path = 0.0;
	krho_imag = h;
	GaussKronrodBoost(f_real, zero_offset, a, tol, second_path);
	// TanhSinhBoost(f_real, zero_offset, a, tol, second_path);
	
	// Third path
	std::complex<double> third_path = 0.0;
	krho_real = a;
	GaussKronrodBoost(f_imag, h, 0.0, tol, third_path);

	// The vertical paths are parametrized by the imaginary part t of krho, so dkrho = j dt there.
	return (J*first_path + second_path + J*third_path);

}


// ==================================================================================
// Partition-extrapolation integration routines
// ==================================================================================

/*! \brief Driver function for integrating the spectral far-field (SFF) tail integral of the MGF using techniques from Michalski, Mosig, 2016.*/
template<class F>
void PartExtrap(const F f, double a, double q, double tol, std::complex<double> &val)
{
	
	// Initialize. The partition budget used to be 11 partitions; slowly converging tails (e.g. same-height points)
	// were then returned unconverged without notice.
	constexpr int kmax = 30;
	double eps = tol;
	std::complex<double> old = 1.0e300, u, w, s = 0.0;
	std::complex<double> A [kmax + 1], B [kmax + 1];
	double X [kmax + 2];
	X[0] = a;
	bool converged = false;

	// The W-algorithm divides by differences of 1/x and by the remainder estimates w, and its result A[0]/B[0] does
	// not change when x or w is rescaled. It is therefore run in units of the partition width q and of the first
	// partition integral: with the raw values, rho in the nanometre range (q = pi/rho ~ 1e9) overflowed A and B after
	// about 25 partitions and returned NaN.
	double Xn [kmax + 2];
	Xn[0] = a/q;
	double w_scale = 1.0;

	// Begin extrapolated integration over partitions
	for (int kk = 1; kk <= kmax+1; kk++)
	{
		// Execute quadrature over the current partition
		X[kk] = X[kk-1] + q;
		Xn[kk] = Xn[kk-1] + 1.0;
		TanhSinh(f, X[kk-1], X[kk], eps, u);

		CheckComplexResultForInfOrNaN(u, "the TanhSinh partition integration result");
		// Stop when the tail underflows; skipping would leave a gap in A and B.
		if (std::abs(u) < std::numeric_limits<double>::min())
		{
			converged = true;
			break;
		}

		// Execute extrapolation		
		s += u;
		if (kk == 1)
			w_scale = std::abs(u);
		w = u/w_scale;
		val = LevinSidi(kk-1, s, w, Xn, A, B);
		CheckComplexResultForInfOrNaN(val, "the Levin-Sidi extrapolation result");
		if (std::abs(val - old) < tol*std::abs(val))
		{
			converged = true;
			break;
		}
		
		old = val;
	}

	if (!converged && SommerfeldTailNonConvergedCounter()++ == 0)
		std::cerr << "[WARNING] PartExtrap(): a Sommerfeld tail did not converge to the relative tolerance " << tol << " within "
		          << kmax + 1 << " partitions; its last extrapolated value is used. Later occurrences are only counted, see "
		          << "SommerfeldTailNonConvergedCount()." << std::endl;
	
	return;

}


/*! \brief Levin transformation function with the Sidi W-algorithm extrapolation.*/
inline std::complex<double> LevinSidi(int k, std::complex<double> Sk, std::complex<double> wk, double *X, std::complex<double> *A, std::complex<double> *B)
{
	
	B[k] = 1.0/wk; 
	A[k] = Sk*B[k];

	for (int j = 1; j <= k; j++)
	{
		double d = 1.0/X[k+1] - 1.0/X[k-j+1]; 
		A[k-j] = (A[k-j+1] - A[k-j])/d;
		B[k-j] = (B[k-j+1] - B[k-j])/d;
	}
	
	return A[0]/B[0];

}


/*! \brief Execute double-exponential tanh-sinh quadrature based on Michalski, Mosig, 2016.*/
template<class F>
void TanhSinh(const F f0, double a, double b, double eps, std::complex<double> &val)
{

	// The algorithm requires the functor to accept two arguments to handle end-point singularities. This should not be an issue for us, so we can side-step this requirement by wrapping the actual functor f0 in a dummy functor.
	auto f = [&f0](double c, double d) -> std::complex<double>
		{
			return f0(c+d);
		};

	double eta = 1.0;
	int maxlev = 5;

	double sigma = (b - a)/2.0;
	double gamma = (b + a)/2.0;
	std::complex<double> s = eta*f(gamma, 0.0);
	
	double h = 0.5;
	double eh = std::exp(h);

	int n;
	TruncIndex(f, eh, n, s, sigma, eta, a, b);
	std::complex<double> old = sigma*h*s;

	for (int m = 1; m <= maxlev; m++)
	{
		double e2h = eh;
		h /= 2.0;
		eh = std::exp(h);
		
		PartSum(f, eh, e2h, n, s, sigma, eta, a, b);
		val = old/2.0 + sigma*h*s;
		
		if (std::abs(val - old) <= eps*std::abs(val))
			break;
		
		old = val;
		n *= 2;
	}

	return;

}


/*! \brief Helper function for tanh-sinh quadrature.*/
template<class F>
void TruncIndex(const F f, double eh, int &n, std::complex<double> &s, double sigma, double eta, double a, double b)
{

	int nmax = 24;
	double kappa = 1.0e-15;
	double ekh = eh;

	// Sum the terms k = 1, 2, ... of the coarsest level until they become negligible. On return, n is the number
	// of terms used: PartSum() needs it to cover the same abscissa range (odd multiples of the halved step) on the
	// finer levels.
	for (n = 1; n <= nmax; n++)
	{
		std::complex<double> t = Term(f, ekh, sigma, eta, a, b);
		s += t;

		if (std::abs(t) <= kappa*std::abs(s))
			break;

		ekh *= eh;
	}

	n = std::min(n, nmax);

	return;
	
}


/*! \brief Helper function for tanh-sinh quadrature.*/
template<class F>
void PartSum(const F f, double eh, double e2h, int &n, std::complex<double> &s, double sigma, double eta, double a, double b)
{
	
	double ekh = eh;
	s = Term(f, ekh, sigma, eta, a, b);

	for (int k = 2; k <= n; k++)
	{
		ekh *= e2h;
		s += Term(f, ekh, sigma, eta, a, b);
	}

	return;
	
}


/*! \brief Helper function for tanh-sinh quadrature.*/
template<class F>
std::complex<double> Term(const F f, double ekh, double sigma, double eta, double a, double b)
{
	
	double q = std::exp(-eta*(ekh - 1.0/ekh));
	double delta = 2.0*q/(1.0 + q);
	double w = eta*(ekh + 1.0/ekh)*delta/(1.0 + q);
	std::complex<double> t = w*(f(a, sigma*delta) + f(b, -sigma*delta));		

	return t;
	
}


// ==================================================================================
// Wrappers for computing the Bessel function
// ==================================================================================

/*! \brief Compute the Bessel function of first kind, J_v(a*x), using the external library AmosBessel.*/
inline void BesselJ_Amos(double v, std::complex<double> x, std::complex<double> a, std::complex<double> &result)
{

	x *= a;
	__complex__ double x_in;
	__real__ x_in = x.real();
	__imag__ x_in = x.imag();
	
	int nOrders = 1;
	int scale = 0;
	
	int info = AmosBessel('J', x_in, v, nOrders, scale, (__complex__ double *) &result);

	if (info != 0)
		std::cout << "[WARNING] BesselJ_Amos(): AmosBessel() returned " << info << std::endl;
	
	return;

}


/*! \brief Compute the next zero of the Bessel function of first kind, J_v, after a given point x, using the external library Boost.*/
inline double BesselJ_NextZero(double v, double x)
{

	// Use tabulated values, if possible
	if ((int)v == 0)
		return BesselJ0_NextZero(x);
	else if ((int)v == 1)
		return BesselJ1_NextZero(x);

	return BesselJ_NextZero_Boost(v, x);

}


// ==================================================================================
// Numerical integration wrapper functions via Boost
// ==================================================================================

/*! \brief Wrapper function that uses Boost's Gauss-Kronrod quadrature routine to integrate a given complex-valued function along the real line. The user-provided functor should accept a single double and return a complex double.*/
template<class F>
void GaussKronrodBoost(const F f, double a, double b, double tol, std::complex<double> &result)
{
	
	const int order = 21, max_levels = 15;
	double error, multiplier = 1.0;

	// Make sure the integration path is from the smaller number to the larger one, for Boost compatibility.
	if (a > b)
	{
		double c = b;
		b = a;
		a = c;
		multiplier = -1.0;
	}

	typedef typename std::complex<double>::value_type value_type;
	boost::math::quadrature::gauss_kronrod<value_type, order> integrator;
	
	result = integrator.integrate(f, a, b, max_levels, tol, &error);

    //result = boost::math::quadrature::trapezoidal(f, a, b, 1.0e-6);
	result *= multiplier;

	return;
	
}


/*! \brief Wrapper function that uses Boost's double-exponential tanh-sinh quadrature routine to integrate a given complex-valued function along the real line. The user-provided functor should accept a single double and return a complex double.*/
template<class F>
void TanhSinhBoost(const F f, double a, double b, double tol, std::complex<double> &result)
{
	
	double multiplier = 1.0;

	// Make sure the integration path is from the smaller number to the larger one, for Boost compatibility.
	if (a > b)
	{
		double c = b;
		b = a;
		a = c;
		multiplier = -1.0;
	}

	typedef typename std::complex<double>::value_type value_type;
	boost::math::quadrature::tanh_sinh<value_type> integrator;

	result = integrator.integrate(f, a, b, tol);
	result *= multiplier;

	return;
	
}


/*! \brief Wrapper function that uses Boost's double-exponential exp-sinh quadrature routine to integrate a given complex-valued function along the real line, from a given start point to positive infinity. The user-provided functor should accept a single double and return a complex double.*/
template<class F>
void ExpSinhBoost(const F f, double a, double tol, std::complex<double> &result)
{
	
	typedef typename std::complex<double>::value_type value_type;
	boost::math::quadrature::exp_sinh<value_type> integrator;

	result = integrator.integrate(f, a, std::numeric_limits<double>::infinity(), tol);

	return;
	
}


#endif





