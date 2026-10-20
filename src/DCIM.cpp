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

/************************** DCIM.cpp ***************************

 * Functionality to compute the spatial domain MGF using the 
 * one-, two-, or three-level DCIM with GPOF.
 * 
 * References:
 *
 * Two-level DCIM:
 * Aksun, "A Robust Approach for the Derivation of Closed-Form
 * Green's Functions", IEEE-MTT, 1996
 * Aksun, Dural, "Clarification of Issues on the Closed-Form
 * Green's Functions in Stratified Media", IEEE-TAP, 2005
 *
 * Three-level DCIM:
 * Alparslan, Aksun, Michalski, "Closed-Form Green’s Functions
 * in Planar Layered Media for All Ranges and Materials",
 * IEEE-MTT, 2010
 *
 * Modifications to the DCIM path:
 * Shuley, Boix, Medina, Horno, "On the Fast Approximation of
 * Green’s Functions in MPIE Formulations for Planar Layered
 * Media", IEEE-MTT, 2002
 * Kipp, Chan, "Complex Image Method for Sources in Bounded
 * Regions of Multilayer Structures", IEEE-MTT, 1994
 *
 * Author: Shashwat Sharma
 * Date Created: Mar 31, 2020

 ***************************************************************/


#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#if defined(USE_MKL)
	#include <mkl.h>
#else
	#include <cblas.h>
#endif

#if defined(_OPENMP)
	#include <omp.h>
#endif

#include "DCIM.hpp"
#include "constants.hpp"
#include "utility.hpp"
#include "lapacke_interface.hpp"
#include "layers.hpp"
#include "spectral_MGF.hpp"

using namespace strata;


// ==================================================================================
// Interface
// ==================================================================================

/*! \brief Driver function to generate DCIM images for a given stackup, with associated z-nodes, and given settings.*/
void DCIM::GenerateImages(LayerManager *_lm, double _f, DCIM_settings _s)
{

	lm = _lm;
	f = _f;
	omega = 2.0*M_PI*f;
	s = _s;

	if (lm->z_nodes.size() < 1)
	{
		std::cout << "[WARNING] DCIM::GenerateImages(): No z-nodes have been defined, so no images were generated." << std::endl;
		return;
	}
	
	// The layer data are frequency dependent; a LayerManager processed at another frequency would silently give the
	// kernels of that frequency (it is shared through a pointer, so it is not re-processed behind its other users).
	if (!lm->layers_processed)
		lm->ProcessLayers(f);
	else if (lm->processed_frequency != f)
		throw std::logic_error("[ERROR] DCIM::GenerateImages(): The LayerManager was processed at " + std::to_string(lm->processed_frequency) + " Hz, not at the requested " + std::to_string(f) + " Hz. Call ProcessLayers() for this frequency, or use one LayerManager per frequency.");


	// ====== Reset all data structures ======

    im.clear();
	imc.clear();
	z_to_idx.clear();
	idxpair_to_image.clear();
	

	// ====== Generate maps between z-nodes and their index in the stackup ======

	// The int pair <ii, zz> is mapped to a unique int using Szudzik's function.
	// Reference: https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
	for (int ii = 0; ii < lm->z_nodes.size(); ii++)
	{
		for (int zz = 0; zz < lm->z_nodes[ii].size(); zz++)
		{
			int idx = ii >= zz ? ii*ii + ii + zz : ii + zz*zz;
			z_to_idx.insert(std::make_pair(lm->z_nodes[ii][zz], idx));
		}
	}

	int map_size = z_to_idx.size();
	

	// ====== Generate images ======

	int idx_image = 0;
	
	// Traverse source layers
	for (int ii = 0; ii < lm->layers.size(); ii++)
	{
		// Traverse observer layers
		for (int mm = 0; mm < lm->layers.size(); mm++)
		{
			// Traverse source z-nodes
			for (int ss = 0; ss < lm->z_nodes[ii].size(); ss++)
			{
				// Traverse observer z-nodes
				for (int tt = 0; tt < lm->z_nodes[mm].size(); tt++)
				{

					double zp = lm->z_nodes[ii][ss];
					double z = lm->z_nodes[mm][tt];

					// Generate images
					im.push_back(std::vector<DCIM_images> ());

					if (s.method == DCIM_TWO_LEVEL)
						GenerateImages_TwoLevel(z, zp, im.back(), false);
					else if (s.method == DCIM_ONE_LEVEL)
						GenerateImages_OneLevel(z, zp, im.back(), false);
					else if (s.method == DCIM_THREE_LEVEL)
						GenerateImages_ThreeLevel(z, zp, im.back(), false);

					// Generate images for the curl
					if (s.compute_curl)
					{
						imc.push_back(std::vector<DCIM_images> ());

						if (s.method == DCIM_TWO_LEVEL)
							GenerateImages_TwoLevel(z, zp, imc.back(), true);
						else if (s.method == DCIM_ONE_LEVEL)
							GenerateImages_OneLevel(z, zp, imc.back(), true);
						else if (s.method == DCIM_THREE_LEVEL)
							GenerateImages_ThreeLevel(z, zp, imc.back(), true);
					}
					
					// Update the map from the z-index pair to this set of images
					int idx_z = z_to_idx[z];
					int idx_zp = z_to_idx[zp];
					idxpair_to_image.insert(std::make_pair(std::make_pair(idx_z, idx_zp), idx_image));

					idx_image++;

				}
			}
		}
	}

	int map_size_new = z_to_idx.size();

	if (map_size_new != map_size)
		std::cout << "[WARNING] DCIM::GenerateImages(): DCIM indexing may have been corrupted due to numerical issues." << std::endl;
	
	images_generated = true;

	
	return;
	
}


/*! \brief Function to retrieve the nearest DCIM images for a given z-zp pair.*/
std::vector<DCIM_images>& DCIM::GetImages(double z, double zp, bool curl_mode)
{

	if (!images_generated)
	{
		throw std::logic_error("[ERROR] DCIM:GetImages(): Images have not been generated. Call DCIM::GenerateImages() first.");
	}

	if (curl_mode && imc.size() != im.size())
	{
		throw std::logic_error("[ERROR] DCIM:GetImages(): Images for the curl have not been generated. Set compute_curl in the DCIM settings.");
	}


	// ====== Locate the z-nodes nearest to z and zp ======

	// The nearest node is searched in the whole (sorted) node map; points beyond the first or last node map to that
	// node. (The former search dereferenced end() or decremented begin() for such points.)
	auto nearest = [](const std::map<double, int> &nodes, double x) -> int
		{
			auto above = nodes.lower_bound(x);
			if (above == nodes.end())
				return std::prev(above)->second;
			if (above == nodes.begin() || above->first == x)
				return above->second;
			auto below = std::prev(above);
			return (x - below->first < above->first - x) ? below->second : above->second;
		};

	int idx_z = nearest(z_to_idx, z);
	int idx_zp = nearest(z_to_idx, zp);


	// ====== Retrive the images at the above index pair ======

	auto found = idxpair_to_image.find(std::make_pair(idx_z, idx_zp));
	if (found == idxpair_to_image.end())
		throw std::logic_error("[ERROR] DCIM:GetImages(): No images were generated for the nearest z-node pair.");
	int idx_image = found->second;


	// ====== Debugging ======

	// std::vector<double> all_z_nodes;
	// for (int ii = 0; ii < lm->z_nodes.size(); ii++)
	// 	all_z_nodes.insert(all_z_nodes.end(), lm->z_nodes[ii].begin(), lm->z_nodes[ii].end());

	// double z_dist = 1.0e300, zp_dist = 1.0e300;
	// double z_debug, zp_debug;
	// for (int ii = 0; ii < all_z_nodes.size(); ii++)
	// {
	// 	if (std::abs(all_z_nodes[ii] - z) < z_dist)
	// 	{
	// 		z_dist = std::abs(all_z_nodes[ii] - z);
	// 		z_debug = all_z_nodes[ii];
	// 	}
	// 	if (std::abs(all_z_nodes[ii] - zp) < zp_dist)
	// 	{
	// 		zp_dist = std::abs(all_z_nodes[ii] - zp);
	// 		zp_debug = all_z_nodes[ii];
	// 	}
	// }			

	// if (std::abs(z_found - z_debug) > 0.0 || std::abs(zp_found - zp_debug) > 0.0)
	// 	std::cout << "z: (" << z << ", " << z_found << ", " << z_debug << "); zp: (" << zp << ", " << zp_found << ", " << zp_debug << ")" << std::endl;


	if (!curl_mode)
		return im[idx_image];
	else
		return imc[idx_image];
	
}


// ==================================================================================
// Two-level DCIM
// ==================================================================================

/*! \brief Get lateral and axial wave number (krho and kz) sample points for the two-level DCIM.*/
void DCIM::GenerateSamplePoints_TwoLevel(DCIM_sample_points &sp, int N1, int N2)
{
	
	sp.k = lm->k_min;

	// Find maximum permittivity among all layers to infer T02
	double epsr_max = 0.0;
	for (int ii = 0; ii < lm->layers.size(); ii++)
		if (std::abs(lm->eps[ii]) > epsr_max)
			epsr_max = std::abs(lm->eps[ii]);
	epsr_max /= eps0;

	// Set the sampling limits
	sp.T02 = 5.0*std::sqrt(epsr_max);
	sp.T01 = 200.0;
	
	// ------ Generate samples along the first path ------

	sp.kz_path1.resize(N1);
	sp.krho_path1.resize(N1);
	
	linspace(0.0, sp.T01, N1, sp.t1);

	for (int ii = 0; ii < N1; ii++)
	{
		sp.kz_path1[ii] = -J*sp.k*(sp.T02 + sp.t1[ii]);
		sp.krho_path1[ii] = std::sqrt(std::pow(sp.k, 2) - std::pow(sp.kz_path1[ii], 2));

		// Ensure that the samples in krho space lie in the top-right complex quadrant
		sp.krho_path1[ii].real(std::abs(std::real(sp.krho_path1[ii])));
		sp.krho_path1[ii].imag(std::abs(std::imag(sp.krho_path1[ii])));
	}

	// ------ Generate samples along the second path ------

	sp.kz_path2.resize(N2);
	sp.krho_path2.resize(N2);

	linspace(1.0e-2, sp.T02, N2, sp.t2);

	for (int ii = 0; ii < N2; ii++)
	{
		sp.kz_path2[ii] = sp.k*(-J*sp.t2[ii] + (1.0 - (sp.t2[ii]/sp.T02)));
		sp.krho_path2[ii] = std::sqrt(std::pow(sp.k, 2) - std::pow(sp.kz_path2[ii], 2));

		// Ensure that the samples in krho space lie in the top-right complex quadrant
		sp.krho_path2[ii].real(std::abs(std::real(sp.krho_path2[ii])));
		sp.krho_path2[ii].imag(std::abs(std::imag(sp.krho_path2[ii])));
	}

	return;

}


/*! \brief Generate images for the two-level DCIM.*/
void DCIM::GenerateImages_TwoLevel(double z, double zp, std::vector<DCIM_images> &im, bool curl_mode)
{

	int N_components;
	if (!curl_mode)
		N_components = s.components.size();
	else
		N_components = s.components_curl.size();
	
	im.resize(N_components);

	// Generate krho points on which to sample the spectral MGF
	int N1 = 100, N2 = 100;
	DCIM_sample_points sp;
	GenerateSamplePoints_TwoLevel(sp, N1, N2);

	int i = lm->FindLayer(zp);
	int m = lm->FindLayer(z);

	// Initialize and set up the spectral MGF object
	SpectralMGF smgf;
	smgf.Initialize(lm, f, s.components, s.extract_quasistatic, s.extract_homogeneous);
	smgf.SetLayers(i, m);

	smgf.SetSourcePoint(zp);
	smgf.SetObservationPoint(z);

	// ------ Generate spectral samples along path 1 ------

	std::vector<std::vector<std::complex<double>>> K_path1;
	GenerateSpectralSamples(smgf, K_path1, sp.krho_path1, sp.kz_path1, N1, N_components, curl_mode);

	// ------ Generate spectral samples along path 2 ------

	std::vector<std::vector<std::complex<double>>> K_path2;
	GenerateSpectralSamples(smgf, K_path2, sp.krho_path2, sp.kz_path2, N2, N_components, curl_mode);

	// Components whose quasistatic terms are singular at krho = 0 are sampled without extraction (see FullKernel())
	if (s.extract_quasistatic)
	{
		SpectralMGF smgf_full;
		smgf_full.Initialize(lm, f, s.components, false, false);
		smgf_full.SetLayers(i, m);
		smgf_full.SetSourcePoint(zp);
		smgf_full.SetObservationPoint(z);
		ReplaceWithFullKernel(smgf_full, K_path1, sp.krho_path1, sp.kz_path1, N1, N_components, curl_mode);
		ReplaceWithFullKernel(smgf_full, K_path2, sp.krho_path2, sp.kz_path2, N2, N_components, curl_mode);
	}

	// ------ Generate images for each component ------

	for (int ii = 0; ii < N_components; ii++)
	{

		// Check the mask that belongs to this mode only: the curl mask has four entries, the potential mask five.
		if (curl_mode ? !s.components_curl[ii] : !s.components[ii])
			continue;

		// Run GPOF on level 1 samples
		std::vector<std::complex<double>> a, alpha;
		RunGPOF(K_path1[ii], sp.t1[2] - sp.t1[1], a, alpha, s.tol_svd, s.tol_eig, s.max_num_images);

		// Modify coefficients to go from t-space to kz-space
		for (int jj = 0; jj < a.size(); jj++)
		{
			a[jj] = a[jj]*std::exp(-sp.T02*alpha[jj]);
			alpha[jj] = alpha[jj]/(J*sp.k);
		}

		// Store level 1 images
		im[ii].a.insert(std::end(im[ii].a), std::begin(a), std::end(a));
		im[ii].alpha.insert(std::end(im[ii].alpha), std::begin(alpha), std::end(alpha));

		// Use level 1 DCIM coefficients to compute the MGF along the level 2 path.
		// Then, this approximation from the actual level 2 samples.
		for (int mm = 0; mm < N2; mm++)
		{
			std::complex<double> kz = sp.kz_path2[mm];
			for (int jj = 0; jj < a.size(); jj++)
				K_path2[ii][mm] -= a[jj]*std::exp(-alpha[jj]*kz);
		}

		// Run GPOF on the modified level 2 samples
		a.clear();
		alpha.clear();
		RunGPOF(K_path2[ii], sp.t2[2] - sp.t2[1], a, alpha, s.tol_svd, s.tol_eig, s.max_num_images);

		// Modify coefficients to go from t-space to kz-space
		for (int jj = 0; jj < a.size(); jj++)
		{
			alpha[jj] = alpha[jj]*sp.T02/((1.0 + J*sp.T02)*sp.k);
			a[jj] = a[jj]*std::exp(sp.k*alpha[jj]);
		}

		// Store level 2 images
		im[ii].a.insert(std::end(im[ii].a), std::begin(a), std::end(a));
		im[ii].alpha.insert(std::end(im[ii].alpha), std::begin(alpha), std::end(alpha));

		// Store the wave number to be used when applying the Sommerfeld identity
		im[ii].k = sp.k;

	}

	return;
						
}


// ==================================================================================
// One-level DCIM
// ==================================================================================

/*! \brief Get lateral and axial wave number (krho and kz) sample points for the one-level DCIM.*/
void DCIM::GenerateSamplePoints_OneLevel(DCIM_sample_points &sp, int N1)
{
	
	sp.k = lm->k_min;

	// Find maximum permittivity among all layers to infer T02
	double epsr_max = 0.0;
	for (int ii = 0; ii < lm->layers.size(); ii++)
		if (std::abs(lm->eps[ii]) > epsr_max)
			epsr_max = std::abs(lm->eps[ii]);
	epsr_max /= eps0;

	// Set the sampling limits
	sp.T02 = 2.0*std::sqrt(epsr_max);
	sp.T01 = 1000.0;

	// ------ Generate samples along the first path ------

	sp.kz_path1.resize(N1);
	sp.krho_path1.resize(N1);
	
	linspace(0.0, sp.T01, N1, sp.t1);

	for (int ii = 0; ii < N1; ii++)
	{
		sp.kz_path1[ii] = -J*sp.k*(sp.T02 + sp.t1[ii]);
		sp.krho_path1[ii] = std::sqrt(std::pow(sp.k, 2) - std::pow(sp.kz_path1[ii], 2));

		// Ensure that the samples in krho space lie in the top-right complex quadrant
		sp.krho_path1[ii].real(std::abs(std::real(sp.krho_path1[ii])));
		sp.krho_path1[ii].imag(std::abs(std::imag(sp.krho_path1[ii])));
	}

	return;

}


/*! \brief Generate images for the one-level DCIM.*/
void DCIM::GenerateImages_OneLevel(double z, double zp, std::vector<DCIM_images> &im, bool curl_mode)
{

	int N_components;
	if (!curl_mode)
		N_components = s.components.size();
	else
		N_components = s.components_curl.size();
	
	im.resize(N_components);

	// Generate krho points on which to sample the spectral MGF
	int N1 = 100;
	DCIM_sample_points sp;
	GenerateSamplePoints_OneLevel(sp, N1);

	int i = lm->FindLayer(zp);
	int m = lm->FindLayer(z);

	// Initialize and set up the spectral MGF object
	SpectralMGF smgf;
	smgf.Initialize(lm, f, s.components, s.extract_quasistatic, s.extract_homogeneous);
	smgf.SetLayers(i, m);

	smgf.SetSourcePoint(zp);
	smgf.SetObservationPoint(z);

	// ------ Generate spectral samples along path 1 ------

	std::vector<std::vector<std::complex<double>>> K_path1;
	GenerateSpectralSamples(smgf, K_path1, sp.krho_path1, sp.kz_path1, N1, N_components, curl_mode);

	// Components whose quasistatic terms are singular at krho = 0 are sampled without extraction (see FullKernel())
	if (s.extract_quasistatic)
	{
		SpectralMGF smgf_full;
		smgf_full.Initialize(lm, f, s.components, false, false);
		smgf_full.SetLayers(i, m);
		smgf_full.SetSourcePoint(zp);
		smgf_full.SetObservationPoint(z);
		ReplaceWithFullKernel(smgf_full, K_path1, sp.krho_path1, sp.kz_path1, N1, N_components, curl_mode);
	}

	// ------ Generate images for each component ------

	for (int ii = 0; ii < N_components; ii++)
	{

		// Check the mask that belongs to this mode only: the curl mask has four entries, the potential mask five.
		if (curl_mode ? !s.components_curl[ii] : !s.components[ii])
			continue;

		// Run GPOF on level 1 samples
		std::vector<std::complex<double>> a, alpha;
		RunGPOF(K_path1[ii], sp.t1[2] - sp.t1[1], a, alpha, s.tol_svd, s.tol_eig, s.max_num_images);

		// Modify coefficients to go from t-space to kz-space
		for (int jj = 0; jj < a.size(); jj++)
		{
			a[jj] = a[jj]*std::exp(-sp.T02*alpha[jj]);
			alpha[jj] = alpha[jj]/(J*sp.k);
		}

		// Store level 1 images
		im[ii].a.insert(std::end(im[ii].a), std::begin(a), std::end(a));
		im[ii].alpha.insert(std::end(im[ii].alpha), std::begin(alpha), std::end(alpha));

		// Store the wave number to be used when applying the Sommerfeld identity
		im[ii].k = sp.k;

	}

	return;
						
}


// ==================================================================================
// Three-level DCIM
// ==================================================================================

/*! \brief Get lateral and axial wave number (krho and kz) sample points for the three-level DCIM.*/
void DCIM::GenerateSamplePoints_ThreeLevel(DCIM_sample_points &sp, int N1, int N2, int N3)
{
	
	sp.k = lm->k_min;

	// Find maximum permittivity among all layers to infer T02
	double epsr_max = 0.0;
	for (int ii = 0; ii < lm->layers.size(); ii++)
		if (std::abs(lm->eps[ii]) > epsr_max)
			epsr_max = std::abs(lm->eps[ii]);
	epsr_max /= eps0;

	// Set the sampling limits
	sp.T03 = 2.0*std::sqrt(epsr_max);
	sp.T02 = 200.0;
	sp.T01 = 2000.0;

	// ------ Generate samples along the first path ------

	sp.kz_path1.resize(N1);
	sp.krho_path1.resize(N1);
	
	linspace(0.0, sp.T01, N1, sp.t1);

	for (int ii = 0; ii < N1; ii++)
	{
		sp.kz_path1[ii] = -J*sp.k*(sp.T03 + sp.T02 + sp.t1[ii]);
		sp.krho_path1[ii] = std::sqrt(std::pow(sp.k, 2) - std::pow(sp.kz_path1[ii], 2));

		// Ensure that the samples in krho space lie in the top-right complex quadrant
		sp.krho_path1[ii].real(std::abs(std::real(sp.krho_path1[ii])));
		sp.krho_path1[ii].imag(std::abs(std::imag(sp.krho_path1[ii])));
	}

	// ------ Generate samples along the second path ------

	sp.kz_path2.resize(N2);
	sp.krho_path2.resize(N2);

	linspace(0.0, sp.T02, N2, sp.t2);

	for (int ii = 0; ii < N2; ii++)
	{
		sp.kz_path2[ii] = -J*sp.k*(sp.T03 + sp.t2[ii]);
		sp.krho_path2[ii] = std::sqrt(std::pow(sp.k, 2) - std::pow(sp.kz_path2[ii], 2));

		// Ensure that the samples in krho space lie in the top-right complex quadrant
		sp.krho_path2[ii].real(std::abs(std::real(sp.krho_path2[ii])));
		sp.krho_path2[ii].imag(std::abs(std::imag(sp.krho_path2[ii])));
	}

	// ------ Generate samples along the third path ------

	sp.kz_path3.resize(N3);
	sp.krho_path3.resize(N3);

	linspace(1.0e-8, sp.T03, N3, sp.t3);

	for (int ii = 0; ii < N3; ii++)
	{
		sp.kz_path3[ii] = sp.k*(-J*sp.t3[ii] + (1.0 - (sp.t3[ii]/sp.T03)));
		sp.krho_path3[ii] = std::sqrt(std::pow(sp.k, 2) - std::pow(sp.kz_path3[ii], 2));

		// Ensure that the samples in krho space lie in the top-right complex quadrant
		sp.krho_path3[ii].real(std::abs(std::real(sp.krho_path3[ii])));
		sp.krho_path3[ii].imag(std::abs(std::imag(sp.krho_path3[ii])));
	}

	return;

}


/*! \brief Generate images for the three-level DCIM.*/
void DCIM::GenerateImages_ThreeLevel(double z, double zp, std::vector<DCIM_images> &im, bool curl_mode)
{

	int N_components;
	if (!curl_mode)
		N_components = s.components.size();
	else
		N_components = s.components_curl.size();

	im.resize(N_components);

	// Generate krho points on which to sample the spectral MGF
	int N1 = 100, N2 = 100, N3 = 100;
	DCIM_sample_points sp;
	GenerateSamplePoints_ThreeLevel(sp, N1, N2, N3);

	int i = lm->FindLayer(zp);
	int m = lm->FindLayer(z);

	// Initialize and set up the spectral MGF object
	SpectralMGF smgf;
	smgf.Initialize(lm, f, s.components, s.extract_quasistatic, s.extract_homogeneous);
	smgf.SetLayers(i, m);

	smgf.SetSourcePoint(zp);
	smgf.SetObservationPoint(z);

	// ------ Generate spectral samples along path 1 ------

	std::vector<std::vector<std::complex<double>>> K_path1;
	GenerateSpectralSamples(smgf, K_path1, sp.krho_path1, sp.kz_path1, N1, N_components, curl_mode);

	// ------ Generate spectral samples along path 2 ------

	std::vector<std::vector<std::complex<double>>> K_path2;
	GenerateSpectralSamples(smgf, K_path2, sp.krho_path2, sp.kz_path2, N2, N_components, curl_mode);

	// ------ Generate spectral samples along path 3 ------

	std::vector<std::vector<std::complex<double>>> K_path3;
	GenerateSpectralSamples(smgf, K_path3, sp.krho_path3, sp.kz_path3, N3, N_components, curl_mode);

	// Components whose quasistatic terms are singular at krho = 0 are sampled without extraction (see FullKernel())
	if (s.extract_quasistatic)
	{
		SpectralMGF smgf_full;
		smgf_full.Initialize(lm, f, s.components, false, false);
		smgf_full.SetLayers(i, m);
		smgf_full.SetSourcePoint(zp);
		smgf_full.SetObservationPoint(z);
		ReplaceWithFullKernel(smgf_full, K_path1, sp.krho_path1, sp.kz_path1, N1, N_components, curl_mode);
		ReplaceWithFullKernel(smgf_full, K_path2, sp.krho_path2, sp.kz_path2, N2, N_components, curl_mode);
		ReplaceWithFullKernel(smgf_full, K_path3, sp.krho_path3, sp.kz_path3, N3, N_components, curl_mode);
	}

	// ------ Generate images for each component ------

	for (int ii = 0; ii < N_components; ii++)
	{

		// Check the mask that belongs to this mode only: the curl mask has four entries, the potential mask five.
		if (curl_mode ? !s.components_curl[ii] : !s.components[ii])
			continue;

		// Run GPOF on level 1 samples
		std::vector<std::complex<double>> a, alpha;
		RunGPOF(K_path1[ii], sp.t1[2] - sp.t1[1], a, alpha, s.tol_svd, s.tol_eig, s.max_num_images);

		// Modify coefficients to go from t-space to kz-space
		for (int jj = 0; jj < a.size(); jj++)
		{
			a[jj] = a[jj]*std::exp(-(sp.T03 + sp.T02)*alpha[jj]);
			alpha[jj] = alpha[jj]/(J*sp.k);
		}

		// Store level 1 images
		im[ii].a.insert(std::end(im[ii].a), std::begin(a), std::end(a));
		im[ii].alpha.insert(std::end(im[ii].alpha), std::begin(alpha), std::end(alpha));

		// Use level 1 DCIM coefficients to compute the MGF along the level 2 path.
		// Then, this approximation from the actual level 2 samples.
		for (int mm = 0; mm < N2; mm++)
		{
			std::complex<double> kz = sp.kz_path2[mm];
			for (int jj = 0; jj < a.size(); jj++)
				K_path2[ii][mm] -= a[jj]*std::exp(-alpha[jj]*kz);
		}

		// Use level 1 DCIM coefficients to compute the MGF along the level 3 path.
		// Then, subtract this approximation from the actual level 3 samples.
		for (int mm = 0; mm < N3; mm++)
		{
			std::complex<double> kz = sp.kz_path3[mm];
			for (int jj = 0; jj < a.size(); jj++)
				K_path3[ii][mm] -= a[jj]*std::exp(-alpha[jj]*kz);
		}

		// Run GPOF on the modified level 2 samples
		a.clear();
		alpha.clear();
		RunGPOF(K_path2[ii], sp.t2[2] - sp.t2[1], a, alpha, s.tol_svd, s.tol_eig, s.max_num_images);

		// Modify coefficients to go from t-space to kz-space
		for (int jj = 0; jj < a.size(); jj++)
		{
			a[jj] = a[jj]*std::exp(-sp.T03*alpha[jj]);
			alpha[jj] = alpha[jj]/(J*sp.k);
		}

		// Store level 2 images
		im[ii].a.insert(std::end(im[ii].a), std::begin(a), std::end(a));
		im[ii].alpha.insert(std::end(im[ii].alpha), std::begin(alpha), std::end(alpha));
	
		// Use level 2 DCIM coefficients to compute the function on the level 3 path.
		// Then, subtract this approximation from the actual level 3 samples.
		for (int mm = 0; mm < N3; mm++)
		{
			// Generate level 2 samples from the level 1 approximation and subtract
			std::complex<double> kz = sp.kz_path3[mm];
			for (int jj = 0; jj < a.size(); jj++)
				K_path3[ii][mm] -= a[jj]*std::exp(-alpha[jj]*kz);
		}

		// Run GPOF on the modified level 3 samples
		a.clear();
		alpha.clear();
		RunGPOF(K_path3[ii], sp.t3[2] - sp.t3[1], a, alpha, s.tol_svd, s.tol_eig, s.max_num_images);

		// Modify coefficients to go from t-space to kz-space
		for (int jj = 0; jj < a.size(); jj++)
		{
			alpha[jj] = alpha[jj]*sp.T03/((1.0 + J*sp.T03)*sp.k);
			a[jj] = a[jj]*std::exp(sp.k*alpha[jj]);
		}

		// Store level 3 images
		im[ii].a.insert(std::end(im[ii].a), std::begin(a), std::end(a));
		im[ii].alpha.insert(std::end(im[ii].alpha), std::begin(alpha), std::end(alpha));

		// Store the wave number to be used when applying the Sommerfeld identity
		im[ii].k = sp.k;

	}

	return;
						
}


// ==================================================================================
// Computational helpers
// ==================================================================================

/*! \brief Replace the samples of the components that FullKernel() selects by samples of the full kernel (smgf_full is set up without extraction).*/
void DCIM::ReplaceWithFullKernel(SpectralMGF &smgf_full, std::vector<std::vector<std::complex<double>>> &K, std::vector<std::complex<double>> &krho, std::vector<std::complex<double>> &kz, int N, int N_components, bool curl_mode)
{

	std::vector<std::vector<std::complex<double>>> K_full;
	GenerateSpectralSamples(smgf_full, K_full, krho, kz, N, N_components, curl_mode);
	for (int ii = 0; ii < N_components; ii++)
		if (FullKernel(ii, curl_mode))
			K[ii] = K_full[ii];

	return;

}


/*! \brief Helper function to generate spectral MGF samples along a given path, with a given pre-initialized spectral MGF instance.*/
void DCIM::GenerateSpectralSamples(SpectralMGF &smgf, std::vector<std::vector<std::complex<double>>> &K, std::vector<std::complex<double>> &krho, std::vector<std::complex<double>> &kz, int N, int N_components, bool curl_mode)
{

	if (curl_mode)
		if (s.curl_GEM)
			smgf.SetCurlToGEM();
		else
			smgf.SetCurlToGHJ();

	K.resize(N_components);
	for (auto &ii : K) ii.resize(N);

	for (int mm = 0; mm < N; mm++)
	{
		smgf.SetRadialWaveNumber(krho[mm]);

		if (!curl_mode)
		{
			smgf.ComputeSpectralMGF();
			for (int ii = 0; ii < K.size(); ii++)
				K[ii][mm] = smgf.GetResult(ii)*kz[mm];
		}
		else
		{
			smgf.ComputeCurlSpectralMGF();
			for (int ii = 0; ii < K.size(); ii++)
				K[ii][mm] = smgf.GetResultCurl(ii)*kz[mm];
		}
	}

	return;

}


/*! \brief Function to approximate given function samples with expnentials via the GPOF method.

Generalized pencil of functions (Hua, Sarkar): y_k ~ sum_j a_j exp(alpha_j k dt). The pencil parameter L defaults to
N/2; max_num_images > 0 overrides it. Only the M singular values above tol_svd times the largest are kept and the
reduced M x M matrix Z = D_M^-1 U_M^H Y2 V_M is used, so that the leading dimensions match the full SVD returned by
ComputeSVD() for any L (they did not for L > N/2), eigenvalues are ordered by magnitude before the tol_eig cut (LAPACK
does not order them), and data without any nonzero singular value - e.g. a component that vanishes by symmetry -
gives no images instead of dividing by zero singular values.*/
void DCIM::RunGPOF(std::vector<std::complex<double>> &y, double dt, std::vector<std::complex<double>> &a, std::vector<std::complex<double>> &alpha, double tol_svd, double tol_eig, int max_num_images)
{

	a.clear();
	alpha.clear();

	int N = y.size(), L = N/2;
	if (max_num_images > 0)
		L = max_num_images;
	if (N < 2 || L < 1 || L > N - 1)
		throw std::invalid_argument("[ERROR] DCIM::RunGPOF(): The pencil parameter must lie between 1 and N-1 (got " + std::to_string(L) + " for N = " + std::to_string(N) + ").");

	// Scale the data to unit magnitude; identically zero data has no images
	double ymax = 0.0;
	for (const auto &v : y)
		ymax = std::max(ymax, std::abs(v));
	if (!(ymax > 0.0) || !std::isfinite(ymax))
		return;

	// Populate matrices Y1 and Y2 (column major, (N-L) x L) from the scaled data
	int m = (N-L), n = L;
	std::vector<std::complex<double>> Y1 (m*n), Y2 (m*n);
	for (int ii = 0; ii < n; ii++) // traverse columns
	{
		for (int jj = 0; jj < m; jj++) // traverse rows
		{
			Y1[jj + ii*m] = y[jj+ii]/ymax;
			Y2[jj + ii*m] = y[jj+ii+1]/ymax;
		}
	}

	// Full SVD of Y1: u is m x m (leading dimension m), vt = V^H is n x n (leading dimension n)
	std::vector<double> s;
	std::vector<std::complex<double>> u, vt;
	ComputeSVD(Y1, m, n, s, u, vt, false);

	int ns = s.size();
	if (ns < 1 || !(s[0] > 0.0))
		return;

	// Keep the singular values whose ratio to the largest one exceeds tol_svd
	int M;
	for (M = 1; M < ns; M++)
		if (s[M] < tol_svd*s[0])
			break;

	// Z = D_M^-1 U_M^H Y2 V_M, with V_M = (first M rows of vt)^H and U_M the first M columns of u
	std::vector<std::complex<double>> c1 (m*M), c2 (M*M), Z (M*M);
	std::complex<double> cblas_alpha = 1.0, cblas_beta = 0.0;

	cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans, m, M, n, &cblas_alpha,
             &Y2[0], m, &vt[0], n, &cblas_beta, &c1[0], m);

	cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, M, M, m, &cblas_alpha,
             &u[0], m, &c1[0], m, &cblas_beta, &c2[0], M);

	for (int jj = 0; jj < M; jj++)
		for (int ii = 0; ii < M; ii++)
			Z[ii + jj*M] = c2[ii + jj*M]/s[ii];

	// Eigenvalues of Z, ordered by decreasing magnitude; drop (relatively) small and zero ones
	std::vector<std::complex<double>> w;
	ComputeEigenvalues(Z, M, w, false);
	std::sort(w.begin(), w.end(), [](const std::complex<double> &p, const std::complex<double> &q) { return std::abs(p) > std::abs(q); });

	int nwt = 0;
	while (nwt < M && std::abs(w[nwt]) > 0.0 && std::isfinite(std::abs(w[nwt])) && std::abs(w[nwt]) >= tol_eig*std::abs(w[0]))
		nwt++;
	if (nwt == 0)
		return;

	// Set up the system of equations for coefficients
	std::vector<std::complex<double>> Y3 (N*nwt);
	for (int ii = 0; ii < nwt; ii++) // traverse columns
		for (int jj = 0; jj < N; jj++) // traverse rows
			Y3[jj + ii*N] = std::pow(w[ii], jj);

	// Solve the system to get the coefficients
	std::vector<std::complex<double>> b (N);
	for (int jj = 0; jj < N; jj++)
		b[jj] = y[jj]/ymax;
	SolveLeastSquares(Y3, b, N, nwt, 1, false);

	// Extract the computed coefficients
	for (int ii = 0; ii < nwt; ii++)
	{
		a.push_back(b[ii]*ymax);
		alpha.push_back(std::log(w[ii])/dt);
	}

	return;

}





