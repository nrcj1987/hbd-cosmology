#!/usr/bin/env python3
"""
Basic usage example for HBD cosmology package
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hbd import HBDParameters, HBDBackground, HBDPerturbations, HBDSpectra

def main():
    """Demonstrate basic HBD functionality"""
    
    # Initialize with fiducial parameters
    params = HBDParameters()
    background = HBDBackground(params)
    perturbations = HBDPerturbations(background)
    spectra = HBDSpectra(background, perturbations)
    
    print("HBD Cosmology Initialized")
    print(f"Geometric dark energy: Ω_Λ = {background.omega_L:.4f}")
    print(f"Low-ℓ suppression: f_low = {params.f_low}")
    print(f"Mid-ℓ wiggles: A_wiggle = {params.A_wiggle}")
    
    # Compute CMB spectra
    ell, cl_tt, cl_ee, cl_bb, cl_te, cl_pp = spectra.compute_all_spectra(ell_max=2000)
    
    # Plot TT spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogy(ell, ell * (ell + 1) * cl_tt / (2 * np.pi), 'b-', label='HBD TT')
    plt.xlabel('Multipole $\ell$')
    plt.ylabel('$D_\ell^{TT}$ [$\mu K^2$]')
    plt.title('HBD CMB Temperature Spectrum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hbd_tt_spectrum.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot multiple spectra
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(ell, ell * (ell + 1) * cl_tt / (2 * np.pi), 'b-', label='TT')
    plt.ylabel('$D_\ell^{TT}$')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.semilogy(ell, ell * (ell + 1) * cl_ee / (2 * np.pi), 'r-', label='EE')
    plt.ylabel('$D_\ell^{EE}$')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.semilogy(ell, ell * (ell + 1) * cl_bb / (2 * np.pi), 'g-', label='BB')
    plt.xlabel('$\ell$'); plt.ylabel('$D_\ell^{BB}$')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(ell, ell * (ell + 1) * cl_te / (2 * np.pi), 'purple', label='TE')
    plt.xlabel('$\ell$'); plt.ylabel('$D_\ell^{TE}$')
    plt.legend()
    
    plt.suptitle('HBD CMB Power Spectra')
    plt.savefig('hbd_all_spectra.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test falsifiability criteria
    print("\nFalsifiability Criteria:")
    print(f"A_wiggle < 0.038: {params.A_wiggle < 0.038}")
    print(f"r_HBD < 1.8e-4: {params.r_HBD < 1.8e-4}")
    
    if params.A_wiggle < 0.038 and params.r_HBD < 1.8e-4:
        print("→ Model would be excluded at >2σ")
    else:
        print("→ Model consistent with current predictions")

if __name__ == "__main__":
    main()