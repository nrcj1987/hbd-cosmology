import numpy as np
from scipy.interpolate import interp1d
from .parameters import HBDParameters

class HBDSpectra:
    """HBD CMB and matter power spectra"""
    
    def __init__(self, background, perturbations):
        self.background = background
        self.perturbations = perturbations
        self.params = background.params
        
    def cmb_tt_spectrum(self, ell: np.ndarray, cl_lcdm: np.ndarray) -> np.ndarray:
        """HBD TT power spectrum"""
        # Low-ℓ suppression
        low_l_suppress = 1.0 - self.params.f_low * np.exp(-ell/15.0)
        
        # Mid-ℓ wiggles
        wiggle = 1.0 + self.params.A_wiggle * np.sin(ell/self.params.ell_wiggle + np.pi/3.0) * \
                np.exp(-(ell - self.params.ell_wiggle)**2 / 20000.0)
        
        return cl_lcdm * low_l_suppress * wiggle
    
    def cmb_ee_spectrum(self, ell: np.ndarray, cl_lcdm: np.ndarray) -> np.ndarray:
        """HBD EE power spectrum"""
        low_l_suppress = 1.0 - 0.08 * np.exp(-ell/20.0)
        wiggle = 1.0 + 0.04 * np.sin(ell/125.0) * \
                np.exp(-(ell - 520)**2 / 25000.0)
        return cl_lcdm * low_l_suppress * wiggle
    
    def cmb_bb_spectrum(self, ell: np.ndarray, cl_lcdm: np.ndarray) -> np.ndarray:
        """HBD BB power spectrum"""
        # Standard tensor contribution
        bb_standard = cl_lcdm
        
        # HBD vector KK contribution
        bb_hbd = 1.8e-2 * (ell/600.0)**2 * np.exp(-ell/1000.0) * \
                (1.0 + 0.6 * np.sin(ell/130.0))
        
        return bb_standard + self.params.r_HBD * bb_hbd
    
    def cmb_te_spectrum(self, ell: np.ndarray, cl_lcdm: np.ndarray) -> np.ndarray:
        """HBD TE power spectrum"""
        te_wiggle = 2.5 * np.sin(ell/110.0 + np.pi/4.0) * \
                   np.exp(-(ell - 500)**2 / 30000.0)
        return cl_lcdm + te_wiggle
    
    def lensing_spectrum(self, ell: np.ndarray, cl_lcdm: np.ndarray) -> np.ndarray:
        """HBD lensing potential spectrum"""
        enhancement = 1.0 + self.params.A_lens * (ell/1500.0)**2 * np.exp(-ell/2000.0)
        return cl_lcdm * enhancement
    
    def matter_power_spectrum(self, k: np.ndarray, z: float = 0.0) -> np.ndarray:
        """HBD matter power spectrum P(k)"""
        # Scale-dependent growth from 5D effects
        a = 1.0 / (1.0 + z)
        D = self.background.growth_function(a)
        
        # Primordial power spectrum
        P_primordial = self.params.A_s * (k / 0.05)**(self.params.n_s - 1.0)
        
        # Transfer function approximation with HBD modifications
        q = k / (self.background.h * 0.1)  # h/Mpc units
        T = np.log(1.0 + 2.34 * q) / (2.34 * q)
        T *= (1.0 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**(-0.25)
        
        # HBD suppression at large scales
        suppression = 1.0 - 0.1 * np.exp(-(k * self.params.L_throat)**2)
        
        return P_primordial * T**2 * D**2 * suppression**2
    
    def compute_all_spectra(self, ell_max: int = 2500):
        """Compute all CMB spectra"""
        ell = np.arange(2, ell_max + 1)
        
        # Get ΛCDM baseline (in real implementation, this would come from CLASS/CAMB)
        cl_tt_lcdm = self._get_lcdm_baseline(ell, 'TT')
        cl_ee_lcdm = self._get_lcdm_baseline(ell, 'EE') 
        cl_bb_lcdm = self._get_lcdm_baseline(ell, 'BB')
        cl_te_lcdm = self._get_lcdm_baseline(ell, 'TE')
        cl_pp_lcdm = self._get_lcdm_baseline(ell, 'PP')  # Lensing
        
        # Apply HBD modifications
        cl_tt = self.cmb_tt_spectrum(ell, cl_tt_lcdm)
        cl_ee = self.cmb_ee_spectrum(ell, cl_ee_lcdm)
        cl_bb = self.cmb_bb_spectrum(ell, cl_bb_lcdm) 
        cl_te = self.cmb_te_spectrum(ell, cl_te_lcdm)
        cl_pp = self.lensing_spectrum(ell, cl_pp_lcdm)
        
        return ell, cl_tt, cl_ee, cl_bb, cl_te, cl_pp
    
    def _get_lcdm_baseline(self, ell: np.ndarray, spectrum_type: str) -> np.ndarray:
        """Get ΛCDM baseline spectrum (placeholder implementation)"""
        # In real implementation, this would call CLASS/CAMB
        # For now, return approximate forms
        
        if spectrum_type == 'TT':
            # Rough approximation of Planck TT spectrum
            return 6000.0 * ell * (ell + 1) / (2 * np.pi) * np.exp(-ell/2000.0)
        elif spectrum_type == 'EE':
            return 50.0 * ell * (ell + 1) / (2 * np.pi) * np.exp(-ell/1500.0)
        elif spectrum_type == 'BB':
            return 0.1 * ell * (ell + 1) / (2 * np.pi) * np.exp(-ell/1000.0)
        elif spectrum_type == 'TE':
            return 100.0 * ell * (ell + 1) / (2 * np.pi) * np.exp(-ell/1800.0)
        elif spectrum_type == 'PP':  # Lensing
            return 5e-7 * ell**2 * (ell + 1)**2 / (2 * np.pi) * np.exp(-ell/2000.0)
        else:
            raise ValueError(f"Unknown spectrum type: {spectrum_type}")