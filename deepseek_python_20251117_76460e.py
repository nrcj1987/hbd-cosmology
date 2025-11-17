import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HBDParameters:
    """HBD model parameters with fiducial values from paper"""
    
    # HBD-specific parameters
    f_low: float = 0.118          # Low-ℓ suppression amplitude
    A_wiggle: float = 0.048       # Mid-ℓ wiggle amplitude  
    ell_wiggle: float = 512.0     # Wiggle centroid multipole
    r_HBD: float = 1.9e-4         # Tensor-to-scalar ratio
    L_throat: float = 1000.0      # Throat length scale [Mpc]
    A_lens: float = 0.12          # Lensing enhancement
    
    # Standard cosmological parameters
    H0: float = 67.8              # Hubble constant [km/s/Mpc]
    Omega_b: float = 0.0224       # Baryon density
    Omega_cdm: float = 0.120      # CDM density
    tau_reio: float = 0.054       # Optical depth
    n_s: float = 0.965            # Scalar spectral index
    A_s: float = 2.1e-9           # Scalar amplitude
    
    # String theory parameters
    N_flux: int = 100             # Flux number
    T3: float = 1.0               # D3-brane tension
    phi_0: float = 0.1            # Brane field value
    
    def __post_init__(self):
        """Compute derived parameters"""
        self.Omega_m = self.Omega_b + self.Omega_cdm
        self.Omega_L = 1.0 / (self.H0/100.0)**2 / self.L_throat**2
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CLASS/CAMB"""
        return {
            'f_low': self.f_low,
            'A_wiggle': self.A_wiggle,
            'ell_wiggle': self.ell_wiggle,
            'r_HBD': self.r_HBD,
            'L_throat': self.L_throat,
            'A_lens': self.A_lens,
            'H0': self.H0,
            'omega_b': self.Omega_b * (self.H0/100.0)**2,
            'omega_cdm': self.Omega_cdm * (self.H0/100.0)**2,
            'tau_reio': self.tau_reio,
            'n_s': self.n_s,
            'A_s': self.A_s,
        }
    
    @classmethod
    def from_mcmc_chain(cls, chain_file: str):
        """Load parameters from MCMC chain file"""
        # Implementation for loading best-fit parameters
        pass