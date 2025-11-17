import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import k0, k1, k2
from .parameters import HBDParameters

class HBDPerturbations:
    """HBD linear perturbation equations"""
    
    def __init__(self, background):
        self.background = background
        self.params = background.params
        
    def modified_poisson(self, k: float, a: float, delta_rho: float) -> float:
        """Modified Poisson equation with 5D effects"""
        # Convert k to multipole ℓ
        chi = self.background.comoving_distance(a)
        ell = k * chi
        
        # Low-ℓ suppression from 5D leakage
        suppression = 1.0 - self.params.f_low * np.exp(-ell/15.0)
        
        # Standard Poisson term with suppression
        H = self.background.Hubble_parameter(a)
        rho_crit = 3.0 * (H * 1e3 / 3.08567758128e22)**2 / (8 * np.pi * 6.67430e-11)
        rho_crit *= (3.08567758128e22)**2 / 1.989e30  # Convert to M_sun/Mpc³
        
        return -4 * np.pi * 6.67430e-11 * rho_crit * a**2 * delta_rho * suppression
    
    def kk_mode_evolution(self, k: np.ndarray, tau: np.ndarray, mode: int = 1):
        """Solve KK mode evolution equations"""
        m_kk = self.kk_mass_spectrum(mode)
        
        def equations(t, y):
            alpha, alpha_dot = y
            a = np.exp(t)  # Assuming t = ln(a)
            H = self.background.Hubble_parameter(a)
            
            # Damping + oscillation
            alpha_ddot = -2 * H * alpha_dot - (k**2 + m_kk**2) * alpha
            
            # Resonance source term
            source = 0.0
            if abs(k - 1.2e-3) < 1e-4:  # Resonance scale
                source = 0.3 * np.sin(0.01 * t) * np.exp(-t/10000.0)
            
            return [alpha_dot, alpha_ddot + source]
        
        # Initial conditions
        alpha0 = 3e-5  # Initial amplitude
        alpha_dot0 = 0.0
        
        sol = solve_ivp(equations, [tau[0], tau[-1]], [alpha0, alpha_dot0],
                       t_eval=tau, method='RK45', rtol=1e-6)
        
        return sol.y[0]  # Return alpha(tau)
    
    def kk_mass_spectrum(self, n: int) -> float:
        """KK mass spectrum m_n = j_{1,n}/L"""
        # Zeros of Bessel J_1
        j1_zeros = [0.0, 3.8317, 7.0156, 10.1735, 13.3237, 16.4706]
        if n < len(j1_zeros):
            return j1_zeros[n] / self.params.L_throat
        else:
            # Asymptotic: j_{1,n} ≈ π(n + 1/4)
            return np.pi * (n + 0.25) / self.params.L_throat
    
    def vector_anisotropic_stress(self, k: float, tau: float, alpha: float) -> float:
        """Vector KK mode contribution to anisotropic stress"""
        m_kk = self.kk_mass_spectrum(1)  # n=1 mode
        return 0.1 * alpha * k**2 / (k**2 + m_kk**2)  # Dimensionless stress
    
    def isw_5d_contribution(self, k: float, tau: float) -> float:
        """5D ISW effect from bulk scalar modes"""
        if k < 0.001:  # Super-horizon scales
            decay = np.exp(-tau/5000.0)
            oscillation = np.sin(k * tau / 100.0)
            return 1e-6 * decay * oscillation
        return 0.0