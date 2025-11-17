import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ellipk, ellipe
from .parameters import HBDParameters

class HBDBackground:
    """HBD background cosmology evolution"""
    
    def __init__(self, params: HBDParameters):
        self.params = params
        self.setup_background()
        
    def setup_background(self):
        """Initialize background quantities"""
        self.H0 = self.params.H0  # km/s/Mpc
        self.h = self.H0 / 100.0
        self.L = self.params.L_throat
        
        # Cosmological parameters in critical density units
        self.omega_b = self.params.Omega_b * self.h**2
        self.omega_cdm = self.params.Omega_cdm * self.h**2
        self.omega_m = self.omega_b + self.omega_cdm
        self.omega_L = 1.0 / (self.L * self.h)**2  # Geometric dark energy
        
        # Radiation density (CMB + neutrinos)
        T_CMB = 2.7255  # K
        self.omega_g = 2.473e-5 / self.h**2  # Photons
        self.omega_ur = 1.7e-5 / self.h**2   # Massless neutrinos
        self.omega_r = self.omega_g + self.omega_ur
        
    def Hubble_parameter(self, a: float) -> float:
        """HBD Hubble parameter H(a) [km/s/Mpc]"""
        H2 = (self.omega_r / a**4 + 
              self.omega_m / a**3 + 
              self.omega_L)
        return self.H0 * np.sqrt(H2)
    
    def conformal_Hubble(self, a: float) -> float:
        """Conformal Hubble parameter H(a) [Mpc^-1]"""
        H_phys = self.Hubble_parameter(a)  # km/s/Mpc
        H_conv = H_phys * 1e3 / 3.08567758128e22  # Convert to s^-1 then Mpc^-1
        return H_conv
    
    def comoving_distance(self, a: float) -> float:
        """Comoving distance integral"""
        def dchi_da(a_val):
            H = self.Hubble_parameter(a_val)
            return 1.0 / (a_val**2 * H)  # dχ/da = 1/(a²H)
        
        result = solve_ivp(dchi_da, [1.0, a], [0.0], 
                          method='RK45', rtol=1e-6)
        return result.y[0, -1]
    
    def luminosity_distance(self, z: float) -> float:
        """Luminosity distance at redshift z"""
        a = 1.0 / (1.0 + z)
        chi = self.comoving_distance(a)
        return chi / a
    
    def growth_function(self, a: float) -> float:
        """Linear growth function D(a)"""
        # Solve growth ODE for HBD cosmology
        def growth_ode(a_val, y):
            D, D_prime = y
            H = self.Hubble_parameter(a_val)
            H_prime = self.Hubble_parameter_derivative(a_val)
            
            D_double_prime = -(3/a_val + H_prime/H) * D_prime + \
                            (3/2) * self.omega_m / (a_val**5 * H**2 / self.H0**2) * D
            return [D_prime, D_double_prime]
        
        # Initial conditions (growing mode)
        a_init = 1e-3
        D_init = a_init
        D_prime_init = 1.0
        
        sol = solve_ivp(growth_ode, [a_init, a], [D_init, D_prime_init],
                       method='RK45', rtol=1e-6)
        D, _ = sol.y[:, -1]
        return D / a  # Normalized to D(a=1)=1
    
    def Hubble_parameter_derivative(self, a: float) -> float:
        """dH/da for growth equation"""
        H = self.Hubble_parameter(a)
        dH2_da = (-4 * self.omega_r / a**5 - 
                  3 * self.omega_m / a**4)
        return 0.5 * dH2_da / H