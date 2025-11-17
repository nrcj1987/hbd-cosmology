import subprocess
import os
import tempfile
import numpy as np
from .parameters import HBDParameters

class HBDClassInterface:
    """Interface between HBD model and CLASS Boltzmann code"""
    
    def __init__(self, params: HBDParameters):
        self.params = params
        
    def generate_class_input(self, output_file: str = None):
        """Generate CLASS parameter file for HBD cosmology"""
        
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ini').name
        
        class_params = {
            'output': 'tCl, pCl, lCl',
            'l_max_scalars': 2500,
            'lensing': 'yes',
            
            # Standard parameters
            'h': self.params.H0 / 100.0,
            'omega_b': self.params.Omega_b * (self.params.H0/100.0)**2,
            'omega_cdm': self.params.Omega_cdm * (self.params.H0/100.0)**2,
            'tau_reio': self.params.tau_reio,
            'n_s': self.params.n_s,
            'A_s': self.params.A_s,
            
            # HBD modifications (would require modified CLASS)
            'HBD_f_low': self.params.f_low,
            'HBD_A_wiggle': self.params.A_wiggle,
            'HBD_ell_wiggle': self.params.ell_wiggle,
            'HBD_r': self.params.r_HBD,
            'HBD_L_throat': self.params.L_throat,
        }
        
        with open(output_file, 'w') as f:
            f.write("# HBD cosmology parameters for CLASS\n\n")
            for key, value in class_params.items():
                f.write(f"{key} = {value}\n")
                
        return output_file
    
    def run_class(self, param_file: str):
        """Run CLASS with given parameter file"""
        try:
            # This would call the CLASS executable
            # result = subprocess.run(['class', param_file], capture_output=True, text=True)
            # return result
            print(f"Would run CLASS with: {param_file}")
            return {"status": "success", "cl_file": "dummy_cl.dat"}
        except Exception as e:
            print(f"CLASS run failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def load_class_output(self, cl_file: str):
        """Load CLASS output spectra"""
        # Placeholder implementation
        ell = np.arange(2, 2501)
        # In real implementation, parse CLASS output file
        return ell, np.zeros_like(ell), np.zeros_like(ell), np.zeros_like(ell), np.zeros_like(ell)