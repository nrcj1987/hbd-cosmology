from hbd import HBDCosmology
import matplotlib.pyplot as plt

# Initialize HBD cosmology
cosmo = HBDCosmology(
    f_low=0.118,
    A_wiggle=0.048, 
    ell_wiggle=512,
    r_HBD=1.9e-4,
    L_throat=1000.0  # Mpc
)

# Compute CMB spectra
ells, cl_tt, cl_ee, cl_bb, cl_te = cosmo.compute_cmb_spectra(l_max=2500)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(ells, cl_tt, label='HBD TT')
plt.xlabel('$\ell$'); plt.ylabel('$C_\ell^{TT}$')
plt.legend(); plt.show()