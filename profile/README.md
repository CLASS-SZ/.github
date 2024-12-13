[![GitHub stars](https://img.shields.io/github/stars/CLASS-SZ/class_sz.svg?style=social&label=Star)](https://github.com/CLASS-SZ/class_sz) [![Documentation Status](https://readthedocs.org/projects/class-sz/badge/?version=latest)](https://class-sz.readthedocs.io/en/latest/index.html) [![PyPI version](https://img.shields.io/pypi/v/classy_sz.svg)](https://pypi.org/project/classy_sz/)
[![arXiv](https://img.shields.io/badge/arXiv-2310.18482-b31b1b.svg)](https://arxiv.org/abs/2310.18482)






# CLASS_SZ

Cosmic Linear Anisotropy Solving System with Machine Learning Accelerated and Accurate CMB, LSS, and Halo Model Observables Computations.

*Class_sz is compatible with **Jax** and now allows for **automatic differentiation** on some of its output, see [here](https://class-sz.readthedocs.io/en/latest/notebooks/classy_szfast_matter_pk_linear.html#Gradients-at-all-k's) for an example on the matter power spectrum. The code can now be used in Hamiltonian Monte Carlo and Simulation Based Inference pipelines.* 

## Documentation

Check our [evolving documentation](https://class-sz.readthedocs.io/en/latest/index.html).

## Installation


To install the code, run: 

```bash
pip install classy_sz
```

(Note that this does not currenty run on Windows OS. If you have a Windows laptop, just install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and make your life easier.)

## Package data 

By default, the neural nets emulators (~1GB of files) will be installed in your home directory. If you're working on a computing cluster or prefer to store the data elsewhere, you can specify a custom directory.

To specify where you want to store the neural nets data, run the following command in your terminal **before** importing the package:

```bash
export PATH_TO_CLASS_SZ_DATA=/path/to/store/class_sz/data
mkdir -p $PATH_TO_CLASS_SZ_DATA/class_sz_data_directory
```

This command sets the `PATH_TO_CLASS_SZ_DATA` variable for the current session.

To ensure this variable is set every time you open a terminal, you can add this line to your `~/.bashrc` or `~/.bash_profile` file automatically using the `echo` command.

For `~/.bashrc` (common for most Linux systems), type in your terminal:
```bash
echo -e "\n# Set path for CLASS-SZ data\nexport PATH_TO_CLASS_SZ_DATA=/path/to/store/class_sz/data" >> ~/.bashrc
echo -e "\n# Create directory for CLASS-SZ data\nmkdir -p \$PATH_TO_CLASS_SZ_DATA/class_sz_data_directory" >> ~/.bashrc
```

To apply the changes immediately:
```bash
source ~/.bashrc
```

(If you use macOS, use `.bash_profile` instead of `bashrc`, replace accordingly above.)

Now, every time you open a terminal, the `PATH_TO_CLASS_SZ_DATA` environment variable will automatically be set to your specified directory, ensuring the neural nets emulators are always stored in the correct location.



(You may also take a loook at our [legacy example notebooks](https://github.com/CLASS-SZ/notebooks), although these are no longer maintained and may not run. We are in the process of moving the material to the [docs](https://class-sz.readthedocs.io/en/latest/index.html) by the end of 2024.)



## Accelerated Computations

To run the machine learning accelerated computations, the idea is to use:

```python
class_sz.compute_class_szfast()
```

In a bit more details, say you are interested in CMB $C_\ell$'s:

```python
from classy_sz import Class as Class_sz

cosmo_params = {
'omega_b': 0.02242,
'omega_cdm':  0.11933,
'H0': 67.66, 
'tau_reio': 0.0561,
'ln10^{10}A_s': 3.047,
'n_s': 0.9665,

'N_ncdm': 1,
'N_ur': 2.0328,
'm_ncdm': 0.06    
}
class_sz = Class_sz()
class_sz.set(cosmo_params)
class_sz.set({
'output':'tCl,lCl,pCl',
'skip_background_and_thermo': 1, # do you want exact solution for background? yes: 1, no: 0 (if "no" you can access exact background quantities via emulators).
})

class_sz.compute_class_szfast()

lensed_cls = class_sz.lensed_cl()
l_fast = lensed_cls['ell']
cl_tt_fast = lensed_cls['tt']
cl_ee_fast = lensed_cls['ee']
cl_te_fast = lensed_cls['te']
cl_pp_fast = lensed_cls['pp']
```

## Some basic info

CLASS_SZ is as fast as it gets, with full parallelization, implementation of high-accuracy neural network emulators, and Fast Fourier Transforms.

CLASS_SZ has been built as an extension of Julien Lesgourgues's [CLASS](https://github.com/lesgourg/class_public) code, therefore the halo model and LSS calculations (essentially based on distances and matter clustering) are always consistent with the cosmological model computed by CLASS. We are doing our best to keep up with CLASS version updates. We are currently working on updating to CLASS v3. 

CLASS_SZ is initially based on Eiichiro Komatsu’s Fortran code [SZFAST](http://wwwmpa.mpa-garching.mpg.de/~komatsu/CRL/clusters/szpowerspectrumks/).

CLASS_SZ's outputs are regularly cross-checked with other CMBxLSS codes, such as:

- [cosmocnc](https://github.com/inigozubeldia/cosmocnc)
- [hmvec](https://github.com/simonsobs/hmvec/tree/master/hmvec)
- [ccl](https://github.com/LSSTDESC/CCL)
- [HaloGen](https://github.com/EmmanuelSchaan/HaloGen/tree/master)
- [yxg](https://github.com/nikfilippas/yxg)
- [halomodel_cib_tsz_cibxtsz](https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz)


## Using the Code

The **class_sz** code is public.

If you use it, please cite:

- [CLASS_SZ: I Overview (Bolliet et al. 2024)](https://arxiv.org/abs/2310.18482)
- [Projected-field kinetic Sunyaev-Zel'dovich Cross-correlations: halo model and forecasts (Bolliet et al. 2023)](https://iopscience.iop.org/article/10.1088/1475-7516/2023/03/039)

If you use accelerated computations, please cite:

- [High-accuracy emulators for observables in LCDM, Neff+LCDM, Mnu+LCDM and wCDM cosmologies (Bolliet et al. 2023)](https://inspirehep.net/literature/2638458)
- [COSMOPOWER: emulating cosmological power spectra for accelerated Bayesian inference from next-generation surveys (Spurio Mancini et al. 2021)](https://arxiv.org/abs/2106.03846)

If you use thermal SZ power spectrum and cluster counts calculations, please cite:

- [Including massive neutrinos in thermal Sunyaev Zeldovich power spectrum and cluster counts analyses (Bolliet et al. 2020)](https://arxiv.org/abs/1906.10359)
- [Dark Energy from the Thermal Sunyaev Zeldovich Power Spectrum (Bolliet et al. 2017)](https://arxiv.org/abs/1712.00788)
- [The Sunyaev-Zel'dovich angular power spectrum as a probe of cosmological parameters (Komatsu and Seljak, 2002)](https://arxiv.org/abs/astro-ph/0205468)

In all these cases, please also cite the original CLASS papers:

- [CLASS I: Overview (Lesgourgues, 2011)](https://arxiv.org/abs/1104.2932)
- [CLASS II: Approximation schemes (Blas, Lesgourgues, Tram, 2011)](http://arxiv.org/abs/1104.2933)

As well as other references listed here: [http://class-code.net](http://class-code.net)

