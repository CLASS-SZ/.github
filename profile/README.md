# CLASS_SZ

Cosmic Linear Anisotropy Solving System with Machine Learning Accelerated and Accurate CMB, LSS, and Halo Model Observables Computations.

## Installing the code

To install the code, run: 

```bash
pip install classy_sz
```

Check our [evolving documentation](https://class-sz.readthedocs.io/en/latest/index.html).


By default, the neural nets emulators (~1GB of files) will be installed in your home directory. If you're working on a computing cluster or prefer to store the data elsewhere, you can specify a custom directory.

To specify where you want to store the neural nets data, run the following command in your terminal **before** importing the package:

```bash
export PATH_TO_CLASS_SZ_DATA=/path/to/store/class_sz/data
mkdir -p $PATH_TO_CLASS_SZ_DATA/class_sz_data_directory
```

This command sets the `PATH_TO_CLASS_SZ_DATA` variable for the current session.

To ensure this variable is set every time you open a terminal, you can add this line to your `~/.bashrc` or `~/.bash_profile` file automatically using the `echo` command.

For `~/.bashrc` (common for most Linux systems):
```bash
echo -e "\n# Set path for CLASS-SZ data\nexport PATH_TO_CLASS_SZ_DATA=/path/to/store/class_sz/data" >> ~/.bashrc
echo -e "\n# Create directory for CLASS-SZ data\nmkdir -p \$PATH_TO_CLASS_SZ_DATA/class_sz_data_directory" >> ~/.bashrc
```

To apply the changes immediately:
```bash
source ~/.bashrc
```

(Replace `bashrc` by `bash_profile` if you use macOS.)

Now, every time you open a terminal, the `PATH_TO_CLASS_SZ_DATA` environment variable will automatically be set to your specified directory, ensuring the neural nets emulators are always stored in the correct location.





(You may also take a loook at our [legacy example notebooks](https://github.com/CLASS-SZ/notebooks), although these are no longer maintained as we move the material to the docs.)



## Computing 

Have a look at the [notebooks](https://github.com/CLASS-SZ/notebooks), there are loads of examples. Because the code is still evolving, we apologize that some of the examples may not run fully smoothly. 
We shall update our notebook toolbox by the end of 2025. 

The idea is: 

```python
from classy_sz import Class as Class_sz
class_sz = Class_sz()
class_sz.set({'output':'tSZ_1h'}) # ask for cross-correlations, tsz, etc.
class_sz.compute()
```



## Accelerated Computations

To run the machine learning accelerated computations, the idea is to change:

```python
class_sz.compute()
```
to:

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

## For developers 

If you are a developer, you may need to modify the C code and the python wrapper to implement your own functions, or modify existing ones. 

CLASS_SZ functionalities are located in the files:

- [**source/class_sz.c**](https://github.com/CLASS-SZ/class_sz/blob/master/class-sz/source/class_sz.c) for the main CLASS_SZ functions,
- [**tools/class_sz_tools.c**](https://github.com/CLASS-SZ/class_sz/blob/master/class-sz/tools/class_sz_tools.c) for other useful routines,
- [**source/class_sz_clustercounts.c**](https://github.com/CLASS-SZ/class_sz/blob/master/class-sz/source/class_sz_clustercounts.c) for tSZ cluster counts. Since March 2024, CLASS_SZ cluster counts calculations are superseded by [cosmocnc](https://github.com/inigozubeldia/cosmocnc) ([Zubeldia & Bolliet 2024](https://arxiv.org/abs/2403.09589)).

And importantly, in the python and cython files:

- [**python/classy.pyx**](https://github.com/CLASS-SZ/class_sz/blob/master/class-sz/python/classy.pyx) for the Python wrapper,
- [**classy_szfast/classy_szfast.py**](https://github.com/CLASS-SZ/classy_szfast/blob/master/classy_szfast/classy_szfast.py) for the Python wrapper for the emulators,
- [**classy_szfast/classy_sz.py**](https://github.com/CLASS-SZ/classy_szfast/blob/master/classy_szfast/classy_sz.py) for the Python wrapper for cobaya,
- [**classy_szfast/cosmosis_classy_szfast_interface.py**](https://github.com/CLASS-SZ/classy_szfast/blob/master/python/classy_szfast/cosmosis_classy_szfast_interface.py) for the Python wrapper for cosmosis. 

To install the C executable, so you can run the C code, you should install from source and compile:

Clean up and compile:

```bash
$ git clone https://github.com/CLASS-SZ/class_sz
$ git clone https://github.com/CLASS-SZ/get_cosmopower_emus.git
$ cd get_cosmopower_emus
$ pip install -e .
$ cd ..
$ git clone https://github.com/CLASS-SZ/class_sz_data.git
$ cd class_sz_data
$ pip install -e .
$ cd ..
$ cd class_sz/class-sz/python
$ git clone https://github.com/CLASS-SZ/classy_szfast
$ cd ..
$ chmod +x select_makefile.sh
$ ./select_makefile.sh
$ chmod +x download_emulators.sh
$ ./download_emulators.sh
$ make clean
$ make -j
$ export PYTHONPATH=$(pwd)/python/classy_szfast:$PYTHONPATH

```

The `-j` flag speeds up the compilation process by using multiple cores. 

If it installs, run the C code with many power spectra output:

```bash
$ ./class_sz class_sz_test.ini
```

The `.ini` files are the parameter files.

If you want to run CLASS and not do the class_sz part, you can! For example:

```bash
$ ./class_sz explanatory.ini
```

This will just run the standard CLASS code and its calculations. All depends on what output you request: if you request a class_sz observable or not.


## Library and Include Path Configuration

It is often the case that some libraries are not found. In general, setting the following paths appropriately should solve your issues:

```bash
export LIBRARY_PATH=/path/to/your/libs:path/to/gsl:path/to/fftw:$LIBRARY_PATH
export C_INCLUDE_PATH=/path/to/your/includes:path/to/gsl:path/to/fftw:$C_INCLUDE_PATH
export DYLD_LIBRARY_PATH="/path/to/your/libs:$DYLD_LIBRARY_PATH" # (Mac M1 users only)
```

To ensure these paths are set every time you open a terminal, you can add these lines to your `~/.bashrc` or `~/.bash_profile` file automatically using the `echo` command.

For `~/.bashrc` (common for most Linux systems):

```bash
echo -e "\n# Set library paths for class_sz\nexport LIBRARY_PATH=/path/to/your/libs:path/to/gsl/:path/to/fftw/:\$LIBRARY_PATH\nexport C_INCLUDE_PATH=/path/to/your/includes:path/to/gsl:path/to/fftw:\$C_INCLUDE_PATH\nexport DYLD_LIBRARY_PATH=\"/path/to/your/libs:\$DYLD_LIBRARY_PATH\" # (Mac M1 users only)" >> ~/.bashrc
```

To apply the changes immediately:

```bash
source ~/.bashrc
```

For `~/.bash_profile` (common for macOS):

```bash
echo -e "\n# Set library paths for class_sz\nexport LIBRARY_PATH=/path/to/your/libraries:path/to/gsl/:path/to/fftw/:\$LIBRARY_PATH\nexport C_INCLUDE_PATH=path/to/gsl/:path/to/fftw/:\$C_INCLUDE_PATH\nexport DYLD_LIBRARY_PATH=\"/path/to/your/libraries:\$DYLD_LIBRARY_PATH\" # (Mac M1 users only)" >> ~/.bash_profile
```

To apply the changes immediately:

```bash
source ~/.bash_profile
```


## Some Tips to Run on Computer Clusters

Use module load, module show to get GSL and FFTW.
At NERSC/Cori/Perlmutter, the code works with gsl/2.7. (There seems to be a problematic behavior during job submission with gsl/2.5.)

For Monte Carlo analyses, we also recall that Mpi4py needs to be correctly installed. Follow:
[Cobaya MPI Installation Guide](https://cobaya.readthedocs.io/en/latest/installation.html#mpi-parallelization-optional-but-encouraged).

At NERSC, these commands may help for mpi4py:

```bash
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
```


## TensorFlow on Mac M1

To install the new version of CLASS_SZ, you will need TensorFlow (needed for the Cosmopower emulators). On M1/M2, make sure you have the arm64 version of conda (if not, you need to remove your entire conda and install the arm64 version for Apple Silicon).

This video might be helpful: [Installing TensorFlow on M1 Mac](https://www.youtube.com/watch?v=BEUU-icPg78).

Then you can follow the standard TensorFlow installation recipe for M1, e.g., [Medium Article](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706) or the [Apple Developer Forums](https://developer.apple.com/forums/thread/697846).

The following line should fix most issues:

```bash
$ conda install -c apple tensorflow-deps
```

## Tips for cosmopower installation

If issues seem to be related to cosmopower, you can try the following commands before installing class_sz:

```bash
$ module load python
$ python3 -m venv /path/to/your/venv
$ source /path/to/your/venv/bin/activate
#
$ pip install --upgrade pip
#
$ module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
$ MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
#
$ pip install numpy scipy
# these below are probably not needed for cosmopower, but we keep them here as they may solve dependicies.
$ pip install healpy camb
$ pip install astropy h5py setuptools "iminuit>=2.0.0" cachetools matplotlib
$ pip install hankl
$ pip install tf-keras cosmopower mcfit
$ pip install -U --force-reinstall charset-normalizer
$ python3 -c 'import cosmopower as cp'
```

And then install class_sz (see top of page).

(Thanks to V. Irsic and R. de Belsunce for this.) 

## Pre M1 Mac

See Makefile_preM1mac for an example makefile for older Macs (without the M1 chip). Some key points include adding paths involving libomp to LDFLAG and INCLUDES.
In python/setup.py, you may also want to modify the extra_link_args list to contain '-lomp' instead of '-lgomp' and add the libomp library path as well to that list. 
For example, extra_link_args=['-lomp', '-lgsl','-lfftw3','-lgslcblas', '-L/usr/local/opt/libomp/lib/'].

This makefile is not maintained anymore but we keep it for reference. If you need to run class_sz on a pre-M1 Mac and have serious issues, please contact us.
