# Installation Guidelines for the drl4marking branch (From Socratis August 18th 2021)

Create a brand new python environment (conda/pyenv). 

All steps below are done with python-3.6.4 and with SWIG version 4.0.2. 

## Install mfem c++ library
````
git clone https://github.com/mfem/mfem.git
cd mfem
git checkout drl4marking
make serial MFEM_SHARED=YES -j
make install
````
## Install PyMFEM (Brendan's fork)
````
git clone https://github.com/brendankeith/PyMFEM.git
cd PyMFEM
git checkout drl4marking
pip install -r drl-requirements.txt
python setup.py install --skip-ext --mfem-prefix="MFEM_DIR/mfem" (MFEM_DIR is the path to the mfem installation from the previous step)
````

