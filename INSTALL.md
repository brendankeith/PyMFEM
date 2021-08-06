# Installation Guidelines for the drl4marking branch (From Socratis August 5th 2021)

All steps below are done with python-3.6.3

1. Download and install MFEM as shared library (suppose we are working inside the directory "topdir").
   Inside topdir execute: 
   (a) git clone https://github.com/mfem/mfem.git
   (b) cd mfem
   (c) git checkout drl4marking
   (d) make serial MFEM_SHARED=YES -j (it's important that MFEM is built as shared library)
   (e) make install (this will create the shared library and include files in topdir/mfem/mfem)

2. Download PyMFEM (Brendan's fork)
   Inside topdir execute: 
   (a) git clone https://github.com/brendankeith/PyMFEM.git
   (b) cd PyMFEM
   (c) git checkout drl4marking
   (d) pip install -r requirements.txt (only if the required dependencies are not already installed)
   (e) python setup.py clean --swig
   (f) python setup.py install --swig
   (g) python setup.py install --ext-only --mfem-prefix="topdir/mfem/mfem"
   (h) python setup.py install --skip-ext

Note: If you want to install from github repo instead from a local mfem branch you skip step 1 and replace   
step 2(g) with python setup.py install --ext-only --mfem-branch drl4marking
   
