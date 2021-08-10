# Installation Guidelines for the drl4marking branch on LC (From Andrew August 6th 2021)

# use conda to create a venv with the correct python version - in /usr/workspace

conda create --name py364 python=3.6.4
conda actiate py364
cd /usr/workspace/gillette/venvs/
python -m venv drl4mkg
conda deactivate
source /usr/workspace/gillette/venvs/drl4mkg/bin/activate
python --version   # Python 3.6.4 :: Anaconda, Inc.
pip install --upgrade pip
pip install -r drl-pip-req.txt # contents of the txt file are copied at the bottom of this file
pip install mfem

# now make MFEM using drl4marking branch - in /usr/workspace

mkdir /usr/workspace/gillette/venvs/mfem4drl
cd /usr/workspace/gillette/venvs/mfem4drl 
git clone https://github.com/mfem/mfem.git
cd mfem/
git checkout drl4marking
make serial MFEM_SHARED=YES -j
make install

# install swig using spack; end of installation will state build path in terminal window

cd /g/g12/gillette/projects/drl_SI
git clone https://github.com/spack/spack.git
cd spack/
./bin/spack install swig

# export swig path, then do PyMFEM build

export PATH=/g/g12/gillette/projects/drl_SI/spack/opt/spack/linux-rhel7-sandybridge/gcc-4.9.3/swig-4.0.2-6ctyo2poemmm5m332zmfp6srgeixjbbn/bin:$PATH
cd ~/projects/drl_SI/PyMFEM/
python setup.py clean --swig 
python setup.py install --swig
python setup.py install --ext-only --mfem-prefix="/usr/workspace/gillette/venvs/mfem4drl/mfem/mfem"   
python setup.py install --skip-ext

# sample usage from LC log in (adjust path names as needed)

cd /g/g12/gillette/projects/drl_SI/PyMFEM/RLModule/examples
salloc -ppdebug
conda deactivate
source /usr/workspace/gillette/venvs/drl4mkg/bin/activate
export PYTHONPATH="/g/g12/gillette/projects/drl_SI/PyMFEM/RLModule"
export PATH=/g/g12/gillette/projects/drl_SI/spack/opt/spack/linux-rhel7-sandybridge/gcc-4.9.3/swig-4.0.2-6ctyo2poemmm5m332zmfp6srgeixjbbn/bin:$PATH
python Test9_Reentrant_Corner.py


# the file drl-pip-req.txt should conatin the following:

absl-py                  ==0.12.0
aiohttp                  ==3.7.4.post0
aiohttp-cors             ==0.7.0
aioredis                 ==1.3.1
astunparse               ==1.6.3
async-timeout            ==3.0.1
atari-py                 ==0.2.9
attrs                    ==21.2.0
blessings                ==1.7
cached-property          ==1.5.2
cachetools               ==4.2.2
certifi                  ==2020.12.5
chardet                  ==4.0.0
click                    ==8.0.1
cloudpickle              ==1.6.0
cmake                    ==3.18.4.post1
colorama                 ==0.4.4
colorful                 ==0.5.4
contextvars              ==2.4
cycler                   ==0.10.0
dataclasses              ==0.8
dm-tree                  ==0.1.6
drawnow                  ==0.72.5
filelock                 ==3.0.12
flatbuffers              ==1.12
future                   ==0.18.2
gast                     ==0.4.0
google-api-core          ==1.28.0
google-auth              ==1.30.1
google-auth-oauthlib     ==0.4.4
google-pasta             ==0.2.0
googleapis-common-protos ==1.53.0
gpustat                  ==0.4.1
grpcio                   ==1.34.1
gym                      ==0.18.0
h5py                     ==3.1.0
hiredis                  ==2.0.0
idna                     ==2.10
idna-ssl                 ==1.1.0
immutables               ==0.15
importlib-metadata       ==4.0.1
jsonschema               ==3.2.0
keras-nightly            ==2.5.0.dev2021032900
Keras-Preprocessing      ==1.1.2
kiwisolver               ==1.3.1
lz4                      ==3.1.3
Markdown                 ==3.3.4
matplotlib               ==3.3.4
msgpack                  ==1.0.2
multidict                ==5.1.0
numpy                    ==1.19.5
nvidia-ml-py3            ==7.352.0
oauthlib                 ==3.1.0
opencensus               ==0.7.13
opencensus-context       ==0.1.2
opencv-python            ==4.5.2.52
opencv-python-headless   ==4.3.0.36
opt-einsum               ==3.3.0
packaging                ==20.9
pandas                   ==1.1.5
Pillow                   ==7.2.0
prometheus-client        ==0.10.1
protobuf                 ==3.17.1
psutil                   ==5.8.0
py-spy                   ==0.3.7
pyasn1                   ==0.4.8
pyasn1-modules           ==0.2.8
pyglet                   ==1.5.0
pyparsing                ==2.4.7
pyrsistent               ==0.17.3
python-dateutil          ==2.8.1
pytz                     ==2021.1
PyYAML                   ==5.4.1
ray                      ==1.3.0
redis                    ==3.5.3
requests                 ==2.25.1
requests-oauthlib        ==1.3.0
rsa                      ==4.7.2
scipy                    ==1.5.4
setuptools               ==57.0.0
six                      ==1.15.0
svgwrite                 ==1.4.1
tabulate                 ==0.8.9
tensorboard              ==2.5.0
tensorboard-data-server  ==0.6.1
tensorboard-plugin-wit   ==1.8.0
tensorboardX             ==2.2
tensorflow               ==2.5.0
tensorflow-estimator     ==2.5.0
termcolor                ==1.1.0
torch                    ==1.8.1
typing-extensions        ==3.7.4.3
urllib3                  ==1.26.4
Werkzeug                 ==2.0.1
wheel                    ==0.36.2
wrapt                    ==1.12.1
yarl                     ==1.6.3
zipp                     ==3.4.1