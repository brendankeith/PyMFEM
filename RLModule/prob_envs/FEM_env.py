'''
   Base class for defining an FEM problem
'''
import os
from os.path import expanduser, join
import mfem.ser as mfem

class FEM_env:

    # constructor
    def __init__(self, **kwargs):
        mesh_name = kwargs.get('mesh_name','l-shape.mesh')
        num_unif_ref = kwargs.get('num_unif_ref',1)
        order = kwargs.get('order',1)

        meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
        mesh = mfem.Mesh(meshfile, 1,1)
        for _ in range(num_unif_ref):
            mesh.UniformRefinement()
        self.initial_mesh = mesh
        self.order = order
        print("Number of Elements in mesh = " + str(self.initial_mesh.GetNE()))

    def reset(self):
        return None

    def step(self, action):
        return None