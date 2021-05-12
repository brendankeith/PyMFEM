'''
   Base class for defining an FEM problem
'''

class FEM_env:

    # constructor
    def __init__(self, mesh, order):
        self.initial_mesh = mesh
        self.order = order
        print("Number of Elements in mesh = " + str(self.initial_mesh.GetNE()))

    def reset(self):
        return None

    def step(self, action):
        return None