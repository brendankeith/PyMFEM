'''
   Class for defining an fem problem, solve and return element error estimators.
'''
from mfem import path
import mfem.ser as mfem
from mfem.ser import intArray
import numpy as np
class fem_problem:
    one  = mfem.ConstantCoefficient(1.0)
    zero = mfem.ConstantCoefficient(0.0)
    # constructor
    def __init__(self, mesh_, order_):
       self.mesh = mesh_
       self.order = order_
       print("Number of Elements in mesh = " + str(self.mesh.GetNE()))

    def setup(self):
        print("Setting up Poisson problem ")
        dim = self.mesh.Dimension()
        fec = mfem.H1_FECollection(self.order, dim)
        self.fespace = mfem.FiniteElementSpace(self.mesh, fec)
        self.a = mfem.BilinearForm(self.fespace)
        self.b = mfem.LinearForm(self.fespace)
        integ = mfem.DiffusionIntegrator(self.one)
        self.a.AddDomainIntegrator(integ)
        self.b.AddDomainIntegrator(mfem.DomainLFIntegrator(self.one))
        self.x = mfem.GridFunction(self.fespace)
        self.x.Assign(0.0)
        self.ess_bdr = intArray(self.mesh.bdr_attributes.Max())
        self.ess_bdr.Assign(1)
        self.flux_fespace = mfem.FiniteElementSpace(self.mesh, fec, dim)
        self.estimator =  mfem.ZienkiewiczZhuEstimator(integ, self.x, self.flux_fespace,
                                                       own_flux_fes = False)

        self.refiner = mfem.ThresholdRefiner(self.estimator)
        self.refiner.SetTotalErrorFraction(0.7)

      #   print("Poisson problem setup finished ")

    def AssembleAndSolve(self):
        self.a.Assemble()
        self.b.Assemble()
        ess_tdof_list = intArray()
      #   self.x.ProjectBdrCoefficient(self.zero, self.ess_bdr)
        self.fespace.GetEssentialTrueDofs(self.ess_bdr, ess_tdof_list)
        A = mfem.OperatorPtr()     
        B = mfem.Vector();  X = mfem.Vector()
        self.a.FormLinearSystem(ess_tdof_list, self.x, self.b, A, X, B, 1)
        AA = mfem.OperatorHandle2SparseMatrix(A)     
        M = mfem.GSSmoother(AA)
        mfem.PCG(AA, M, B, X, 3, 200, 1e-12, 0.0)
        self.a.RecoverFEMSolution(X,self.b,self.x)
      #   print("Poisson problem solved ")

    def GetLocalErrors(self):
        errors = self.estimator.GetLocalErrors() 
        return errors

    def RefineAndUpdate(self, theta):
        self.refiner.SetTotalErrorFraction(theta)
        self.refiner.Apply(self.mesh)
        self.fespace.Update()
        self.x.Update()
        self.a.Update()
        self.b.Update()
      #   print("Mesh refined and updated ")





