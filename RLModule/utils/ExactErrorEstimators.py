'''

    Fake estimator classes for exact solutions

'''
import mfem.ser as mfem

class L2ErrorEstimator(mfem.ErrorEstimator):

    def __init__(self, x : mfem.GridFunction, soln_coeff : mfem.Coefficient):
        self.x = x
        self.soln_coeff = soln_coeff
        self.element_errors = mfem.Vector(0)
        self.irs =[mfem.IntRules.Get(i, 10) for i in range(mfem.Geometry.NumGeom)]

    def GetTotalError(self):
        return self.x.ComputeL2Error(self.soln_coeff, self.irs)

    def GetLocalErrors(self):
        NE = self.x.FESpace().GetNE()
        self.element_errors.SetSize(NE)
        total_error = self.x.ComputeElementL2Errors(self.soln_coeff,self.element_errors, self.irs)
        return self.element_errors

    def Reset(self):
        self.element_errors.SetSize(0)
        return None

class H10ErrorEstimator(mfem.ErrorEstimator):

    def __init__(self, x : mfem.GridFunction, gradsoln_coeff : mfem.VectorCoefficient):
        self.x = x
        self.gradsoln_coeff = gradsoln_coeff
        self.element_errors = mfem.Vector(0)
        self.irs =[mfem.IntRules.Get(i, 10) for i in range(mfem.Geometry.NumGeom)]

    def GetTotalError(self):
        return self.x.ComputeGradError(self.gradsoln_coeff, self.irs)

    def GetLocalErrors(self):
        NE = self.x.FESpace().GetNE()
        self.element_errors.SetSize(NE)
        total_error = self.x.ComputeElementGradErrors(self.gradsoln_coeff,self.element_errors, self.irs)
        return self.element_errors

    def Reset(self):
        self.element_errors.SetSize(0)
        return None

class H1ErrorEstimator(mfem.ErrorEstimator):

    def __init__(self, x : mfem.GridFunction, soln_coeff : mfem.Coefficient, gradsoln_coeff : mfem.VectorCoefficient):
        self.x = x
        self.soln_coeff = soln_coeff
        self.gradsoln_coeff = gradsoln_coeff
        self.element_errors = mfem.Vector(0)
        self.irs =[mfem.IntRules.Get(i, 10) for i in range(mfem.Geometry.NumGeom)]

    def GetTotalError(self):
        return self.x.ComputeH1Error(self.soln_coeff,self.gradsoln_coeff)

    def GetLocalErrors(self):
        NE = self.x.FESpace().GetNE()
        self.element_errors.SetSize(NE)
        total_error = self.x.ComputeElementH1Errors(self.soln_coeff,self.gradsoln_coeff,self.element_errors)
        return self.element_errors

    def Reset(self):
        self.element_errors.SetSize(0)
        return None