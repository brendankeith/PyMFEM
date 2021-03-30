'''
   RL AMR example 1
      This is a version of MFEM Example 1 with an adaptive mesh
      refinement loop where the optimal bulk parameter is learned
'''
from mfem import path
from mfem.common.arg_parser import ArgParser
import mfem.ser as mfem
from mfem.ser import intArray
from os.path import expanduser, join
import numpy as np
from problem_fem import *
import os

def_meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', 'star.mesh'))

def run(order = 1, static_cond = False,
        meshfile = def_meshfile, visualization = False):

   if (visualization):
      sol_sock = mfem.socketstream("localhost",19916)
      sol_sock.precision(8)

   mesh = mfem.Mesh(meshfile, 1,1)
   mesh.UniformRefinement()

   poisson = fem_problem(mesh,order)
   poisson.setup()
   poisson.AssembleAndSolve()

   max_dofs = 50000
   it = 0
   while True:
     cdofs = poisson.fespace.GetTrueVSize();
     print("AMR iteration " + str(it))
     print("Number of unknowns: " + str(cdofs))
     poisson.AssembleAndSolve() 

     if (cdofs > max_dofs):
        print("Reached the maximum number of dofs. Stop.")
        break

     if (visualization):
        sol_sock.send_solution(poisson.mesh,  poisson.x)
        
     errors = poisson.GetLocalErrors()
     print("Total l2-norm of estimated errors = " + str(errors.Norml2()) )

   #   This is where we get the new theta from the agent
   # .... 
     theta = 0.7
     poisson.RefineAndUpdate(theta)    
     it = it +1
   
if __name__ == "__main__":
   parser = ArgParser(description='Ex1 (Laplace Problem)')
   parser.add_argument('-m', '--mesh',
                       default = '../../data/star.mesh',
                       action = 'store', type = str,
                       help='Mesh file to use.')
   parser.add_argument('-vis', '--visualization',
                       action = 'store_true',
                       help='Enable GLVis visualization')
   parser.add_argument('-o', '--order',
                       action = 'store', default = 1, type=int,
                       help = "Finite element order (polynomial degree) or -1 for isoparametric space.");
   parser.add_argument('-sc', '--static-condensation',
                       action = 'store_true', 
                       help = "Enable static condensation.")
   args = parser.parse_args()
   parser.print_options(args)

   order = args.order
   static_cond = args.static_condensation
   meshfile =expanduser(join(os.path.dirname(__file__), args.mesh))
   visualization = args.visualization

   run(order=order,
       static_cond=static_cond,
       meshfile=meshfile,
       visualization=visualization)
