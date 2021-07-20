from math import sin, cos, atan2
import mfem.ser as mfem
import numpy as np
import os
from os.path import expanduser, join


mesh_name = 'l-shape-benchmark.mesh'
# mesh_name = 'circle_3_4.mesh'
meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
mesh = mfem.Mesh(meshfile)
mesh.EnsureNodes()

order = 1
dim = mesh.Dimension()
fec = mfem.H1_FECollection(order, dim)
fespace = mfem.FiniteElementSpace(mesh, fec)

# omega = 0
# omega = np.pi/4
omega = 3*np.pi/4
nodes = mesh.GetNodes()
num_nodes = int(nodes.Size()/2)
for i in range(num_nodes):
    x = nodes[2*i]
    y = nodes[2*i+1]
    theta = atan2(y, x)
    if x > 0 and abs(y) < 1e-6:
        theta = 0.0
    elif y < 0:
        theta += 2*np.pi
    delta_theta = theta/(3*np.pi/2) * (np.pi/2 - omega)
    x_tmp = x
    y_tmp = y
    x = x_tmp*cos(delta_theta) - y_tmp*sin(delta_theta)
    y = x_tmp*sin(delta_theta) + y_tmp*cos(delta_theta)
    nodes[2*i] = x
    nodes[2*i+1] = y

sol_sock = mfem.socketstream("localhost", 19916)
sol_sock.precision(8)
zerogf = mfem.GridFunction(fespace)
zerogf.Assign(0.0)
sol_sock.send_solution(mesh, zerogf)
sol_sock.send_text('keys ARjlmp*******')



def ReentrantCorner(omega, meshfile):
    #mesh_name = 'l-shape-benchmark.mesh'
    # mesh_name = 'circle_3_4.mesh'
    #meshfile = expanduser(join(os.path.dirname(__file__), '../..', 'data', mesh_name))
    mesh = mfem.Mesh(meshfile)
    mesh.EnsureNodes()

    order = 1
    dim = mesh.Dimension()
    fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec)

    # omega = 0
    # omega = np.pi/4
    # omega = 3*np.pi/4
    nodes = mesh.GetNodes()
    num_nodes = int(nodes.Size()/2)
    for i in range(num_nodes):
        x = nodes[2*i]
        y = nodes[2*i+1]
        theta = atan2(y, x)
        if x > 0 and abs(y) < 1e-6:
            theta = 0.0
        elif y < 0:
            theta += 2*np.pi
        delta_theta = theta/(3*np.pi/2) * (np.pi/2 - omega)
        x_tmp = x
        y_tmp = y
        x = x_tmp*cos(delta_theta) - y_tmp*sin(delta_theta)
        y = x_tmp*sin(delta_theta) + y_tmp*cos(delta_theta)
        nodes[2*i] = x
        nodes[2*i+1] = y

    #sol_sock = mfem.socketstream("localhost", 19916)
    ##sol_sock.precision(8)
    #zerogf = mfem.GridFunction(fespace)
    #zerogf.Assign(0.0)
    #sol_sock.send_solution(mesh, zerogf)
    #sol_sock.send_text('keys ARjlmp*******')

    return mesh

    
