#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:02:03 2019

@author: sallandt

Builds system matrices and saves them. Also calculates an initial control. xerus dependencies can be deleted.
"""
import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle

num_valuefunctions = 31
T = 0.3
t_vec = np.linspace(0, T, num_valuefunctions)
print(t_vec)



b = 1 # left end of Domain
horizon = 1
a = -1 # right end of Domain
n = 32 # spacial discretization points that are considered
tau = 1e-3 # time step size
nu = 1 # diffusion constant
lambd = 0.1 # cost parameter
gamma = 0 # discount factor, 0 for no discount
interval_half = 2 # integration area of HJB equation is [-interval_half, interval_half]**n
boundary = 'Neumann' # use 'Neumann' or "Dirichlet
use_full_model = True # if False, model is reduced to r Dimensions
r = n # Model is reduced to r dimensions, only if use_full_model == False
pol_deg = 4
def build_matrices(n, boundary_condition, r = 0):
    s = np.linspace(a, b, n)    # gridpoints
    if boundary_condition == 'Dirichlet':
        print('Dirichlet boundary')
        h = (b-a) / (n+1)
        A = -2*np.diag(np.ones(n), 0) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
        A = nu / h**2 * A
        B = np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1)
        B = 1/(2*h) * B
        Q = tau*h*np.eye(n)
    elif boundary_condition == 'Neumann':
        print('Neumann boundary')
        h = (b-a)/(n-1)             # step size in space
        A = -2*np.diag(np.ones(n), 0) + np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
        A[0,1] = 2; A[n-1, n-2] = 2
        A = nu / h**2 * A
        Q = tau*h*np.eye(n)
        Q[0,0] /=2; Q[n-1,n-1] /=2  # for neumann boundary
    else:
        print('Wrong boundary!')
    _B = (np.bitwise_and(s > -0.4, s < 0.4))*1.0
    B = np.zeros(shape=(n, 1))   
    B[:, 0] = _B
    C = B
    control_dim = B.shape[1]
    R = lambd * np.identity(control_dim)
    P = R*10
    Pi = la.solve_continuous_are(A, B, Q/tau, R)
    return A, B, C, Q, R, P, Pi

def reduce_model(Pi, r, use_full_model, order='lefttoright'):
    n = np.shape(Pi)[0]
    if use_full_model:
        print('did not reduce model, because use_full_model ==', use_full_model)
        proj_full = np.eye(n)
        inj_full = np.eye(n)
        proj = np.eye(n)
        inj = np.eye(n)
        
    else:
        u, v = la.eigh(Pi)
        u_order = np.argsort(np.abs(u))   # sort by absolute values
        u = np.flip(u[u_order])      # sort from largest to smallest EV
        v = np.flip(v[:,u_order],1)
        if order=='lefttoright':
            inj = v[:, :r]
            inj_full = v
            proj = inj.T
            proj_full = inj_full.T
        else:
            print('This part has to be tested!')
            r_half = int(np.floor(r/2))
            perm = np.zeros(shape=r)
            perm[0] = r_half
            for i0 in range(1, r_half+1):
                if(r_half - i0 >= 0):
                    perm[2*i0 - 1] = r_half - i0
                if(r_half + i0 <r):
                    perm[2*i0] = r_half + i0
            perm_mat = np.zeros((len(perm), len(perm)))
            for idx, i in enumerate(perm):
                perm_mat[int(idx), int(i)] = 1
            inj = np.dot(inj, perm_mat)
            proj = inj.T
            
            rr = n
            rr_half = int(np.floor(rr/2))
            perm = np.zeros(shape=rr)
            perm[0] = rr_half
            for i0 in range(1, rr_half+1):
                if(rr_half - i0 >= 0):
                    perm[2*i0 - 1] = rr_half - i0
                if(rr_half + i0 <rr):
                    perm[2*i0] = rr_half + i0
            
            perm_mat = np.zeros((len(perm), len(perm)))
            for idx, i in enumerate(perm):
                perm_mat[int(idx), int(i)] = 1
            #
            inj_full = np.dot(v, perm_mat)
            proj_full = inj_full.T
    return proj, inj, proj_full, inj_full



load = np.zeros([4])
load[0] = lambd; load[1] = gamma; load[2] = interval_half; load[3] = tau

A, B, C, Q, R, P, Pi = build_matrices(n, boundary, 0)
proj, inj, proj_full, inj_full = reduce_model(Pi, r, use_full_model)
P_discr = P * tau
P_inv = la.inv(P)
R_discr = R * tau
R_inv = la.inv(R)

A_proj = proj @ A @ inj
Pi_proj = proj @ Pi @ inj
Q_proj = proj @ Q @ inj
#


np.save("A_proj", A_proj)
np.save("inj", inj)
np.save("proj", proj)
np.save("inj_full", inj_full)
np.save("proj_full", proj_full)
np.save("A", A)
np.save("save_me", load)
np.save("B", B)
np.save("C", C)
np.save("Q", Q)
np.save("Pi_proj", Pi_proj)
np.save("Pi_cont", Pi)
np.save("R", R_discr)
np.save("R_inv", R_inv)
np.save("P", P_discr)
np.save("P_inv", P_inv)
np.save('t_vec', t_vec)
#

'delete from here if you do not want to use xerus'

set_V_new = True
print(set_V_new)

import orth_pol

load_me = np.load('save_me.npy')
interval_half = load_me[2]
print("interval_half", interval_half)
pol, dpol = orth_pol.calc_pol(interval_half, -interval_half, 2)

_round = True
# load_mat = Q_proj
#    load_mat = np.eye(n)
load_mat = Pi
riccati = True

V_prev = 0
# V_prev = xe.load_from_file('V_0')
#V_prev = xe.load_from_file('V_int2')
# previous and new polynomial order
new = 2
r_new = A_proj.shape[0]
pol_deg_vec = [pol_deg+1]*r_new #+ [5]*(32-r_new)
desired_rank = 4
#pol_type = 'Legendre'
pol_type = 'H1'

c_mat = np.zeros(shape=[3,3])
for i0 in range(3):
    for i1 in range(i0+1):
        c_mat[i0,i0 - i1] = pol[i0].c[i1]


if riccati:
    Pi_cont = np.load('Pi_cont.npy')
    proj = np.load('proj.npy')
    inj = np.load('inj.npy')
    Pi_proj = proj @ Pi_cont @ inj



new = new+1
A = np.load('A.npy')
r = A_proj.shape[0]
desired_ranks = [desired_rank]*(r-1)
# desired_ranks = [3, 4, 5, 6, 7, 6, 5, 4, 3]
#desired_ranks = [6]*(r_new-1)# + [2]*(r-r_new)
desired_ranks = [ 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 3 ]
# desired_ranks = [x+1 for x in desired_ranks]

# print("desired_rank", desired_rank)
#V_new = xe.TTTensor.random([new]*r, desired_ranks) 
V_new = xe.TTTensor.random(pol_deg_vec, desired_ranks)
print("pol_deg_vec", V_new.dimensions)
desired_ranks = V_new.ranks()
print("desired_ranks", desired_ranks)
if type(V_prev) == xe.TTTensor and not riccati:
    print("type(V_prev) == xe.TTTensor")
    prev = V_prev.ranks()[1]
    print("prev:", V_prev.ranks())
    for iter_0 in range(r):
        comp = V_prev.get_component(iter_0)
        comp.resize_mode(mode=1, newDim=new, cutPos=prev)
        if iter_0 != 0:
            if comp.dimensions[0] != desired_ranks[iter_0-1]:
                comp.resize_mode(mode=0, newDim=desired_ranks[iter_0-1], cutPos=comp.dimensions[0])
                comp = comp + 0.0000000000000000000001*xe.Tensor.random(comp.dimensions)
        if iter_0 != r-1:
            if comp.dimensions[2] != desired_ranks[iter_0]:
                comp.resize_mode(mode=2, newDim=desired_ranks[iter_0], cutPos=comp.dimensions[2])
                comp = comp + 0.0000000000000000000001*xe.Tensor.random(comp.dimensions)
        V_new.set_component(iter_0, comp)
    V_new.canonicalize_left()


if (type(V_prev) == xe.TTOperator or riccati):
    print("type(V_prev) == xe.TTOperator")
    if not riccati:
        tens_V_prev = xe.Tensor(V_prev)
        print(tens_V_prev.size)
    else:
        Pi_cont_tens = xe.Tensor.from_buffer(load_mat)
        tens_V_prev = Pi_cont_tens
    print("try qtt_sparse to TT")
    full = False
    r_tilde = r
    if int(np.sqrt(tens_V_prev.size) / r) != np.sqrt(tens_V_prev.size) / r:
        sys.exit("wrong dimensions!")
    prev = int(np.log(tens_V_prev.size) / np.log(r_tilde))
    print("prev:", prev)
    tens_V_prev.reinterpret_dimensions([r_tilde]*prev)
#    c1, c2 = xe.indices(2)
#    tens_V_prev(c1,c2) << tens_V_prev(c1,c2) + tens_V_prev(c2,c1)
#    tens_V_prev = tens_V_prev/2
    if prev != 2:
        sys.exit("previous polynomial order !=2 is not yet implemented!")
    tens_V_prev.reinterpret_dimensions([r_tilde]*prev)
    V_new = xe.TTTensor()
    if pol_type == 'Monom':
        _dirac_0 = xe.Tensor.dirac([1, new, 1],[0,0,0])
        _dirac_1 = xe.Tensor.dirac([1, new, 1],[0,1,0])*interval_half
        _dirac_2 = xe.Tensor.dirac([1, new, 1],[0,2,0])*interval_half*interval_half
    elif pol_type == 'Legendre':
        _dirac_0 = xe.Tensor.dirac([1, new, 1],[0,0,0])
        _dirac_1 = xe.Tensor.dirac([1, new, 1],[0,1,0])*interval_half
        _dirac_2 = 2/3*xe.Tensor.dirac([1, new, 1],[0,2,0])*interval_half*interval_half \
            + 1/3*xe.Tensor.dirac([1, new, 1],[0,0,0])*interval_half*interval_half

    elif pol_type == 'H1':
        _dirac_0 = xe.Tensor.dirac([1, new, 1],[0,0,0]) /c_mat[0,0]
        _dirac_1 = xe.Tensor.dirac([1, new, 1],[0,1,0]) /c_mat[1,1]
        _dirac_2 = -1*xe.Tensor.dirac([1, new, 1],[0,0,0]) * c_mat[2,0] /c_mat[0,0] /c_mat[2,2]\
        + xe.Tensor.dirac([1, new, 1],[0,2,0])/c_mat[2,2]
    for iter_0 in range(r):
        for iter_1 in range(r):
            rank_1= 0*xe.TTTensor.random(pol_deg_vec, [1]*(r-1))
            for iter_2 in range(r):
                if iter_2 != iter_1 and iter_2 != iter_0:
                    if _dirac_0.dimensions[1] != pol_deg_vec[iter_2]:
                        _dirac_0.resize_mode(1, pol_deg_vec[iter_2])
                    rank_1.set_component(iter_2, 1*_dirac_0)
                    _dirac_0.resize_mode(1, new)
                elif iter_1 != iter_0:
                    if _dirac_1.dimensions[1] != pol_deg_vec[iter_2]:
                        _dirac_1.resize_mode(1, pol_deg_vec[iter_2])
                    rank_1.set_component(iter_2, 1*_dirac_1)
                    _dirac_1.resize_mode(1, new)
                else:
                    if _dirac_2.dimensions[1] != pol_deg_vec[iter_2]:
                        _dirac_2.resize_mode(1, pol_deg_vec[iter_2])
                    rank_1.set_component(iter_2, 1*_dirac_2)
                    _dirac_2.resize_mode(1, new)
            rank_1.canonicalize_left()
#            print(xe.frob_norm(_dirac_0),xe.frob_norm(_dirac_1),xe.frob_norm(_dirac_2))
            if(full):
                rank_1 = (tens_V_prev[iter_0+1, iter_1+1]) * rank_1
            else:
                rank_1 = (tens_V_prev[iter_0, iter_1]) * rank_1
            if(iter_0 == 0 and iter_1 == 0):
                V_new = rank_1
            else:
                V_new = V_new + rank_1
#        print(iter_0)
        V_new.canonicalize_left()
    # print("ranks before round", V_new.ranks())
    V_new.round(1e-12)
    # print("ranks before round", V_new.ranks())

V_new.move_core(0)

def adapt_ranks(U, S, Vt,smin):
    i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3 = xe.indices(13)
    res = xe.Tensor()
    #S
    Snew = xe.Tensor([S.dimensions[0]+1,S.dimensions[1]+1])
    Snew.offset_add(S, [0,0])
    Snew[S.dimensions[0],S.dimensions[1]] = 0.01 * smin
    
    #U
    onesU = xe.Tensor.ones([U.dimensions[0],U.dimensions[1]])
    Unew = xe.Tensor([U.dimensions[0],U.dimensions[1],U.dimensions[2]+1])
    Unew.offset_add(U, [0,0,0])
    res(i1,i2) << U(i1,i2,k1) * U(j1,j2,k1) * onesU(j1,j2)
    onesU = onesU - res
    res(i1,i2) << U(i1,i2,k1) * U(j1,j2,k1) * onesU(j1,j2)
    onesU = onesU - res
    onesU.reinterpret_dimensions([U.dimensions[0],U.dimensions[1],1])
    onesU/= onesU.frob_norm()
    Unew.offset_add(onesU, [0,0,U.dimensions[2]])
    
    #Vt
    onesVt = xe.Tensor.ones([Vt.dimensions[1],Vt.dimensions[2]])
    Vtnew = xe.Tensor([Vt.dimensions[0]+1,Vt.dimensions[1],Vt.dimensions[2]])
    Vtnew.offset_add(Vt, [0,0,0])
    res(i1,i2) << Vt(k1,i1,i2) * Vt(k1,j1,j2) * onesVt(j1,j2)
    onesVt = onesVt - res
    res(i1,i2) << Vt(k1,i1,i2) * Vt(k1,j1,j2) * onesVt(j1,j2)
    onesVt = onesVt - res
    onesVt.reinterpret_dimensions([1,Vt.dimensions[1],Vt.dimensions[2]])
    onesVt/= onesVt.frob_norm()
    Vtnew.offset_add(onesVt, [Vt.dimensions[0],0,0])
    

    
    return Unew, Snew, Vtnew


    #loop over each component from left to right
    
if _round:
    # print("ranks before kick", V_new.ranks())
    kick_rank = [0]*(V_new.order()-1)
    d = V_new.order()
    Smu_left,Gamma, Smu_right, Theta, U_left, U_right, Vt_left, Vt_right = (xe.Tensor() for i in range(8))
    i1,i2,i3,i4,i5,i6,j1,j2,j3,j4,k1,k2,k3 = xe.indices(13)
    V_new.round(desired_ranks)
    while (V_new.ranks() != desired_ranks):
        print(V_new.ranks())
        for mu in range(r_new):
            V_new.move_core(0)
            # get singular values and orthogonalize wrt the next core mu
            if mu > 0:
                    # get left and middle component
                Gmu_left = V_new.get_component(mu-1)
                Gmu_middle = V_new.get_component(mu)
                (U_left(i1,i2,k1), Smu_left(k1,k2), Vt_left(k2,i3)) << xe.SVD(Gmu_left(i1,i2,i3))
                Gmu_middle(i1,i2,i3) << Vt_left(i1,k2) *Gmu_middle(k2,i2,i3)
                if V_new.ranks()[mu-1] < desired_ranks[mu-1]:
                        U_left, Smu_left, Gmu_middle  = adapt_ranks(U_left, Smu_left, Gmu_middle,1e-8)
#                        print("after kick", V_new.ranks())
                Gmu_middle(i1,i2,i3) << Smu_left(i1,k1)*Gmu_middle(k1,i2,i3)
                V_new.set_component(mu-1, U_left)
                V_new.set_component(mu, Gmu_middle)

            if mu < d - 1:
                # get middle and rightcomponent
                Gmu_middle = V_new.get_component(mu)
                Gmu_right = V_new.get_component(mu+1)
                (U_right(i1,i2,k1), Smu_right(k1,k2), Vt_right(k2,i3)) << xe.SVD(Gmu_middle(i1,i2,i3))
                Gmu_right(i1,i2,i3) << Vt_right(i1,k1) *Gmu_right(k1,i2,i3)
                if V_new.ranks()[mu] < desired_ranks[mu]:
                    U_right, Smu_right, Gmu_right  = adapt_ranks(U_right, Smu_right, Gmu_right,1e-8)
#                    print("after kick", V_new.ranks())
                Gmu_middle(i1,i2,i3) << U_right(i1,i2,k1) * Smu_right(k1,i3)
                V_new.set_component(mu, Gmu_middle)
                V_new.set_component(mu+1, Gmu_right)
        print("after kick", V_new.ranks())
    V_new.move_core(0)



#if _round:
#    print("ranks before kick", V_new.ranks())
#    kick_rank = [0]*(V_new.order()-1)
#    while (V_new.ranks() != desired_ranks):
#        for iter_0 in range(V_new.order() - 1):
#            kick_rank[iter_0] = desired_ranks[iter_0] - V_new.ranks()[iter_0]
##            print(kick_rank[iter_0], iter_0)
#            if (kick_rank[iter_0] <= 0):
#                kick_rank[iter_0] = 1
#        randd = 0.0000000000001 * xe.TTTensor.random(pol_deg_vec, kick_rank)
#        randd.canonicalize_right()
#        print("before", V_new.ranks())
#        V_new = V_new + randd
#        print("after kick", V_new.ranks())
#        V_new.canonicalize_left()
#        print("after canonicalize", V_new.ranks())
#        V_new.round(desired_ranks)
#        print("after round", V_new.ranks(), "desired: ", desired_ranks)
    #V_new.round(V.ranks())
    
    
    
    
    
#print("V_new.ranks()", V_new.ranks())
#print("desired_ranks", desired_ranks)

#V_new.round(1e-6)
print("V_new.ranks()", V_new.ranks())
# if not V_prev == None:
    # pickle.dump(V_prev, open("V_prev", 'wb'))
pickle.dump(0*V_new, open("V_new", 'wb'))
