#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:58:33 2020

@author: sallandt

This code runs the Policy Iteration algorithm using Tensor Trains as function
approximator.
Loads functions from the general 
"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode, pol_it, optimize
import time

run_set_dynamics = True
if run_set_dynamics == True:
    import set_dynamics
    
vfun = valuefunction_TT.Valuefunction_TT()
# vfun.test()
testOde = ode.Ode()



dof_factor = 6  # samples = dof_factor*(ordersoffreedom)
nos_test_set = 1001  # number of samples of the test set


horizon = 1
n_sweep = 1000
rel_val_tol = 1e-4
rel_tol = 1e-2
max_pol_iter = 1
max_iter_Phi = 1

dof = vfun.calc_dof()
nos = dof_factor * dof
print('number of samples', nos)

step_size = 5
step_size_before = 1
max_iter = 1e5
grad_tol = 1e-8
optimize_params = [step_size, step_size_before, max_iter, grad_tol]
calc_opt = optimize.Open_loop_solver(testOde, optimize_params)
# calc_opt.initialize_new(vfun.calc_grad, np.linspace(0, 0.01, 11))
# x_opt, u_opt = calc_opt.calc_optimal_control(np.ones(16), -np.ones((1, 11)))
polit_params = [nos, nos_test_set, n_sweep, rel_val_tol, rel_tol, max_pol_iter, max_iter_Phi, horizon]

# testOde.test()
testpolit = pol_it.Pol_it(vfun, testOde, calc_opt, polit_params)


t00 = time.time()
t01 = time.perf_counter()
# solve HJB
testpolit.solve_HJB()
t10 = time.time()
t11 = time.perf_counter()
print('The calculations took:, time(), perf_counter()', t10 - t00, t11 - t01 )
# xe.save_to_file(testpolit.v.V, 'V_optimized')
