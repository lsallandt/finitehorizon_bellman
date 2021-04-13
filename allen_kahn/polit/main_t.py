#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:58:33 2020

@author: sallandt

"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import valuefunction_TT, ode, pol_it, optimize
import matplotlib.pyplot as plt
import time

run_set_dynamics = True
if run_set_dynamics == True:
    import set_dynamics
    
load_num = 'V_'

vfun = valuefunction_TT.Valuefunction_TT(load_num, True)
testOde = ode.Ode()
vfun.calc_end_reward_grad = testOde.calc_end_reward_grad
# testOde.test()

t_vec_p = np.load('t_vec_p.npy')
print('t_vec_p', t_vec_p)
T = t_vec_p[-1]
n = vfun.V[0].order(); a = -1; b = 1
x = np.linspace(a, b, n)
max_num_initial_values = 2
min_num_initial_values = 0

step_size = .5
step_size_before = 0.02
max_iter = 1e5
grad_tol = 1e-8
print('set last element to zero')
vfun.V[-1] = 0*vfun.V[-2]

diff_vec = []
for i0 in range(len(vfun.V) - 1):
    diff_vec.append(xe.frob_norm(vfun.V[i0+1] - vfun.V[i0]) / xe.frob_norm(vfun.V[i0]))
    print('frobnorm', xe.frob_norm(vfun.V[i0]))

plt.figure()
plt.plot(diff_vec)
# plt.show()

initial_values = np.zeros(shape=(n, max_num_initial_values-min_num_initial_values))
if min_num_initial_values == 0:
    delta = 2
#    initial_values[:, 0] = sample_0
#    initial_values[:, 0] = sample_failed
#    initial_values[:, 0] = -np.ones(n)
    #initial_values[0:int(n/2), 0] = 1
#    initial_values[:, 0] = np.ones(n); initial_values[0:int(2*n/3), 0] =0x0
#    initial_values[:, 0] = -np.ones(n); initial_values[0:int(n/3), 0] = 1
    initial_values[:,0] = delta * ( x - 1)**2*(x+1)**2
    # initial_values[:,0] = 0.6
    # initial_values[:, 0] = 1
#    initial_values[:,0] = np.exp(-2*(10*x-3)**2)
    #initial_values[:,0] = np.sin(10*x)
#    initial_values[:, 0] = np.exp(-2*(x)**2)
    #initial_values[:,0] = sigma * np.random.uniform(0,1,n)
if max_num_initial_values < 5:
    if max_num_initial_values > 1:
        if min_num_initial_values <= 1:
            initial_values[:, 1-min_num_initial_values] = 1.2
#            initial_values[:, 1] = -np.ones(n); initial_values[0:int(n/2), 1] = 1
#            initial_values[:, 1] = -0.8*(x < -0.33) + 0.8*(x > -0.33) 
#            initial_values[:,1-min_num_initial_values] = np.sin(10*x) + 1
            # initial_values[:,1-min_num_initial_values] = np.sin(10*x) + 1.25*np.abs(x)
        if max_num_initial_values > 2:
            if min_num_initial_values <= 2:
#                initial_values[:, 2-min_num_initial_values] = sample_2
    #            initial_values[:, 2] = inj @ proj @ sample_max
                initial_values[:, 2-min_num_initial_values] = 2*np.exp(-10*(x-0.5)**2) + np.exp(-10*(x+0.5)**2)
            if max_num_initial_values > 3:
                    initial_values[:, 3-min_num_initial_values] = 1.2*np.ones(n)
else:
    for _iter in range(0, max_num_initial_values):
#        if _iter < max_num_initial_values/2:
        if _iter < 0:
            sigma = 1
            initial_values[:,_iter] = sigma * np.random.uniform(-3,3,n)
        else:
            deg = np.random.randint(2,20)
            pol_coeff = np.random.randn(deg)
            pol = np.poly1d(pol_coeff)
            initial_values[:,_iter] = pol(x) * ( x - 1)*(x+1)
            initial_values[:,_iter] = 1.75*initial_values[:,_iter] / np.amax(np.abs(initial_values[:,_iter])) 
# initial_values /= 2

plt.figure()
plt.plot(initial_values)

load_me = np.load('save_me.npy')
lambd = load_me[0]
interval_half = load_me[2]
tau = load_me[3]

steps = np.linspace(0, T, int(T/tau)+1)
m = len(steps)
control_dim = 1
optimize_params = [step_size, step_size_before, max_iter, grad_tol]
optimize_fun = optimize.Open_loop_solver(testOde, optimize_params)
optimize_fun.initialize_new(testOde.calc_end_reward_grad, steps)
# optimize_fun.initialize_new(vfun.calc_grad, steps)

def calc_u_V(t, x):
    return testOde.calc_u(t, x, vfun.calc_grad(t, x))



Pi_cont = np.load('Pi_cont.npy')
def calc_u_riccati(t, x):
    return testOde.calc_u(0, x, 2*Pi_cont@x)


def test_value_function(_step, calc_u, _x0, calc_cost):
    x_vec = np.zeros(shape=(len(steps), n))
    u_vec = np.zeros(shape=(len(steps), control_dim))
    x_vec[0, :] = _x0
    u_vec[0, :] = calc_u(steps[0], x_vec[0, :])
    cost = 1/2*calc_cost(0, x_vec[0, :], u_vec[0, :])
    for i0 in range(len(steps)-1):
        x_vec[i0+1, :] = _step(steps[i0], x_vec[i0, :], u_vec[i0, :])
        u_vec[i0+1, :] = calc_u(steps[i0+1], x_vec[i0+1, :])
        add_cost = calc_cost(steps[i0+1], x_vec[i0+1, :], u_vec[i0+1, :])
        cost += add_cost
    cost -= add_cost/2
    print('cost before add', cost)
    cost += testOde.calc_end_reward(0, x_vec[-1,:])
    print('cost after add', cost)
    return x_vec, u_vec, cost



def test_value_function_experimental(_step, calc_u, _x0, calc_cost):
    sol =  testOde.solver(steps, _x0, calc_u)
    solshapebefore = sol.shape
    solreshaped = sol.reshape((sol.shape[0], -1))
    u_mat = np.zeros((testOde.R_discr.shape[0], solshapebefore[1]))
    for i0 in range(u_mat.shape[1]):
        u_mat[:,i0] = calc_u(steps[i0], sol[:,i0])
    u_matreshaped = u_mat.reshape((u_mat.shape[0], -1))
    rewards = testOde.calc_reward(0, solreshaped, u_matreshaped)
    rew_MC = np.trapz(rewards)
    y_mat = sol
    rew_MC += testOde.calc_end_reward(steps[-1], y_mat[:,-1])
    return y_mat.T, u_mat.T, rew_MC

def calc_opt(x0, u0, calc_cost):
    x_vec, u_vec = optimize_fun.calc_optimal_control(x0, u_hjb.T)
    cost = 1/2*calc_cost(0, x_vec[:, 0], u_vec[:, 0])
    for i0 in range(len(steps)-1):
        add_cost = calc_cost(steps[i0+1], x_vec[:,i0+1 ], u_vec[:, i0+1])
        cost += add_cost
    cost -= add_cost/2
    cost += testOde.calc_end_reward(0, x_vec[:,-1])
    return x_vec.T, u_vec.T, cost

for i0 in range(max_num_initial_values - min_num_initial_values):
    x0 = initial_values[:,i0]
    x_hjb, u_hjb, cost_hjb = test_value_function(testOde.step,calc_u_V,x0,testOde.calc_reward)
    x_lqr, u_lqr, cost_lqr = test_value_function(testOde.step,calc_u_riccati,x0,testOde.calc_reward)
    x_opt, u_opt, cost_opt = calc_opt(x0, u_hjb, testOde.calc_reward)
    print('cost_hjb', cost_hjb, 'eval_V(x0)',  vfun.eval_V(0, x0))
    print('cost_lqr', cost_lqr)
    print('cost_opt', cost_opt, 'norm x_opt', la.norm(x_opt))
    plt.figure()
    plt.plot(steps, u_hjb, steps, u_opt)
    plt.xlabel("time")
    plt.legend(['HJB', 'optimal'])
    plt.grid(True)
    plt.rcParams.update({'font.size': 14})
#     plt.figure()
#     plt.plot(steps, u_hjb, steps, u_lqr, steps, u_opt)
#     plt.xlabel("time")
#     plt.legend(['HJB', 'LQR', 'optimal'])
#     plt.grid(True)
#     plt.rcParams.update({'font.size': 14})
plt.show()
