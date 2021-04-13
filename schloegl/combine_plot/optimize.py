#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:07:29 2020

@author: sallandt
"""
import numpy as np
"""
Calculates an optimal control using gradient descent of an optimal control problem
\int_0^T x^t Q x + u^t R u dt
where
\dot x = A x + NL(x) + B u
and T is the final time and t is the matrix transposed

Variables:
    A system matrix of size n,n
    B system matrix of size n,control_dim
    Q system matrix of size n,n
    R system matrix of size control_dim,control_dim
    tau time step in discretization of ODE
    m Number of time steps, i.e. T = tau * m
    n A.shape[1]
    control_dim B.shape[1]
    step is a function that solves the ODE, i.e.
    x(tau) = step(x(0), u(0))
"""

import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy
import time

class Open_loop_solver:
    def __init__(self, ode, optimize_params, calc_final = None, t_steps = None):
        self.ode = ode # need step adjoint(x, u); step(t, x, u)
        load_me = np.load('save_me.npy')
        [self.step_size, self.step_size_before, self.max_iter, self.grad_tol] = optimize_params
        self.calc_final = calc_final
        self.t_steps = t_steps


    def calc_optimal_control(self, x0, u0):
        u_opt = u0
        grad1 = 0*u0
        i0 = 0
        curr_gradient_norm = self.grad_tol + 1
        while curr_gradient_norm > self.grad_tol and i0 < self.max_iter:
            grad2 = self.gradJp(x0, u_opt)
            direction = -self.step_size*(grad2 + self.step_size_before*grad1)
            u_opt =  u_opt + direction
            curr_gradient_norm = np.sqrt(np.trapz(grad2**2, self.t_steps))
            i0 += 1
            if i0 % 100 == 0:
                print('i0, curr_gradient_norm', i0, curr_gradient_norm)
        # print('i0, curr_gradient_norm', i0, curr_gradient_norm)
        x_opt = self.solve_fixed_u(x0, u_opt)
        return x_opt, u_opt


    def solve_fixed_u(self, x0, u):
        x= np.zeros((x0.size, len(self.t_steps)))
        x[:,0] = x0
        for i0 in range(len(self.t_steps)-1):
            x[:,i0+1] = self.ode.step(self.t_steps[i0], x[:,i0], u[:, i0])
        return x


    def gradJp(self, x0, u):
        ygrad = self.solve_fixed_u(x0, u)
        p = self.solve_peq(ygrad, u)
        c =  2*self.ode.R @ u +self.ode.B.T @ p
        return c


    def solve_peq(self, x, u):
        p = np.zeros(shape=(x.shape[0], u.shape[1]))
        p[:,-1] = self.calc_final(self.t_steps[-1], x[:, -1])
        for i0 in range(p.shape[1]-1, 0, -1):
            p[:, i0-1] = self.ode.step_adjoint(self.t_steps[i0], p[:,i0], x[:,i0])
    #        p[:, i0-1] = np.dot(A_tilde_T, p[:, i0] + tau*(x[:,i0] * (B.T @ p[:,i0]) + (1.5 - 0.15*x[:,i0])* np.exp(-0.1*x[:,i0]) * p[:,i0] + 2*Q@x[:,i0] ))
        return p


    # Set final condition c_T = grad v^*(x) and t_steps
    def initialize_new(self, calc_final, t_steps):
        self.calc_final = calc_final
        self.t_steps = t_steps

