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

np.random.seed(1)
plt.close('all')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

    
load_num_openloop = 'V_o_'
load_num_polit = 'V_p_'

v_openloop = valuefunction_TT.Valuefunction_TT(load_num_openloop, True)
v_polit = valuefunction_TT.Valuefunction_TT(load_num_polit, True)
testOde = ode.Ode()
# testOde.test()

t_vec = np.load('t_vec.npy')
T = t_vec[-1]
n = v_openloop.V[0].order(); a = -1; b = 1
x = np.linspace(a, b, n)
max_num_initial_values = 2
min_num_initial_values = 0

step_size = .5
step_size_before = 0.02
max_iter = 1e4
grad_tol = 1e-8
print('set last element to zero')
v_openloop.V[-1] = 0*v_openloop.V[-2]
v_polit.V[-1] = 0*v_polit.V[-2]

# diff_vec = []
# for i0 in range(len(v_openloop.V) - 1):
#     diff_vec.append(xe.frob_norm(v_openloop.V[i0+1] - vfun.V[i0]) / xe.frob_norm(vfun.V[i0]))
#     print('frobnorm', xe.frob_norm(v_openloop.V[i0]))
# 
# plt.figure()
# plt.plot(diff_vec)
# # plt.show()

xxxlinspacecondition = (max_num_initial_values == 100)
initial_values = np.zeros(shape=(n, max_num_initial_values-min_num_initial_values))
if min_num_initial_values == 0:
    delta = 2
    initial_values[:,0] = delta * ( x - 1)**2*(x+1)**2
if max_num_initial_values < 5:
    if max_num_initial_values > 1:
        if min_num_initial_values <= 1:
            initial_values[:, 1-min_num_initial_values] = 1.4
        if max_num_initial_values > 2:
            if min_num_initial_values <= 2:
                initial_values[:, 2-min_num_initial_values] = 2*np.exp(-10*(x-0.5)**2) + np.exp(-10*(x+0.5)**2)
            if max_num_initial_values > 3:
                    initial_values[:, 3-min_num_initial_values] = 1.2*np.ones(n)
elif xxxlinspacecondition:
    plot_vec = []
    for i0 in range(max_num_initial_values):
        xi = (2*i0/max_num_initial_values - 0)
        plot_vec.append(xi)
        initial_values[:, i0] = xi
else:
    for _iter in range(0, max_num_initial_values):
        # if _iter < max_num_initial_values/2:
        if _iter < 0:
            sigma = 1
            initial_values[:,_iter] = sigma * np.random.uniform(-3,3,n)
        else:
            deg = np.random.randint(2,20)
            pol_coeff = np.random.randn(deg)
            pol = np.poly1d(pol_coeff)
            initial_values[:,_iter] = pol(x) * ( x - 1)*(x+1)
            initial_values[:,_iter] = 1.9*initial_values[:,_iter] / np.amax(np.abs(initial_values[:,_iter])) 

plt.figure()
plt.plot(x, initial_values)
plt.legend([r'$x_0$', r'$x_1$'])
# plt.show()

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

def calc_u_openloop(t, x):
    return testOde.calc_u(t, x, v_openloop.calc_grad(t, x))

def calc_u_polit(t, x):
    return testOde.calc_u(t, x, v_polit.calc_grad(t, x))


A_lin = np.load('A.npy')# + np.eye(n)
B = np.load('B.npy')
Q = np.load('Q.npy') / tau
R =np.load('R.npy') / tau
BBT = B @ np.linalg.inv(R) @ B.T

from scipy import integrate
def rhs_ric(x, t):
    y = x.reshape((n, n))
    y = A_lin.T @ y + y.T @ A_lin - y.T @ BBT @ y + Q
    return  y.reshape(x.shape)
def calc_riccati_solution():
    # sol = integrate.odeint(rhs_ric, np.zeros(n**2), steps)
    sol =  integrate.odeint(rhs_ric, Q.reshape(n**2), steps)
    ric_list = []
    for i0 in range(m):
        # print(sol.shape)
        ric_list.append(sol[m - i0 - 1, :].reshape((n, n)))
    return ric_list


ric_list = calc_riccati_solution()




def calc_u_lqr(t, x):
    t_point = 0
    while t_vec[t_point] < t:
        t_point += 1
    curr_ric = ric_list[t_point]
    return testOde.calc_u(t, x, 2*curr_ric@x)
    # return _calc_u(calc_grad_riccati(x), x)

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
    cost += testOde.calc_end_reward(0, x_vec[-1,:])
    return x_vec, u_vec, cost



def test_value_function_experimental(_step, calc_u, _x0, calc_cost):
    sol =  testOde.solver(steps, _x0, calc_u)
    solshapebefore = sol.shape
    solreshaped = sol.reshape((sol.shape[0], -1))
    print('sholshapebefore, solreshaped.shape', solshapebefore, solreshaped.shape)
    u_mat = np.zeros((testOde.R_discr.shape[0], solshapebefore[1]))
    for i0 in range(u_mat.shape[1]):
        u_mat[:,i0] = calc_u(steps[i0], sol[:,i0])
    u_matreshaped = u_mat.reshape((u_mat.shape[0], -1))
    rewards = testOde.calc_reward(0, solreshaped, u_matreshaped)
    print('rewards.shape', rewards.shape)
    rew_MC = np.trapz(rewards)
    y_mat = sol
    rew_MC += testOde.calc_end_reward(steps[-1], y_mat[:,-1])
    return y_mat.T, u_mat.T, rew_MC

def calc_opt(x0, u0, calc_cost):
    print('x0, u_openloop shapes', x0.shape, u_openloop.shape)
    x_vec, u_vec = optimize_fun.calc_optimal_control(x0, u_openloop.T)
    cost = 1/2*calc_cost(0, x_vec[:, 0], u_vec[:, 0])
    for i0 in range(len(steps)-1):
        add_cost = calc_cost(steps[i0+1], x_vec[:,i0+1 ], u_vec[:, i0+1])
        cost += add_cost
    cost -= add_cost/2
    cost += testOde.calc_end_reward(0, x_vec[:,-1])
    return x_vec.T, u_vec.T, cost

cost_lqr = 0
cost_openloop = 0
cost_polit = 0
cost_opt = 0
cost_lqr_vec = np.zeros(max_num_initial_values-min_num_initial_values)
cost_openloop_vec = np.zeros(max_num_initial_values-min_num_initial_values)
cost_opt_vec = np.zeros(max_num_initial_values-min_num_initial_values)
cost_polit_vec = np.zeros(max_num_initial_values-min_num_initial_values)
diff_bellman_lqr_vec = np.zeros(max_num_initial_values-min_num_initial_values)
diff_bellman_openloop_vec = np.zeros(max_num_initial_values-min_num_initial_values)
diff_bellman_polit_vec = np.zeros(max_num_initial_values-min_num_initial_values)
u_opt_list = []
for i0 in range(max_num_initial_values - min_num_initial_values):
    print('i0, max', i0, max_num_initial_values - min_num_initial_values)
    x0 = initial_values[:,i0]
    x_openloop, u_openloop, cost_openloop_temp = test_value_function(testOde.step,calc_u_openloop,x0,testOde.calc_reward)
    x_polit, u_polit, cost_polit_temp = test_value_function(testOde.step,calc_u_polit,x0,testOde.calc_reward)
    # x_lqr, u_lqr, cost_lqr = test_value_function(testOde.step,calc_u_lqr,x0,testOde.calc_reward)
    x_lqr, u_lqr, cost_lqr_temp = test_value_function(testOde.step,calc_u_lqr,x0,testOde.calc_reward)
    x_opt, u_opt, cost_opt_temp = calc_opt(x0, u_openloop, testOde.calc_reward)
    # x_opt, u_opt, cost_opt_temp = x_openloop, u_openloop, cost_openloop_temp
    cost_lqr += cost_lqr_temp
    cost_openloop += cost_openloop_temp
    cost_polit += cost_polit_temp
    cost_opt += cost_opt_temp
    # x_opt, u_opt, cost_opt = x_lqr, u_lqr, cost_lqr
    print('cost_openloop', cost_openloop_temp, 'eval_V(x0)',  v_openloop.eval_V(0, x0))
    print('cost_polit', cost_polit_temp, 'eval_V(x0)',  v_polit.eval_V(0, x0))
    print('cost_lqr', cost_lqr_temp)
    print('cost_opt', cost_opt_temp)
    cost_lqr_vec[i0] = cost_lqr_temp
    cost_openloop_vec[i0] = cost_openloop_temp
    cost_polit_vec[i0] = cost_polit_temp
    cost_opt_vec[i0] = cost_opt_temp 
    diff_bellman_openloop_vec[i0] = (v_openloop.eval_V(0, x0) - cost_openloop_temp)**2
    diff_bellman_polit_vec[i0] = (v_polit.eval_V(0, x0) - cost_polit_temp)**2
    diff_bellman_lqr_vec[i0] = (x0 @ Pi_cont @ x0 - cost_lqr_temp)**2
    if i0 < 4:
        plt.figure()
        plt.plot(steps, u_lqr)
        plt.plot(steps, u_openloop)
        plt.plot(steps, u_polit, '--')
        plt.plot(steps, u_opt)
        plt.xlabel("time")
        plt.ylabel('control')
        plt.legend(['lqr', 'open-loop', 'pol. it.', 'optimal'])
        plt.grid(True)
        plt.rcParams.update({'font.size': 14})
#     plt.figure()
#     plt.plot(steps, u_openloop, steps, u_lqr, steps, u_opt)
#     plt.xlabel("time")
#     plt.legend(['HJB', 'LQR', 'optimal'])
#     plt.grid(True)
#     plt.rcParams.update({'font.size': 14})
# plt.show()

mask_lqr = ~np.isnan(cost_lqr_vec)
mask_openloop = ~np.isnan(cost_openloop_vec)
mask_polit = ~np.isnan(cost_polit_vec)
mask_opt = ~np.isnan(cost_opt_vec)
print('percentage stabilized lqr', np.mean(mask_lqr))
print('percentage stabilized openloop', np.mean(mask_openloop))
print('percentage stabilized polit', np.mean(mask_polit))
print('percentage stabilized opt', np.mean(mask_opt))

print("mean cost_lqr:      ", np.mean(cost_lqr_vec[mask_lqr]))
print("mean cost_openloop: ", np.mean(cost_openloop_vec[mask_lqr]))
print("mean cost_polit:    ", np.mean(cost_polit_vec[mask_lqr]))
print("mean cost_opt:      ", np.mean(cost_opt_vec[mask_lqr]))

print("mean bellman_lqr:      ", np.mean(diff_bellman_lqr_vec[mask_lqr]))
print("mean bellman_openloop: ", np.mean(diff_bellman_openloop_vec[mask_lqr]))
print("mean bellman_polit:    ", np.mean(diff_bellman_polit_vec[mask_lqr]))


print('cost_lqr', cost_lqr)
print('cost_openloop', cost_openloop)
print('cost_polit', cost_polit)
print('cost_opt', cost_opt)




# print("quotient", cost_lqr / cost_openloop)

if xxxlinspacecondition:
    plt.figure()
    plt.plot(plot_vec, cost_lqr_vec)
    plt.plot(plot_vec, cost_openloop_vec)
    plt.plot(plot_vec, cost_polit_vec, '--')
    plt.plot(plot_vec, cost_opt_vec)
    plt.rcParams.update({'font.size': 14})
    plt.legend(['lqr', 'open-loop', 'pol. it.', 'optimal'])
    plt.xlabel(r'$x_i$')
    plt.ylabel(r'cost')
    plt.show()


# plt.figure()
# plt.plot( x, initial_values, x, B)
# plt.legend(['B'])
# plt.grid(True)
# plt.rcParams.update({'font.size': 14})
# #xlabel('x');
# plt.show()




#plt.subplot(2,1,1)
if(max_num_initial_values - min_num_initial_values == 2):
    labels = ['$x_0$', '$x_1$']
    
    ind = np.arange(len(labels))
    width = 0.20  # the width of the bars
    
    fig, ax = plt.subplots(2,1)
    rects1 = ax[0].bar(ind - 1.5*width, np.round(cost_lqr_vec, 2), width, label='LQR')
    rects2 = ax[0].bar(ind - 0.5*width, np.round(cost_openloop_vec, 2), width, label='open-loop')
    rects3 = ax[0].bar(ind + 0.5*width, np.round(cost_polit_vec, 2), width, label='pol. it.')
    rects4 = ax[0].bar(ind + 1.5*width, np.round(cost_opt_vec, 2), width, label='opt')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_ylabel('Cost')
    # ax[0].set_ylim(0, 7)
    #ax[0].set_title('Scores by group and gender')
    ax[0].set_xticks(ind)
    ax[0].set_xticklabels(labels)
    #ax[0].legend()
    
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax[0].annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 0.1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    fig.tight_layout()
    
    #plt.show()
    
    #plt.subplot(2,1,2)
    
    #labels = ['$V(x_0)$', '$V(x_1)$']
    #labels = [r'$| v(x_0) - \mathcal J(x_0, \alpha(x_0))|^2$', r'$| v(x_1) - \mathcal J(x_1, \alpha(x_1))|^2$']
    labels = ['$x_0$', '$x_1$']
    #labels = [r'$\alpha$', '$x_1$']
    
    ind = np.arange(len(labels))
    width = 0.25  # the width of the bars
    #fig, ax[1] = plt.subplots()
    rects1 = ax[1].bar(ind - width, np.round(diff_bellman_lqr_vec, 16), width, label='Riccati')
    rects2 = ax[1].bar(ind, np.round(diff_bellman_openloop_vec, 16), width, label='$L_2$')
    rects3 = ax[1].bar(ind + width, np.round(diff_bellman_polit_vec, 16), width, label='$H_1$')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[1].set_ylabel(r'  $| v(x_i) - \mathcal J(x_i, \alpha(x_i))|^2$')
    plt.yscale('log')
    #ax[1].set_title('Scores by group and gender')
    ax[1].set_xticks(ind)
    ax[1].set_xticklabels(labels)
    #ax[1].legend()
    #autolabel(rects1)
    #autolabel(rects2)
    #autolabel(rects3)
    fig.tight_layout()
    # plt.show()

### plot average
#diff_bellman_lqr = la.norm(diff_bellman_lqr_vec, 1)
#diff_bellman_openloop = la.norm(diff_bellman_openloop_vec, 1)
#diff_bellman_polit = la.norm(diff_bellman_polit_vec, 1)
#labels = ['avg.  cost', 'avg.  Bellman error']
#
#ind = np.arange(len(labels))
#width = 0.25  # the width of the bars
#fig, ax = plt.subplots()
#rects1 = ax.bar(ind - width, np.round([cost_lqr / (max_num_initial_values - min_num_initial_values), diff_bellman_lqr / (max_num_initial_values - min_num_initial_values)], 2), width, label='Riccati')
#rects2 = ax.bar(ind, np.round([cost_openloop / (max_num_initial_values - min_num_initial_values), diff_bellman_openloop / (max_num_initial_values - min_num_initial_values)], 2), width, label='$L_2$')
#rects3 = ax.bar(ind + width, np.round([cost_polit / (max_num_initial_values - min_num_initial_values), diff_bellman_polit / (max_num_initial_values - min_num_initial_values)], 2), width, label='$H_1$')
#
## Add some text for labels, title and custom x-axis tick labels, etc.
##ax[0].set_ylabel('Cost')
##ax.set_ylim(0, 7)
##ax[0].set_title('Scores by group and gender')
#ax.set_xticks(ind)
#ax.set_xticklabels(labels)
##ax[0].legend()
#
#
#def autolabel(rects):
#    """Attach a text label above each bar in *rects*, displaying its height."""
#    for rect in rects:
#        height = rect.get_height()
#        ax.annotate('{}'.format(height),
#                    xy=(rect.get_x() + rect.get_width()/2, height),
#                    xytext=(0, 0.1),  # 3 points vertical offset
#                    textcoords="offset points",
#                    ha='center', va='bottom')
#
#
#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#
#fig.tight_layout()
#
#plt.show()


### plot half
try:
    diff_bellman_lqr_1 = la.norm(diff_bellman_lqr_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 2)
except ValueError:
    diff_bellman_lqr_1 = 0
diff_bellman_openloop_1 = la.norm(diff_bellman_openloop_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 2)
diff_bellman_polit_1 = la.norm(diff_bellman_polit_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 2)
try:
    diff_bellman_lqr_2 = la.norm(diff_bellman_lqr_vec[int((max_num_initial_values - min_num_initial_values)/2):], 2)
except ValueError:
    diff_bellman_lqr_2 = 0
diff_bellman_openloop_2 = la.norm(diff_bellman_openloop_vec[int((max_num_initial_values - min_num_initial_values)/2):], 2)
diff_bellman_polit_2 = la.norm(diff_bellman_polit_vec[int((max_num_initial_values - min_num_initial_values)/2):], 2)
try:
    cost_lqr_1 = la.norm(cost_lqr_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 1)
except ValueError:
    cost_lqr_1 = 0
cost_openloop_1 = la.norm(cost_openloop_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 1)
cost_polit_1 = la.norm(cost_polit_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 1)
cost_opt_1 = la.norm(cost_opt_vec[:int((max_num_initial_values - min_num_initial_values)/2)], 1)
try:
    cost_lqr_2 = la.norm(cost_lqr_vec[int((max_num_initial_values - min_num_initial_values)/2):], 1)
except ValueError:
    cost_lqr_2 = 0
cost_openloop_2 = la.norm(cost_openloop_vec[int((max_num_initial_values - min_num_initial_values)/2):], 1)
cost_polit_2 = la.norm(cost_polit_vec[int((max_num_initial_values - min_num_initial_values)/2):], 1)
cost_opt_2 = la.norm(cost_opt_vec[int((max_num_initial_values - min_num_initial_values)/2):], 1)
avg_cost_lqr_vec = [cost_lqr_1 / int((max_num_initial_values - min_num_initial_values)/2), cost_lqr_2 / int((max_num_initial_values - min_num_initial_values)/2)]
avg_cost_openloop_vec = [cost_openloop_1 / int((max_num_initial_values - min_num_initial_values)/2), cost_openloop_2 / int((max_num_initial_values - min_num_initial_values)/2)]
avg_cost_polit_vec = [cost_polit_1 / int((max_num_initial_values - min_num_initial_values)/2), cost_polit_2 / int((max_num_initial_values - min_num_initial_values)/2)]
avg_cost_opt_vec = [cost_opt_1 / int((max_num_initial_values - min_num_initial_values)/2), cost_opt_2 / int((max_num_initial_values - min_num_initial_values)/2)]
avg_diff_bellman_lqr = [diff_bellman_lqr_1 / int((max_num_initial_values - min_num_initial_values)/2), diff_bellman_lqr_2 / int((max_num_initial_values - min_num_initial_values)/2)]
avg_diff_bellman_openloop = [diff_bellman_openloop_1 / int((max_num_initial_values - min_num_initial_values)/2), diff_bellman_openloop_2 / int((max_num_initial_values - min_num_initial_values)/2)]
avg_diff_bellman_polit = [diff_bellman_polit_1 / int((max_num_initial_values - min_num_initial_values)/2), diff_bellman_polit_2 / int((max_num_initial_values - min_num_initial_values)/2)]

labels = [r'$x_0 \sim \mathcal U(-2,2)$', '$x_1 \sim $ polynomial distribution']

ind = np.arange(len(labels))
width = 0.20  # the width of the bars

fig, ax = plt.subplots(2,1)
rects1 = ax[0].bar(ind - 1.5*width, np.round(avg_cost_lqr_vec, 2), width, label='LQR')
rects2 = ax[0].bar(ind - 0.5*width, np.round(avg_cost_openloop_vec, 2), width, label='open-loop')
rects3 = ax[0].bar(ind + 0.5*width, np.round(avg_cost_polit_vec, 2), width, label='pol. it.')
rects4 = ax[0].bar(ind + 1.5*width, np.round(avg_cost_opt_vec, 2), width, label='opt')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('avg. Cost')
#ax[0].set_ylim(0, 7)
#ax[0].set_title('Scores by group and gender')
ax[0].set_xticks(ind)
ax[0].set_xticklabels(labels)
#ax[0].legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax[0].annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 0.1),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

#plt.show()

#plt.subplot(2,1,2)

#labels = ['$V(x_0)$', '$V(x_1)$']
labels = [r'$x_0 \sim \mathcal U(-2,2)$', '$x_1 \sim $ polynomial distribution']
# labels = [r' avg. $| v(x_0) - \mathcal J(x_0, \alpha(x_0))|^2$', r'$| v(x_1) - \mathcal J(x_1, \alpha(x_1))|^2$']
#labels = [r'$\alpha$', '$x_1$']

ind = np.arange(len(labels))
width = 0.25  # the width of the bars
#fig, ax[1] = plt.subplots()
rects1 = ax[1].bar(ind - width, np.round(avg_diff_bellman_lqr, 16), width, label='Riccati')
rects2 = ax[1].bar(ind, np.round(avg_diff_bellman_openloop, 16), width, label='$L_2$')
rects3 = ax[1].bar(ind + width, np.round(avg_diff_bellman_polit, 16), width, label='$H_1$')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel(r' avg. $| v(x_i) - \mathcal J(x_i, \alpha(x_i))|^2$')
plt.yscale('log')
#ax[1].set_title('Scores by group and gender')
ax[1].set_xticks(ind)
ax[1].set_xticklabels(labels)
#ax[1].legend()
#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
fig.tight_layout()
plt.show()



#ind = np.arange(3)
#plt.figure()
#plt.subplot(2,1,1)
#plt.bar(ind,[cost_lqr_vec[0], cost_openloop_vec[0], cost_polit_vec[0]], 0.35)
#plt.xticks(ind, ('Riccati', 'L2', 'H1'))
#plt.ylim([0.9*min([cost_lqr_vec[0], cost_openloop_vec[0], cost_polit_vec[0]]), 1.1*max([cost_lqr_vec[0], cost_openloop_vec[0], cost_polit_vec[0]])])
#plt.subplot(2,1,2)
#plt.bar(ind,[cost_lqr_vec[1], cost_openloop_vec[1], cost_polit_vec[1]], 0.35)
#plt.xticks(ind, ('Riccati', 'L2', 'H1'))
#plt.ylim([0.9*min([cost_lqr_vec[1], cost_openloop_vec[1], cost_polit_vec[1]]), 1.1*max([cost_lqr_vec[1], cost_openloop_vec[1], cost_polit_vec[1]])])
#plt.figure()
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.plot(x, initial_values)
#plt.legend([r'$x_3$', r'$x_4$', r'$x_$'])
#plt.grid(True)
#plt.rcParams.update({'font.size': 14})
##xlabel('x');
#plt.show()
