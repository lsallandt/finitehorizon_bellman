
import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle
import scipy
import time

class Pol_it:
    def __init__(self, initial_valuefun, ode, polit_params):
        self.v = initial_valuefun
        self.ode = ode
        [self.nos, self.nos_test_set, self.n_sweep, self.rel_val_tol, self.rel_tol, self.max_pol_iter, self.max_iter_Phi, self.horizon] = polit_params
        load_me = np.load('save_me.npy')
        self.interval_half = load_me[2]
        self.samples, self.samples_test = self.build_samples(-self.interval_half, self.interval_half)
        self.data_x = self.v.prepare_data_before_opt(self.samples)
        self.constraints_list = self.construct_constraints_list()
        self.t_vec_p = self.v.t_vec_p
        self.current_time = 0
        self.current_end_time = 0
        self.curr_ind = 0
        self.v.calc_end_reward_grad = self.ode.calc_end_reward_grad


    def build_samples(self, samples_min, samples_max):
        samples_dim = self.ode.A.shape[0]
        samples_mat = np.zeros(shape=(samples_dim, self.nos))
        samples_mat_test_set = np.zeros(shape=(samples_dim, self.nos_test_set))
        np.random.seed(1)
        for i0 in range(self.nos):
            samples_mat[:, i0] = np.random.uniform(samples_min, samples_max, samples_dim)
        for i0 in range(self.nos_test_set):
            samples_mat_test_set[:, i0] = np.random.uniform(samples_min, samples_max, samples_dim)
        return samples_mat, samples_mat_test_set

    def construct_constraints_list(self):
        # return None
        n = self.ode.A.shape[0]
        xvec = np.zeros(shape=(n, n+1))
        P_list = self.v.P_batch(xvec)
        dP_list = self.v.dP_batch(xvec)
        for i0 in range(n):
            P_list[i0][:,i0] = dP_list[i0][:,i0]
        return P_list

    
    def solve_HJB(self, start_num = None):
        if type(start_num) is not int:
            start_num = len(self.t_vec_p) -2
        for i0 in range(start_num, -1, -1):
            self.curr_ind = i0
            if i0 is not start_num:
                print('set V', i0)
                self.v.V[self.curr_ind] = self.v.V[self.curr_ind+1]
            self.current_time = self.v.t_vec_p[i0]
            ind_end = np.minimum(len(self.t_vec_p) - 1, i0+self.horizon)
            self.current_end_time = self.v.t_vec_p[ind_end]
            self.current_t_points = np.linspace(self.current_time, self.current_end_time, int(np.round((self.current_end_time - self.current_time)/self.ode.tau))+1 )
            print('ind_end', ind_end, 't_start, t_end', self.current_time, self.current_end_time, 'current_t_points', self.current_t_points, ((self.current_end_time - self.current_time)/self.ode.tau)+1, np.round((self.current_end_time - self.current_time)/self.ode.tau)+1)
            self.solve_HJB_fixed_time()


    def solve_HJB_fixed_time(self):
        pol_iter = 0 
        rel_diff = 1
        pol_it_counter = 0
        while(rel_diff > self.rel_tol and pol_iter < self.max_pol_iter):
            pol_iter += 1
            V_old = 1*self.v.V[self.curr_ind]
            y_mat , rew_MC = self.build_rhs_batch(self.samples)
            y_mat_test, rew_MC_test = self.build_rhs_batch(self.samples_test)
            data_y = self.v.prepare_data_while_opt(self.current_time, y_mat)
            data = [self.data_x, data_y[1], rew_MC, self.constraints_list]
            params = [self.n_sweep, self.rel_val_tol]
            print('rhs built')
            
            self.v.solve_linear_HJB(data, params)
            pickle.dump(self.v.V[self.curr_ind], open('V_{}'.format(str(self.curr_ind)), 'wb'))
            try:
                rel_diff = xe.frob_norm(self.v.V[self.curr_ind] - V_old) / xe.frob_norm(V_old)
            except:
                rel_diff = 1
            mean_error_test_set = self.calc_mean_error(self.samples_test, y_mat_test, rew_MC_test)
            print('num', pol_it_counter, "rel_diff", rel_diff, 'frob_norm(V)', xe.frob_norm(self.v.V[self.curr_ind]), 'frob_norm(V_old)', xe.frob_norm(V_old), 'avg. gen error', mean_error_test_set)
            # print('mean', mean_error, 'eval_V_batch(V, xmat)[0]', eval_V_batch(V, samples_mat)[0], eval_V_batch(V, samples_mat).shape, samples_mat[:,0], y_mat[:,0])
            pol_it_counter += 1
    
    def calc_mean_error(self, xmat, ymat, rew_MC):
        error = (self.v.eval_V(self.current_time, xmat) - rew_MC)



#         x_vec = np.zeros(shape=(100, n))
#         u_vec = np.zeros(shape=(100, 1))
#         x_vec[0, :] = xmat[:,0]
#         cost = 1/2*self.ode.calc_reward(0, x_vec[0, :], u_vec[0, :])
#         for i0 in range(len(steps)-1):
#             u_vec[i0, :] = _calc_u(x_vec[i0, :], steps[i0])
#             x_vec[i0+1, :] = _step(0, x_vec[i0, :], u_vec[i0, :])
#             add_cost = _calc_cost(0, x_vec[i0, :], u_vec[i0, :])
#             cost += add_cost
#         cost -= add_cost/2
        return np.linalg.norm(error)**2 / rew_MC.size


    def calc_u(self, t, x):
        grad = self.v.calc_grad(t, x)
        return self.ode.calc_u(t, x, grad)

    
    def build_rhs_batch(self, samples):
#         x_mat = samples
#         u_mat = self.calc_u(0, x_mat)
#         rew_MC = 1/2*self.ode.calc_reward(0, x_mat, u_mat)
#         #_rew_MC[i0] += self.ode.calc_reward(x, u)
#         for i1 in range(self.max_iter_Phi):
#             x_mat = self.ode.step(0, x_mat, u_mat)
#             u_mat = self.calc_u(0, x_mat)
#             reward = self.ode.calc_reward(0, x_mat, u_mat)
#             rew_MC += reward
#         y_mat = x_mat
#         rew_MC -= reward/2
#         # rew_MC += eval_V_batch(V_eval, x_mat)
#         return y_mat, rew_MC
        x_mat = samples
        u_mat = self.calc_u(self.current_time, x_mat)
        reward_mat = np.zeros((x_mat.shape[1], len(self.current_t_points)))
        eval_V_mat = []
        reward_mat[:, 0] = self.ode.calc_reward(self.current_time, x_mat, u_mat)
        rew_MC = np.zeros(x_mat.shape[1])
        change_valuefunction_counter = 0
        add_list = []
        curr_ind = 1*self.curr_ind
        for i0 in range(1, len(self.current_t_points)):
            curr_t = self.current_t_points[i0]
            x_mat = self.ode.step(curr_t, x_mat, u_mat)
            u_mat = self.calc_u(self.current_time, x_mat)
            reward_mat[:, i0] = self.ode.calc_reward(curr_t, x_mat, u_mat)
            # if curr_t >= self.v.t_vec_p_p[-1] or curr_t >= self.v.t_vec_p[curr_ind+1] - 1e-8:
            # print('i0', i0, 'curr_t', curr_t)
            if curr_t >= self.v.t_vec_p[curr_ind+1] - 1e-8:
                curr_ind += 1
                add_list.append(i0)
                if curr_t >= self.v.t_vec_p[-1] - 1e-13:
                    eval_V_mat.append(self.ode.calc_end_reward(curr_t, x_mat))
                    # print('add final')
                else:
                    eval_V_mat.append(self.v.eval_V(curr_t, x_mat))
                    # print('add evalV')

        # print('add_list', add_list)
        # print('reward_mat.shape', reward_mat.shape)
        if len(add_list) > 1:
            for i0 in range(len(add_list) - 1, len(add_list)):
                rew_MC += scipy.integrate.trapz(reward_mat[:,:(add_list[i0]+1)], axis=1) + eval_V_mat[i0]
                print((add_list[i0]+2), reward_mat.shape)
            rew_MC /= 1
        else:
            for i0 in range(0, len(add_list)):
                rew_MC += scipy.integrate.trapz(reward_mat[:,:(add_list[i0]+1)], axis=1) + eval_V_mat[i0]
                print((add_list[i0]+2), reward_mat.shape)
            rew_MC /= 1
        # print('rew_MC', rew_MC)
        # print('eval_V_mat', eval_V_mat)
        y_mat = x_mat
        return y_mat, rew_MC

        # rew_MC /= len(add_list)



    def build_rhs_batch_experimental(self, samples):
        # t_points = np.linspace(self.current_time, steps*self.ode.tau, self.current_end_time)
        t_points = np.linspace(self.current_time, self.current_end_time, int(np.round((self.current_end_time - self.current_time)/self.ode.tau))+1 )
        print('curr, end, numsteps, int(numsteps)', self.current_time, self.current_end_time, ((self.current_end_time - self.current_time)/self.ode.tau +1 ), int((self.current_end_time - self.current_time)/self.ode.tau) + 1 )
        print('t_points', t_points)
        sol =  self.ode.solver(t_points, samples, self.calc_u)
        solshapebefore = sol.shape
        solreshaped = sol.reshape((sol.shape[0], -1))
        print('sholshapebefore, solreshaped.shape', solshapebefore, solreshaped.shape)
        u_mat = np.zeros((self.ode.R_discr.shape[0], solshapebefore[1], solshapebefore[2]))
        for i0 in range(u_mat.shape[2]):
            print('t_points[i0]', t_points[i0])
            u_mat[:,:,i0] = self.calc_u(t_points[i0], sol[:,:,i0])
        for i0 in range(980, len(self.v.t_vec)):
            print('frobnorms', i0, xe.frob_norm(self.v.V[i0]))
        print('norm(u_mat)', la.norm(u_mat), la.norm(u_mat, axis=(0,1)))
        u_matreshaped = u_mat.reshape((u_mat.shape[0], -1))
        rewards = self.ode.calc_reward(0, solreshaped, u_matreshaped)
        reward_mat = rewards.reshape((solshapebefore[1], solshapebefore[2]))
        rew_MC = np.trapz(reward_mat, axis=1)
        larger_bool = np.logical_and(sol>=-self.interval_half, sol<= self.interval_half)
        numentries = sol.shape[0]*sol.shape[2]
        for i0 in range(samples.shape[1]):
            if np.count_nonzero(larger_bool[:, i0, :]) != numentries:
                    sol[:,i0, -1] = sol[:, i0, 0]
                    rew_MC[i0] = 0

        larger = np.count_nonzero(np.logical_or(sol<-self.interval_half, sol > self.interval_half))
        if larger > 0:
            print('num entries larger than', self.interval_half,':', larger)
        y_mat = sol[:, :, -1]
        return y_mat, rew_MC
        




