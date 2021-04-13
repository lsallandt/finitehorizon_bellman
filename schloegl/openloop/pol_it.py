
import xerus as xe
import numpy as np
from scipy import linalg as la
import pickle
import scipy
import time
import joblib as jl




class Pol_it:
    def __init__(self, initial_valuefun, ode, optimize_fun, polit_params):
        self.v = initial_valuefun
        self.ode = ode
        [self.nos, self.nos_test_set, self.n_sweep, self.rel_val_tol, self.rel_tol, self.max_pol_iter, self.max_iter_Phi, self.horizon] = polit_params
        load_me = np.load('save_me.npy')
        self.interval_half = load_me[2]
        self.samples, self.samples_test = self.build_samples(-self.interval_half, self.interval_half)
        self.data_x = self.v.prepare_data_before_opt(self.samples)
        self.constraints_list = self.construct_constraints_list()
        self.t_vec = self.v.t_vec
        self.current_time = 0
        self.current_end_time = 0
        self.curr_ind = 0
        self.optimize_fun = optimize_fun
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
            start_num = len(self.t_vec) -2
        for i0 in range(start_num, -1, -1):
            self.curr_ind = i0
            if i0 is not start_num:
                print('set V', i0)
                self.v.V[self.curr_ind] = self.v.V[self.curr_ind+1]
            self.current_time = self.v.t_vec[i0]
            ind_end = np.minimum(len(self.t_vec) - 1, i0+self.horizon)
            self.current_end_time = self.v.t_vec[ind_end]
            self.current_t_points = np.linspace(self.current_time, self.current_end_time, int(np.round((self.current_end_time - self.current_time)/self.ode.tau))+1 )
            if i0 == start_num:
                self.u_mat = np.zeros((self.ode.R.shape[0], self.samples.shape[1], len(self.current_t_points)))
            elif self.u_mat.shape[2] != len(self.current_t_points):
                self.u_mat = np.dstack((self.u_mat, np.zeros(self.u_mat[:,:,0].shape)))
            if self.current_end_time != self.t_vec[-1]:
                self.optimize_fun.initialize_new(self.v.calc_grad, self.current_t_points)
                print('set calc_grad')
            else:
                self.optimize_fun.initialize_new(self.ode.calc_end_reward_grad, self.current_t_points)
                print('set calc_end_reward_grad')
            print('ind_end', ind_end, 't_start, t_end', self.current_time, self.current_end_time, 'current_t_points', self.current_t_points, ((self.current_end_time - self.current_time)/self.ode.tau)+1, np.round((self.current_end_time - self.current_time)/self.ode.tau)+1)
            self.solve_HJB_fixed_time()


    def solve_HJB_fixed_time(self):
        pol_iter = 0 
        rel_diff = 1
        pol_it_counter = 0
        while(rel_diff > self.rel_tol and pol_iter < self.max_pol_iter):
            pol_iter += 1
            V_old = 1*self.v.V[self.curr_ind]
            t00 = time.time()
            t01 = time.perf_counter()
            y_mat , rew_MC = self.build_rhs_batch_experimental(self.samples)
            t10 = time.time()
            t11 = time.perf_counter()
            print('The calculations took:, time(), perf_counter()', t10 - t00, t11 - t01 )
            y_mat_test, rew_MC_test = self.build_rhs_batch_experimental_test(self.samples_test)
            data_y = self.v.prepare_data_while_opt(self.current_time, y_mat)
            data = [self.data_x, data_y[1], rew_MC, self.constraints_list]
            params = [self.n_sweep, self.rel_val_tol]
            print('rhs built')
            
            self.v.solve_linear_HJB(data, params)
            # print('self.v.eval_V(0, samples[:,0]), rew_MC[0]', self.v.eval_V(self.current_time, self.samples[:,0]), rew_MC[0])
            # print('self.v.eval_V(0, samples[:,1]), rew_MC[1]', self.v.eval_V(self.current_time, self.samples[:,1]), rew_MC[1])
            # print('self.v.eval_V(0, samples[:,2]), rew_MC[2]', self.v.eval_V(self.current_time, self.samples[:,2]), rew_MC[2])
            # print('self.v.eval_V(0, samples[:,3]), rew_MC[3]', self.v.eval_V(self.current_time, self.samples[:,3]), rew_MC[3])
            # print('self.v.eval_V(0, samples[:,4]), rew_MC[4]', self.v.eval_V(self.current_time, self.samples[:,4]), rew_MC[4])
            pickle.dump(self.v.V[self.curr_ind], open('V_{}'.format(str(self.curr_ind)), 'wb'))
            try:
                rel_diff = xe.frob_norm(self.v.V[self.curr_ind] - V_old) / xe.frob_norm(V_old)
            except:
                rel_diff = 1
            mean_error_test_set = self.calc_mean_error(self.samples, y_mat, rew_MC)
            print('num', pol_it_counter, "rel_diff", rel_diff, 'frob_norm(V)', xe.frob_norm(self.v.V[self.curr_ind]), 'frob_norm(V_old)', xe.frob_norm(V_old), 'avg. gen error', mean_error_test_set)
            # print('mean', mean_error, 'eval_V_batch(V, xmat)[0]', eval_V_batch(V, samples_mat)[0], eval_V_batch(V, samples_mat).shape, samples_mat[:,0], y_mat[:,0])
            pol_it_counter += 1
    
    def calc_mean_error(self, xmat, ymat, rew_MC):
        if self.current_end_time == self.t_vec[-1]:
            error = (self.v.eval_V(self.current_time, xmat) - rew_MC)
            # error = (self.v.eval_V(self.current_time, xmat) - self.ode.calc_end_reward(self.current_end_time, ymat) - rew_MC)
        else:
            error = (self.v.eval_V(self.current_time, xmat) - rew_MC)
        return np.linalg.norm(error)**2 / rew_MC.size


    def calc_u(self, t, x):
        grad = self.v.calc_grad(t, x)
        return self.ode.calc_u(t, x, grad)

    
    def build_rhs_batch_experimental(self, samples):
        # print('curr, end, numsteps', self.current_time, self.current_end_time, len(self.current_t_points))
        x_mat = np.zeros(samples.shape+(len(self.current_t_points),))
        # print('x_mat.shape', x_mat.shape, self.u_mat.shape)
        # x_mat = jl.Parallel(n_jobs=2)(i0 for i0 in range(samples.shape[1]))


        ret = jl.Parallel(n_jobs=-1)(jl.delayed(self.optimize_fun.calc_optimal_control)(samples[:,i0], self.u_mat[:,i0,:]) for i0 in range(samples.shape[1]))
        for i0 in range(samples.shape[1]):
            x_mat[:,i0,:], self.u_mat[:,i0,:] = ret[i0]
            # x_mat[:,i0,:], self.u_mat[:,i0,:] = self.optimize_fun.calc_optimal_control(samples[:,i0], self.u_mat[:,i0,:])
            # if i0 % 1000 == 0:
                # print('i0, curr_gradient_norm', i0, samples.shape[1])


        # for i0 in range(980, len(self.v.t_vec)):
             # print('frobnorms', i0, xe.frob_norm(self.v.V[i0]))
        # print('norm(self.u_mat)', la.norm(self.u_mat), la.norm(self.u_mat, axis=(0,1)))
        # print('x_mat[:,0,:]', la.norm(x_mat[:,0,:]))
        # print('u_mat[:,0,:]', (self.u_mat[:,0,:]))
        # print('u_mat.shape', self.u_mat.shape)
        x_mat_shapebefore = x_mat.shape
        x_mat_reshaped = x_mat.reshape((x_mat.shape[0], -1))
        u_matreshaped = self.u_mat.reshape((self.u_mat.shape[0], -1))
        rewards = self.ode.calc_reward(0, x_mat_reshaped, u_matreshaped)
        reward_mat = rewards.reshape((x_mat_shapebefore[1], x_mat_shapebefore[2]))
        rew_MC = np.trapz(reward_mat, axis=1)
        larger_bool = np.logical_and(x_mat>=-self.interval_half, x_mat<= self.interval_half)
        numentries = x_mat.shape[0]*x_mat.shape[2]
        for i0 in range(samples.shape[1]):
            if np.count_nonzero(larger_bool[:, i0, :]) != numentries:
                    x_mat[:,i0, -1] = x_mat[:, i0, 0]
                    rew_MC[i0] = 0

        larger = np.count_nonzero(np.logical_or(x_mat<-self.interval_half, x_mat > self.interval_half))
        if larger > 0:
            print('num entries larger than', self.interval_half,':', larger)
        y_mat = x_mat[:, :, -1]
        # print('xmat,ymat.shape', x_mat.shape, y_mat.shape)
        if self.current_end_time == self.t_vec[-1]:
            rew_MC += self.ode.calc_end_reward(self.current_end_time, y_mat)
        else:
            rew_MC += self.v.eval_V(self.current_end_time, y_mat)
        return y_mat, rew_MC
        



    def build_rhs_batch_experimental_test(self, samples):
        # print('curr, end, numsteps', self.current_time, self.current_end_time, len(self.current_t_points))
        x_mat = np.zeros(samples.shape+(len(self.current_t_points),))
        u_mat = np.zeros((self.ode.R.shape[0], samples.shape[1], len(self.current_t_points)))
        # print('x_mat.shape', x_mat.shape, u_mat.shape)

        for i0 in range(samples.shape[1]):
            x_mat[:,i0,:], u_mat[:,i0,:] = self.optimize_fun.calc_optimal_control(samples[:,i0], u_mat[:,i0,:])
            # if i0 % 1000 == 0:
                # print('i0, curr_gradient_norm', i0, samples.shape[1])


        # for i0 in range(980, len(self.v.t_vec)):
            # print('frobnorms', i0, xe.frob_norm(self.v.V[i0]))
        # print('norm(u_mat)', la.norm(u_mat), la.norm(u_mat, axis=(0,1)))
        x_mat_shapebefore = x_mat.shape
        x_mat_reshaped = x_mat.reshape((x_mat.shape[0], -1))
        u_matreshaped = u_mat.reshape((u_mat.shape[0], -1))
        rewards = self.ode.calc_reward(0, x_mat_reshaped, u_matreshaped)
        reward_mat = rewards.reshape((x_mat_shapebefore[1], x_mat_shapebefore[2]))
        rew_MC = np.trapz(reward_mat, axis=1)
        larger_bool = np.logical_and(x_mat>=-self.interval_half, x_mat<= self.interval_half)
        numentries = x_mat.shape[0]*x_mat.shape[2]
        for i0 in range(samples.shape[1]):
            if np.count_nonzero(larger_bool[:, i0, :]) != numentries:
                    x_mat[:,i0, -1] = x_mat[:, i0, 0]
                    rew_MC[i0] = 0

        larger = np.count_nonzero(np.logical_or(x_mat<-self.interval_half, x_mat > self.interval_half))
        if larger > 0:
            print('num entries larger than', self.interval_half,':', larger)
        y_mat = x_mat[:, :, -1]
        if self.current_end_time == self.t_vec[-1]:
            # print('add endrewrad')
            rew_MC += self.ode.calc_end_reward(self.current_end_time, y_mat)
        else:
            # print('self.current_end_time', self.current_end_time)
            # print('rew_MC before add', rew_MC)
            # print('eval_V', self.v.eval_V(self.current_end_time, y_mat))
            rew_MC += self.v.eval_V(self.current_end_time, y_mat)
        return y_mat, rew_MC

