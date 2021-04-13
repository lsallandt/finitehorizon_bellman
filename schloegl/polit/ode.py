import xerus as xe
import numpy as np
from scipy import linalg as la
import scipy.integrate as integrate


class Ode:
    def __init__(self):
        self.A = np.load('A.npy')
        self.B = np.load('B.npy')
        self.Q_discr = np.load('Q.npy')
        self.R_discr =np.load('R.npy')
        load_me = np.load('save_me.npy')
        self.lambd = load_me[0]
        self.interval_half = load_me[2]
        self.tau = load_me[3]
        self.R_inv =np.load('R_inv.npy')
        self.Q = self.Q_discr/self.tau
        self.R = self.R_discr/self.tau

        self.A_tilde_T = np.linalg.inv(np.eye(self.A.shape[0]) - self.tau * self.A.T) # used for semiimplicit backwards solution

    def step(self, t, x, u):
        return self.step_rk4(t, x, u, self.rhs_schloegl)


    def step_rk4(self, t, x, u, rhs):
        k1 = self.tau * rhs(t, x, u)
        k2 = self.tau * rhs(t+self.tau/2, x + k1/2, u)
        k3 = self.tau * rhs(t+self.tau/2, x + k2/2, u)
        k4 = self.tau * rhs(t+self.tau, x + k3, u)
        return x + 1/6*(k1 + 2*k2 + 2*k3 + k4)


    def rhs_schloegl(self, t, x, u):
        return self.f(t, x) + self.g(t, x) @ u

    def step_adjoint(self, t, p, x):
        return np.dot(self.A_tilde_T, p + self.tau*(self.NL_adjoint(t, p,x) + 2*self.Q@x ))


    def NL_adjoint(self, t, p, x):
        # return 0         # lin
        return 3*x**2*p   # schloegl
        # return (-3*x**2 + 1)*p   # allen-kahn


    def f(self, t, x):
        return self.A@x+self.NL(t, x)


    def NL(self, t, x):
        return x**3


    def g(self, t, x):
        return self.B


    def solver(self, t_points, x, calc_u_fun):
        xshape = x.shape
        num_entries = x.size
        xreshaped = x.reshape(num_entries)
        def rhs_ode(t,x):
            xorig_shape = x.reshape(xshape)
            ret =  (self.f(t, xorig_shape) + self.g(t, xorig_shape) @ calc_u_fun(t, xorig_shape))
            return ret.reshape(num_entries)
        y = integrate.solve_ivp(rhs_ode, [t_points[0], t_points[-1]], xreshaped, t_eval=t_points)
        y_mat = y.y.reshape(xshape+(len(t_points),))
        return y_mat


    def q(self, t, x):
        if len(x.shape) == 1:
            return x.T @ self.Q_discr @ x
        else:
            return np.einsum('il,ik,kl->l', x,self.Q_discr,x)

    def r(self, t, u):
        if len(u.shape) == 1:
            return u.T @ self.R_discr @ u
        else:
            return np.einsum('il,ik,kl->l', u,self.R_discr,u)

    def calc_reward(self, t, x, u):
        return self.q(t, x) + self.r(t, u)

    
    def calc_end_reward(self, t, x):
        if len(x.shape) == 1:
            return x.T @ self.Q_discr @ x/self.tau
        else:
            return np.einsum('il,ik,kl->l', x,self.Q_discr,x)/self.tau

    def calc_end_reward_grad(self, t, x):
        if len(x.shape) == 1:
            return 2 * self.Q_discr @ x/self.tau
        else:
            return 2*np.einsum('ik,kl->il', self.Q_discr,x)/self.tau


    def calc_u(self, t, x, grad):
        if len(x.shape) == 1:
            return -self.R_inv @ np.dot(grad, self.B)/2
        else:
            u_mat = np.tensordot(grad, self.B, axes=((0),(0)))
            return -self.R_inv @ u_mat.T / 2


    def test(self):
        t = 0.
        n = self.A.shape[0]
        m = self.B.shape[1]
        start = np.ones((n,2))
        print('start', start)
        control = np.zeros((m, 2))
        end = self.step(0, start, control)
        print('end', end)
        print('rewards', self.calc_reward(t, start, control), self.calc_reward(t, end, control))
        print('control', self.calc_u(t, start, start))
        return 0



# testOde = Ode()
# testOde.test()
