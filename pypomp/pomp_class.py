from .internal_functions import *



class Pomp:
    MONITORS = 1

    def __init__(self, rinit, rproc, dmeas, ys, theta, covars=None):
        if rinit is None:
            raise TypeError("rinit cannot be None")
        if rproc is None:
            raise TypeError("rproc cannot be None")
        if dmeas is None:
            raise TypeError("dmeas cannot be None")
        if ys is None:
            raise TypeError("ys cannot be None")
        if theta is None:
            raise TypeError("theta cannot be None")

        self.rinit = rinit
        self.rproc = rproc
        self.dmeas = dmeas
        self.ys = ys
        self.theta = theta
        self.covars = covars
        self.rprocess = jax.vmap(self.rproc, (0, None, 0, None))
        self.rprocesses = jax.vmap(rproc, (0, 0, 0, None))
        self.dmeasure = jax.vmap(self.dmeas, (None, 0, None))
        self.dmeasures = jax.vmap(self.dmeas, (None, 0, 0))

    # def rinits(self, thetas, J, covars):
    #     return rinits_internal(self.rinit, thetas, J, covars)

    def mop(self, J, alpha=0.97, key=None):
        return mop_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha, key)

    def mop_mean(self, J, alpha=0.97, key=None):
        return mop_internal_mean(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, alpha,
                                 key)

    def pfilter(self, J, thresh=100, key=None):
        return pfilter_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars, thresh,
                                key)

    def pfilter_mean(self, J, thresh=100, key=None):
        return pfilter_internal_mean(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                     thresh, key)

    def perfilter(self, J, sigmas, a=0.9, thresh=100, key=None):

        return perfilter_internal(self.theta, self.ys, J, sigmas, self.rinit, self.rprocesses, self.dmeasures,
                                  ndim=self.theta.ndim, covars=self.covars, a=a, thresh=thresh, key=key)

    def perfilter_mean(self, J, sigmas, a=0.9, thresh=100, key=None):

        return perfilter_internal_mean(self.theta, self.ys, J, sigmas, self.rinit, self.rprocesses, self.dmeasures,
                                       ndim=self.theta.ndim, covars=self.covars, a=a, thresh=thresh, key=key)

    def pfilter_pf(self, J, thresh=100, key=None):
        return pfilter_pf_internal(self.theta, self.ys, J, self.rinit, self.rprocess, self.dmeasure, self.covars,
                                   thresh, key)

    def mif(self, sigmas, sigmas_init, M=10, a=0.9, J=100, thresh=100, monitor=False, verbose=False):

        return mif_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses,
                            self.dmeasures,
                            sigmas, sigmas_init, self.covars, M, a, J, thresh, monitor, verbose)

    def train(self, theta_ests, J=5000, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, max_ls_itn=10,
              thresh=100, verbose=False, scale=False, ls=False, alpha=1):

        return train_internal(theta_ests, self.ys, self.rinit, self.rprocess, self.dmeasure, self.covars, J, Jh, method,
                              itns, beta, eta, c, max_ls_itn, thresh, verbose, scale, ls, alpha)

    def fit(self, sigmas=None, sigmas_init=None, M=10, a=0.9,
            J=100, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1,
            max_ls_itn=10, thresh_mif=100, thresh_tr=100, verbose=False, scale=False, ls=False, alpha=0.1, monitor=True,
            mode="IFAD"):

        return fit_internal(self.theta, self.ys, self.rinit, self.rprocess, self.dmeasure, self.rprocesses,
                            self.dmeasures, sigmas, sigmas_init, self.covars, M=M, a=a, J=J, Jh=Jh, method=method,
                            itns=itns, beta=beta, eta=eta, c=c, max_ls_itn=max_ls_itn, thresh_mif=thresh_mif,
                            thresh_tr=thresh_tr, verbose=verbose, scale=scale, ls=ls, alpha=alpha, monitor=monitor,
                            mode=mode)
