from .pomp_class import *



def fit(pomp_object=None, J=100, Jh=1000, theta=None, rinit=None, rprocess=None, dmeasure=None, rprocesses=None,
        dmeasures=None, ys=None, sigmas=None, sigmas_init=None, covars=None, M=10, a=0.9, method='Newton', itns=20,
        beta=0.9, eta=0.0025, c=0.1,
        max_ls_itn=10, thresh_mif=100, thresh_tr=100, verbose=False, scale=False, ls=False, alpha=0.1, monitor=True,
        mode="IFAD"):
    if pomp_object is not None:
        if mode == "IF2" or mode == "IFAD":
            if sigmas is not None and sigmas_init is not None:
                return pomp_object.fit(sigmas, sigmas_init, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn,
                                       thresh_mif, thresh_tr, verbose, scale, ls, alpha, monitor, mode)
            else:
                raise ValueError("Invalid Argument Input with Missing sigmas or sigmas_init")
        elif mode == "GD":
            return pomp_object.fit(sigmas, sigmas_init, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif,
                                   thresh_tr, verbose, scale, ls, alpha, monitor, mode)
        else:
            raise ValueError("Invalid Mode Input")
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        if mode == "IF2" or mode == "IFAD":
            if rprocesses is not None and dmeasures is not None and sigmas is not None and sigmas_init is not None:
                return fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init,
                                    covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, thresh_tr,
                                    verbose, scale, ls, alpha, monitor, mode)
            else:
                raise ValueError("Invalid Argument Input with Missing workhorse or sigmas")
        elif mode == "GD":
            return fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init,
                                covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, thresh_tr,
                                verbose, scale, ls, alpha, monitor, mode)
        else:
            raise ValueError("Invalid Mode Input")
    else:
        raise ValueError("Invalid Argument Input with Missing Required Argument")
