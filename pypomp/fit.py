from .pomp_class import *
from .internal_functions import _fit_internal


def fit(pomp_object=None, J=100, Jh=1000, theta=None, rinit=None, rprocess=None, dmeasure=None, rprocesses=None,
        dmeasures=None, ys=None, sigmas=None, sigmas_init=None, covars=None, M=10, a=0.9, method='Newton', itns=20,
        beta=0.9, eta=0.0025, c=0.1,
        max_ls_itn=10, thresh_mif=100, thresh_tr=100, verbose=False, scale=False, ls=False, alpha=0.1, monitor=True,
        mode="IFAD"):
    """
    An outside function controlling which fit operation to use for likelihood maximization algorithm of POMP model 
    by executing on a pomp class object or by calling function 'fit_internal' directly. 


    Args:
        pomp_object (Pomp, optional): An instance of the POMP class. If provided, the function will execute on 
                                      this object to conduct the fit operation. Defaults to None.
        J (int, optional):  J (int, optional): The number of particles in iterated filtering and the number of
                            particles in the MOP objective for obtaining the gradient in gradient optimization. 
                            Defaults to 100.
        Jh (int, optional): The number of particles in the MOP objective for obtaining the Hessian matrix. 
                            Defaults to 1000.
        theta (array-like, optional): Initial parameters involved in the POMP model. Defaults to None.
        rinit (function, optional): Simulator for the initial-state distribution. Defaults to None.
        rprocess (function, optional): Simulator for the process model. Defaults to None.
        dmeasure (function, optional): Density evaluation for the measurement model. Defaults to None.
        rprocesses (function, optional): Simulator for the perturbed process model. Defaults to None.
        dmeasures (function, optional): Density evaluation for the perturbed measurement model. Defaults to None.
        ys (array-like, optional): The measurement array. Defaults to None.
        sigmas (float, optional): Perturbed factor. Defaults to None.
        sigmas_init (float, optional): Initial perturbed factor. Defaults to None.
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        M (int, optional): Maximum algorithm iteration for iterated filtering. Defaults to 10.
        a (float, optional): Decay factor for sigmas. Defaults to 0.9.
        method (str, optional): The gradient optimization method to use, including Newton method, weighted Newton 
                                method BFGS method, gradient descent. Defaults to 'Newton'.
        itns (int, optional): Maximum iteration for the gradient optimization. Defaults to 20.
        beta (float, optional): Initial step size. Defaults to 0.9.
        eta (float, optional): Initial step size. Defaults to 0.0025.
        c (float, optional): The user-defined Armijo condition constant. Defaults to 0.1.
        max_ls_itn (int, optional): The maximum number of iterations for the line search algorithm. Defaults to 10.
        thresh_mif (float, optional): Threshold value to determine whether to resample particles in iterated filtering.
                                      Defaults to 100.
        thresh_tr (float, optional): Threshold value to determine whether to resample particles in gradient
                                     optimization. Defaults to 100.
        verbose (bool, optional): Boolean flag controlling whether to print out the log-likelihood and parameter
                                  information. Defaults to False.
        scale (bool, optional): Boolean flag controlling normalizing the direction or not. Defaults to False.
        ls (bool, optional): Boolean flag controlling using the line search or not. Defaults to False.
        alpha (float, optional): Discount factor. Defaults to 0.1.
        monitor (bool, optional): Boolean flag controlling whether to monitor the log-likelihood value. Defaults to True.
        mode (str, optional):The optimization algorithm to use, including 'IF2', 'GD', and 'IFAD'. Defaults to "IFAD".

    Raises:
        ValueError: Missing the required arguments 'sigmas' or 'sigmas_init' in 'IF2' or 'IFAD' when executing on the 
                    pomp class object
        ValueError: Invalid mode input when executing on the pomp class object
        ValueError: Missing the required arguments, workhorses, 'sigmas' or 'sigmas_init' in 'IF2' or 'IFAD' when
                    calling the function 'fit_internal' directly.
        ValueError: Invalid mode input when calling 'fit_internal' function directly.
        ValueError: Missing the pomp class object and required arguments for calling 'fit_internal' directly

    Returns:
        tuple: A tuple containing:
        - An array of negative log-likelihood through the iterations
        - An array of parameters through the iterations
    """
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
                return _fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init,
                                    covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, thresh_tr,
                                    verbose, scale, ls, alpha, monitor, mode)
            else:
                raise ValueError("Invalid Argument Input with Missing workhorse or sigmas")
        elif mode == "GD":
            return _fit_internal(theta, ys, rinit, rprocess, dmeasure, rprocesses, dmeasures, sigmas, sigmas_init,
                                covars, M, a, J, Jh, method, itns, beta, eta, c, max_ls_itn, thresh_mif, thresh_tr,
                                verbose, scale, ls, alpha, monitor, mode)
        else:
            raise ValueError("Invalid Mode Input")
    else:
        raise ValueError("Invalid Argument Input with Missing Required Argument")
