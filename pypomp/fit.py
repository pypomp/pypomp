from .pomp_class import *



def fit(pomp_object=None, J=100, Jh=1000, theta=None, rinit=None, rprocess=None, dmeasure=None, rprocesses=None,
        dmeasures=None, ys=None, sigmas=None, sigmas_init=None, covars=None, M=10, a=0.9, method='Newton', itns=20,
        beta=0.9, eta=0.0025, c=0.1,
        max_ls_itn=10, thresh_mif=100, thresh_tr=100, verbose=False, scale=False, ls=False, alpha=0.1, monitor=True,
        mode="IFAD"):
    """_summary_

    Args:
        pomp_object (_type_, optional): _description_. Defaults to None.
        J (int, optional): _description_. Defaults to 100.
        Jh (int, optional): _description_. Defaults to 1000.
        theta (_type_, optional): _description_. Defaults to None.
        rinit (_type_, optional): _description_. Defaults to None.
        rprocess (_type_, optional): _description_. Defaults to None.
        dmeasure (_type_, optional): _description_. Defaults to None.
        rprocesses (_type_, optional): _description_. Defaults to None.
        dmeasures (_type_, optional): _description_. Defaults to None.
        ys (_type_, optional): _description_. Defaults to None.
        sigmas (_type_, optional): _description_. Defaults to None.
        sigmas_init (_type_, optional): _description_. Defaults to None.
        covars (_type_, optional): _description_. Defaults to None.
        M (int, optional): _description_. Defaults to 10.
        a (float, optional): _description_. Defaults to 0.9.
        method (str, optional): _description_. Defaults to 'Newton'.
        itns (int, optional): _description_. Defaults to 20.
        beta (float, optional): _description_. Defaults to 0.9.
        eta (float, optional): _description_. Defaults to 0.0025.
        c (float, optional): _description_. Defaults to 0.1.
        max_ls_itn (int, optional): _description_. Defaults to 10.
        thresh_mif (int, optional): _description_. Defaults to 100.
        thresh_tr (int, optional): _description_. Defaults to 100.
        verbose (bool, optional): _description_. Defaults to False.
        scale (bool, optional): _description_. Defaults to False.
        ls (bool, optional): _description_. Defaults to False.
        alpha (float, optional): _description_. Defaults to 0.1.
        monitor (bool, optional): _description_. Defaults to True.
        mode (str, optional): _description_. Defaults to "IFAD".

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
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
