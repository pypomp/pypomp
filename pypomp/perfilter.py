from .pomp_class import *
from .internal_functions import _perfilter_internal


def perfilter(pomp_object=None, J=50, rinit=None, rprocesses=None, dmeasures=None, theta=None, ys=None, sigmas=None,
              covars=None, thresh=100, key=None):
    """
    An outside function for perturbed particle filtering algorithm. It receives two kinds of input - pomp class
    object or the required arguments of 'perfilter_internal' function, and executes on the object or calls
    'perfilter_internal' directly.

    Args:
        pomp_object (Pomp, optional): An instance of the POMP class. If provided, the function will execute on 
                                      this object to conduct the perturbed particle filtering algorithm.
                                      Defaults to None.
        J (int, optional): _description_. Defaults to 50.
        rinit (function, optional): Simulator for the initial-state distribution. Defaults to None.
        rprocesses (function, optional): Simulator for the process model. Defaults to None.
        dmeasures (function, optional): Density evaluation for the measurement model. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP model. Defaults to None.
        ys (array-like, optional):The measurement array. Defaults to None.
        sigmas (array-like, optional): Perturbed factor. Defaults to None.
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        thresh (float, optional): Threshold value to determine whether to resample particles. 
                                  Defaults to 100.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Raises:
        ValueError: Missing the pomp class object and required arguments for calling 'perfilter_internal' directly

    Returns:
        tuple: A tuple containing:
        - Negative log-likelihood value
        - An updated perturbed array of parameters.
    """
    if pomp_object is not None:
        return pomp_object.perfilter(J=J, sigmas=sigmas, thresh=thresh, key=key)
    elif rinit is not None and rprocesses is not None and dmeasures is not None and theta is not None and ys is not \
            None and sigmas is not None:
        return _perfilter_internal(theta=theta, ys=ys, J=J, sigmas=sigmas, rinit=rinit, rprocesses=rprocesses,
                                  dmeasures=dmeasures, ndim=theta.ndim, covars=covars, thresh=thresh, key=key)
    else:
        raise ValueError("Invalid Arguments Input")
