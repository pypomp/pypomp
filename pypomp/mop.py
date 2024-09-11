from .pomp_class import *
from .internal_functions import _mop_internal


def mop(pomp_object=None, J=50, rinit=None, rprocess=None, dmeasure=None, theta=None, ys=None, covars=None, alpha=0.97,
        key=None):
    """
    An outside function for MOP algorithm. It receives two kinds of input - pomp class object or the required arguments
    of 'mop_internal' function, and executes on the object or calls 'mop_internal' directly.
   
    Args:
        pomp_object (Pomp, optional): An instance of the POMP class. If provided, the function will execute on 
                                      this object to conduct the MOP algorithm. Defaults to None.
        J (int, optional): The number of particles. Defaults to 50.
        rinit (function, optional): Simulator for the initial-state distribution. Defaults to None.
        rprocess (function, optional): Simulator for the process model. Defaults to None.
        dmeasure (function, optional): Density evaluation for the measurement model. Defaults to None.
        theta (array-like, optional): Parameters involved in the POMP model. Defaults to None.
        ys (array-like, optional): The measurement array. Defaults to None.
        covars (array-like, optional): Covariates or None if not applicable. Defaults to None.
        alpha (float, optional): Discount factor. Defaults to 0.97.
        key (jax.random.PRNGKey, optional): The random key. Defaults to None.

    Raises:
        ValueError: Missing the pomp class object and required arguments for calling 'mop_internal' directly

    Returns:
        float: Negative log-likelihood value
    """
    if pomp_object is not None:
        return pomp_object.mop(J, alpha, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return _mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key)
    else:
        raise ValueError("Invalid Arguments Input")
