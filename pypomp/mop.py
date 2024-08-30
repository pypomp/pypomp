from .pomp_class import *



def mop(pomp_object=None, J=50, rinit=None, rprocess=None, dmeasure=None, theta=None, ys=None, covars=None, alpha=0.97,
        key=None):
    if pomp_object is not None:
        return pomp_object.mop(J, alpha, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key)
    else:
        raise ValueError("Invalid Arguments Input")
