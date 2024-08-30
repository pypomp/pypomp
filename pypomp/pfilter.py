from .pomp_class import *



def pfilter(pomp_object=None, J=50, rinit=None, rprocess=None, dmeasure=None, theta=None, ys=None, covars=None,
            thresh=100, key=None):
    if pomp_object is not None:
        return pomp_object.pfilter(J, thresh, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key)
    else:
        raise ValueError("Invalid Arguments Input")
