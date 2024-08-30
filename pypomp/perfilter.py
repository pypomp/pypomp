from .pomp_class import *



def perfilter(pomp_object=None, J=50, rinit=None, rprocesses=None, dmeasures=None, theta=None, ys=None, sigmas=None,
              covars=None, a=0.95, thresh=100, key=None):
    if pomp_object is not None:
        return pomp_object.perfilter(J=J, sigmas=sigmas, a=a, thresh=thresh, key=key)
    elif rinit is not None and rprocesses is not None and dmeasures is not None and theta is not None and ys is not None and sigmas is not None:
        return perfilter_internal(theta=theta, ys=ys, J=J, sigmas=sigmas, rinit=rinit, rprocesses=rprocesses,
                                  dmeasures=dmeasures, ndim=theta.ndim, covars=covars, a=a, thresh=thresh, key=key)
    else:
        raise ValueError("Invalid Arguments Input")
