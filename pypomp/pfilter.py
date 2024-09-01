from .pomp_class import *



def pfilter(pomp_object=None, J=50, rinit=None, rprocess=None, dmeasure=None, theta=None, ys=None, covars=None,
            thresh=100, key=None):
    """_summary_

    Args:
        pomp_object (_type_, optional): _description_. Defaults to None.
        J (int, optional): _description_. Defaults to 50.
        rinit (_type_, optional): _description_. Defaults to None.
        rprocess (_type_, optional): _description_. Defaults to None.
        dmeasure (_type_, optional): _description_. Defaults to None.
        theta (_type_, optional): _description_. Defaults to None.
        ys (_type_, optional): _description_. Defaults to None.
        covars (_type_, optional): _description_. Defaults to None.
        thresh (int, optional): _description_. Defaults to 100.
        key (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if pomp_object is not None:
        return pomp_object.pfilter(J, thresh, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return pfilter_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, thresh, key)
    else:
        raise ValueError("Invalid Arguments Input")
