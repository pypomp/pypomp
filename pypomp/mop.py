from .pomp_class import *



def mop(pomp_object=None, J=50, rinit=None, rprocess=None, dmeasure=None, theta=None, ys=None, covars=None, alpha=0.97,
        key=None):
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
        alpha (float, optional): _description_. Defaults to 0.97.
        key (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if pomp_object is not None:
        return pomp_object.mop(J, alpha, key)
    elif rinit is not None and rprocess is not None and dmeasure is not None and theta is not None and ys is not None:
        return mop_internal(theta, ys, J, rinit, rprocess, dmeasure, covars, alpha, key)
    else:
        raise ValueError("Invalid Arguments Input")
