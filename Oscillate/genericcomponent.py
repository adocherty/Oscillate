"""
Interface for generated code
"""

__all__ = ['generic_real_product_response']

from _generatedrealproductexpansions import *
#from _generatedproductexpansions import *


def generic_real_product_response(a,b, real=False):
    """
    Divert to most appropriate generated product function
    """
    
    Na = a.shape[0]
    Nb = b.shape[0]
    
    if Na==3:
        c = direct_real_product_abc1(a,b)
    elif Na==5:
        c = direct_real_product_abc2(a,b)
    elif Na==7:
        c = direct_real_product_abc3(a,b)
    elif Na==9:
        c = direct_real_product_abc4(a,b)
    elif Na==11:
        c = direct_real_product_abc5(a,b)
    elif Na==13:
        c = direct_real_product_abc6(a,b)
    elif Na==15:
        c = direct_real_product_abc7(a,b)
    elif Na==17:
        c = direct_real_product_abc8(a,b)
    elif Na==19:
        c = direct_real_product_abc9(a,b)
    elif Na==21:
        c = direct_real_product_abc10(a,b)
    else:
        #Fill in array
        raise RuntimeError, "Not yet implemented"

    return c



def generic_triple_product_response(a,b, real=False):
    """
    Divert to most appropriate generated product function
    
    Calculates
    
        y = a^2 b

    with a complex multiscale signal 
    
    """
    Na = a.shape[0]
    Nb = b.shape[0]
    
    if Na==3:
        c = direct_triple_product_abc1(a,b)
    elif Na==5:
        c = direct_triple_product_abc2(a,b)
    elif Na==7:
        c = direct_triple_product_abc3(a,b)
    elif Na==9:
        c = direct_triple_product_abc4(a,b)
    elif Na==11:
        c = direct_triple_product_abc5(a,b)
    elif Na==13:
        c = direct_triple_product_abc6(a,b)
    elif Na==15:
        c = direct_triple_product_abc7(a,b)
    elif Na==17:
        c = direct_triple_product_abc8(a,b)
    elif Na==19:
        c = direct_triple_product_abc9(a,b)
    elif Na==21:
        c = direct_triple_product_abc10(a,b)
    elif Na==23:
        c = direct_triple_product_abc11(a,b)
    elif Na==25:
        c = direct_triple_product_abc12(a,b)
    else:
        #Fill in array
        raise RuntimeError, "Not yet implemented"

    return c
