

import numpy as np

def discrete_entr(pX):
    """
    Calculates the entropy of the distribution pX, i.e., 
    sum(-pX*log2(pX))

    """
    pX = np.asarray(pX, dtype=float)

    if abs(sum(pX) - 1) >= 1e-6:
        raise ValueError('pX must be a valid probability distribution, i.e. sum(pX) == 1')
    
    idx = pX > 0
    H = np.sum(-pX[idx] * np.log2(pX[idx]))

    return H

def discrete_cross_entr(pX, pY):
    """
    Calculates the cross-entropy of the distribution pX and pY, i.e., 
    sum(-pX*log2(py))

    """
    pX = np.asarray(pX, dtype=float)
    pY = np.asarray(pY, dtype=float)

    if pX.size != pY.size:
        raise ValueError('pX must be of same size of pY')

    if abs(sum(pX) - 1) >= 1e-6:
        raise ValueError('pX must be a valid probability distribution, i.e. sum(pX) == 1')
    else:
        pX = pX/np.sum(pX)

    if abs(sum(pY) - 1) >= 1e-6:
        raise ValueError('pY must be a valid probability distribution, i.e. sum(pX) == 1')
    else:
        pY = pY/np.sum(pY)

    idx = pX > 0
    H = np.sum(-pX[idx] * np.log2(pY[idx]))

    return H

def discrete_kl_div(pX, pY):
    """
    Calculates the KL divergence D(pX||pY) of the distribution pX and pY, i.e., 
    sum(-pX*log2(pX/pY))

    """
    KL = -discrete_entr(pX) + discrete_cross_entr(pX, pY)
    return KL