from modules.constants import *
from modules.miedema_model import omega_calc
from scipy.optimize import fsolve
from modules.energy_contributions import Eelastic,surfEnergy

def LangmuirMcLean(cov,mA,mB,mC,dictInfo,T):
    """
    Compute the surface composition using the Langmuir-McLean equation
    
    Parameters
    ----------
    cov : adsorbant coverage
    mA,mB,mC : elements
    dictInfo : dictionary with materials information
    T : temperature
    q : [guess surface mA, guess surface mB, segregationEnthalpyGuess]
    
    Returns
    -------
    F: [final surface composition, final Hseg]
    
    """
    cA=dictInfo[mA]['C']
    cB=dictInfo[mB]['C']
    eHA=dictInfo[mA]['EH']
    eHB=dictInfo[mB]['EH']
    eHC=dictInfo[mC]['EH']
    delta_gamma_sigma_AM = surfEnergy(mA,mC,dictInfo,T)
    delta_gamma_sigma_BM = surfEnergy(mB,mC,dictInfo,T)
    EelasticAM = Eelastic(mA,mC,mB,dictInfo)
    EelasticBM = Eelastic(mB,mC,mA,dictInfo)
    
    omegaAM=omega_calc(mA,mC,mB,dictInfo)
    omegaBM=omega_calc(mB,mC,mA,dictInfo)
    omegaAB=omega_calc(mA,mB,mC,dictInfo)
    
    omegaprime = omegaAB-omegaAM-omegaBM 

    def solveHseg(q):
        x = q[0] # surf cA
        y = q[1] # surf cB
        Q1 = q[2] # HsegA
        Q2 = q[3] # HsegB
        
        F = np.empty((4))
        F[0] = (cA/(1-cA-cB))*(np.exp(-Q1/(kB*T)))*(1-x-y) - x# langmuir mclean
        F[1] = (cB/(1-cA-cB))*(np.exp(-Q2/(kB *T)))*(1-x-y) - y# langmuir mclean
        F[2] = delta_gamma_sigma_AM + 2*omegaAM *(Zl*(cA-x)+Zv*(cA-1/2))+omegaprime*(Zl*(y-cB)-Zv*cB) - EelasticAM + cov*(eHA-eHC) - Q1
        F[3] = delta_gamma_sigma_BM + 2*omegaBM *(Zl*(cB-y)+Zv*(cB-1/2))+omegaprime*(Zl*(x-cA)-Zv*cA) - EelasticBM + cov*(eHB-eHC) - Q2
        
        return F
    
    
    HsegAGuess = delta_gamma_sigma_AM + 2*omegaAM *(Zv*(cA-1/2))+omegaprime*(-Zv*cB) - EelasticAM + cov*(eHA-eHC)
    HsegBGuess = delta_gamma_sigma_BM + 2*omegaBM *(Zv*(cB-1/2))+omegaprime*(-Zv*cA) - EelasticBM + cov*(eHB-eHC) 

    xs = fsolve(solveHseg,[0.2,0.75,HsegAGuess,HsegBGuess])
    return xs
