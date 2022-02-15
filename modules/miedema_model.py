from scipy.optimize import fsolve
from modules.constants import *

def Hsol(material1,material2):
    """
    Computing the solution enthalpy

    Parameters
    ----------
    material1 : dictionary
        Contains info on the material
        format accepted: {"c":cAA, "V2_3": VAA2_3, "n":nAA, "phi":phiAA, "V":VAA}
    material2 : dictionary
        DESCRIPTION.

    Returns
    -------
    Hsol_AinB: float
        solution enthalpy of A in B
    Hsol_BinA: float
        solution enthalpy of B in A
    fAB: float
        degree of A surrounded by B, e.g. if equal to 1, A fully surrounded by B
    fBA: float
        degree of B surrounded by A

    """

    # defining variables to used for computations later 
    cAA, cBB = material1["c"], material2["c"]
    VAA, VBB = material1["V"], material2["V"]

    delta_n = material1["n"]-material2["n"]           #electron density difference        
    n_av = (1/material1["n"]+1/material2["n"])/2      #average electron density
    delta_phi = material1["phi"]-material2["phi"]     #chemical potential difference

    def solveVolume(p):
        """
        Two sets of equations are created and solved using fsolve
        fBA computed through cAS and vAlloy, needed for Hmix.           
            {x = x0 (1+alpha*fBA(x,y) (phiA-phiB))
            {y = y0 (1+alpha*fAB(x,y) (phiB-phiA))

        Parameters
        ----------
        p : tuple
           (VA2_3 = x, VB2_3 = y) the original volumes

        Returns
        -------
        (VA2_3, VB2_3): tuple of floats
            the corrected volumes

        """  
        x, y = p
        cAS=(cAA*x)/(cAA*x+cBB*y)              # eq.10
        cBS=(cBB*y)/(cAA*x+cBB*y)              # 1-cAS
        fBA=cBS*(1+8*((cAS**2*cBS**2)))     # eq.9
        fAB=cAS*(1+8*((cAS**2*cBS**2)))
        return (x-material1["V2_3"]*(1+alpha*fBA*delta_phi),
                y-material2["V2_3"]*(1+alpha*fAB*-delta_phi))

    VA_alloy, VB_alloy =  fsolve(solveVolume, (material1["V2_3"], material2["V2_3"])) # corrected values for volume
    cAS=(cAA*VA_alloy)/(cAA*VA_alloy+cBB*VB_alloy) # eq.10 with corrected volumes
    cBS=(cBB*VB_alloy)/(cAA*VA_alloy+cBB*VB_alloy)              # 1-cAS

    fBA=cBS*(1+8*((cAS**2*cBS**2)))     # eq.9 corrected value
    fAB=cAS*(1+8*((cAS**2*cBS**2)))

    # corerction factor proposed by Wang et al. (2007) AinB = BinA
    SxAinB = 1 - (0.5*cAA*cBB*abs(VAA-VBB))/(cAA*cAA*VAA+cBB*cBB*VBB)

    Hsol_AinB=SxAinB*VA_alloy/n_av*(-P*(delta_phi)**2+Q*(delta_n)**2) # eq.11
    Hsol_BinA=SxAinB*VB_alloy/n_av*(-P*(delta_phi)**2+Q*(delta_n)**2) # eq.11

    return Hsol_AinB, Hsol_BinA, fAB, fBA

def omega_calc(a1Type,a2Type,cType,dictInfo):
    """
    compute the omega parameter from the mixing enthalpy

    Parameters
    ----------
    a1Type : atom a1
    a2Type : atom a2
    cType : third atom
    dictInfo : dictionary with materials information

    Returns
    -------
    omega : value of omegaA1A2

    """    
    # setting the concentrations of the atoms 
    cAA, cBB, cCC = dictInfo[a1Type]["C"], dictInfo[a2Type]["C"], dictInfo[cType]["C"]
    VAA2_3, VBB2_3, VCC2_3 = dictInfo[a1Type]["A"], dictInfo[a2Type]["A"],  dictInfo[cType]["A"]
    nAA, nBB, nCC = dictInfo[a1Type]["n"], dictInfo[a2Type]["n"], dictInfo[cType]["n"]
    phiAA, phiBB, phiCC = dictInfo[a1Type]["phi"], dictInfo[a2Type]["phi"], dictInfo[cType]["phi"]
    VAA, VBB, VCC = dictInfo[a1Type]["V"], dictInfo[a2Type]["V"], dictInfo[cType]["V"]
    
    # computing the mixed variables
    cAB = cAA + cBB
    VAB2_3 = (cAA*VAA+cBB*VBB)/cAB
    phiAB = (cAA*phiAA + cBB*phiBB)/cAB
    nAB = (cAA*nAA+cBB*nBB)/cAB
    VAB = (cAA*VAA+cBB*VBB)/cAB
    
    # A in B 
    materialA = {"c":cAA, "V2_3": VAA2_3, "n":nAA, "phi":phiAA, "V":VAA}
    materialB = {"c":cBB, "V2_3": VBB2_3, "n":nBB, "phi":phiBB, "V":VBB}
    Hsol_AinB, Hsol_BinA, fAB, fBA = Hsol(materialA,materialB)
    
    # C in AB 
    materialC = {"c":cCC, "V2_3": VCC2_3, "n":nCC, "phi":phiCC, "V":VCC}
    materialAB = {"c":cAB, "V2_3": VAB2_3, "n":nAB, "phi":phiAB, "V":VAB}
    Hsol_CinAB, Hsol_ABinC, fCAB, fABC = Hsol(materialC,materialAB)

    HAinB = cAA*cBB*(fBA*Hsol_AinB + fAB*Hsol_BinA)
    HCinAB = (cAA + cBB) * cCC * (fABC*Hsol_CinAB + fCAB*Hsol_ABinC)
    Hmix = (1/3) * (HAinB + HCinAB)
    
    # compute omega IJ
    omega = Hmix/Z/cAA/cBB/kj_eV
    
    return omega