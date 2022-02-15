from modules.constants import *
from modules.miedema_model import Hsol

def surfEnergy(a1Type,a2Type,dictInfo,T):
    """
    Compute the surface energy  

    Parameters
    ----------
    a1Type : The type of the a1 atom
    a2Type : Type of a2 atom
    dictInfo : dictionary with materials information
    T : temperature

    Returns
    -------
    delta_gamma_sigma_AB : surface energy

    """
    sigA = dictInfo[a1Type]["A"]/1e4            #surface area of A converted to [m2/mol] (/1e4)
    sigB = dictInfo[a2Type]["A"]/1e4            #surface area of B converted to [m2/mol] (/1e4)
    Hvap_A = g*(dictInfo[a1Type]['y']*sigA+b*T)       #calculated Hvap of A
    Hvap_B = g*(dictInfo[a2Type]['y']*sigB+b*T)       #calculated Hvap of B
    gamma_sigma_A = 0.174*Hvap_A
    gamma_sigma_B = 0.174*Hvap_B
    delta_gamma_sigma_AB=(gamma_sigma_A-gamma_sigma_B)/kj_eV # surface energy
    return delta_gamma_sigma_AB

def Eelastic(a1Type,a2Type,cType,dictInfo):
    """
    Compute the elastic energy

    Parameters
    ----------
    a1Type : The type of the a1 atom
    a2Type : Type of a2 atom
    cType : Type of the final atom in the ternary system
    dictInfo : dictionary with materials information

    Returns
    -------
    EelasticAB : Elastic energy

    """
    
    # The subscripts A and B correspond to the solute and solvent element
    # assign variables depending on the types of the atoms which are used as input
    KAA, KBB  = dictInfo[a1Type]["K"], dictInfo[a2Type]["K"]
    GAA, GBB = dictInfo[a1Type]["G"], dictInfo[a2Type]["G"]
    VAA, VBB, VCC = dictInfo[a1Type]["V"], dictInfo[a2Type]["V"], dictInfo[cType]["V"]
    cAA, cBB, cCC = dictInfo[a1Type]["C"], dictInfo[a2Type]["C"], dictInfo[cType]["C"]
   
    materialA = {"c":cAA, "V2_3": dictInfo[a1Type]["A"], "n":dictInfo[a1Type]["n"], "phi":dictInfo[a1Type]["phi"], "V":VAA}
    materialB = {"c":cBB, "V2_3": dictInfo[a2Type]["A"], "n":dictInfo[a2Type]["n"], "phi":dictInfo[a2Type]["phi"], "V":VBB}
    materialC = {"c":cCC, "V2_3": dictInfo[cType]["A"],  "n":dictInfo[cType]["n"], "phi":dictInfo[cType]["phi"], "V":VCC}

    Hsol_AinB, Hsol_BinA, fAB, fBA = Hsol(materialA,materialB)
    Hsol_BinC, Hsol_CinB, fBC, fCB = Hsol(materialB,materialC)
    Hsol_AinC, Hsol_CinA, fAC, fCA = Hsol(materialA,materialC)
       
    lambda_i = (Hsol_BinA - Hsol_CinA)**2
    lambda_j = (Hsol_AinB - Hsol_CinB)**2
    
    delta_ij = lambda_i / (lambda_i+lambda_j)
    delta_ji = lambda_j / (lambda_i+lambda_j)
    
    gam_A = cAA+delta_ij*cCC 
    gam_B = cBB+delta_ji*cCC   
    
    EelasticAinB=((2*KAA*GBB*(VAA-VBB)**2)/(3*KAA*VBB+4*GBB*VAA))/kj_eV 
    EelasticBinA=((2*KBB*GAA*(VBB-VAA)**2)/(3*KBB*VAA+4*GAA*VBB))/kj_eV 
    
    EelasticAB = gam_A * gam_B*(gam_B*EelasticAinB + gam_A * EelasticBinA)
    return EelasticAB