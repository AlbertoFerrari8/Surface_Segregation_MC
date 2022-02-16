# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:23:16 2020

Miedema + Monte Carlo for Surface Segregation 
"""
import numpy as np
import pandas as pd

from itertools import product
from pathlib import Path
import sys 

from ase.build import fcc111, add_adsorbate
from ase import io
from ase.geometry import wrap_positions

import random
from random import sample

from scipy.interpolate import interp1d
from scipy.optimize import fsolve


configFile = pd.read_csv('input/config.csv')  

mA = configFile['mA'][0]
mB = configFile['mB'][0]
mC = configFile['mC'][0]
cA = configFile['cA'][0]
cB = configFile['cB'][0]
Tend = configFile['T'][0]

"""" Structure Parameters """
# cA = 0.2   # fraction solute <0.5
# cB = 0.2   # fraction solute <0.5
# mA = "Ag"     # solute 
# mB = "Cu"     # solute
# mC = "Pd"     # solvent


vacA = 15     # vacuum thickness
SC = (12,12,100)

# set both to 0 for a vacuum environment ALSO comment set EHtot in energy calc
coverage = 1 # fraction coverage, e.g. coverage =0.5 and 24 surf atoms, 12 adsorbate atoms
p_adsorbate_addRemove = 0.01 # chance of adding/removing hydrogen atoms to the surface

Press = 1
Press0 = 1


""" Monte Carlo Parameters """
N_STEPS = 3000000           # number of MC steps
N_PRINT = 10000             # frequency for printing
SEED = 1                    # seed for randomizer
Tstart = 2000               # starting temperature [K], 
# Tend = 500                  # final temperature [K]  
T_ramp = 0.05 * N_STEPS     # ramping length for temperature
Press = 1
Press0 = 1

cC = 1-cA-cB                # initial concentration of C

# import data for el_list.csv. Then we set constants to the variables
data = pd.read_csv("input/el_list.csv").set_index('el')

# set variables from csv file
# we can write this as a dictionary such that dictInfo becomes unnecessary
VA, VA2_3, nA1_3, phiA, gammaA, KA, GA, at_nrA, lattice_parameter, eHA = data.at[mA, 'volume'], data.at[mA, 'area'], data.at[mA, 'n'], data.at[mA, 'phi'], data.at[mA, 'gamma'], data.at[mA, 'K'], data.at[mA, 'G'], data.at[mA, 'at_nr'], data.at[mA, 'A_lat'], data.at[mA, 'eH']
VB, VB2_3, nB1_3, phiB, gammaB, KB, GB, at_nrB, eHB = data.at[mB, 'volume'], data.at[mB, 'area'], data.at[mB, 'n'], data.at[mB, 'phi'], data.at[mB, 'gamma'], data.at[mB, 'K'], data.at[mB, 'G'], data.at[mB, 'at_nr'], data.at[mB, 'eH']
VC, VC2_3, nC1_3, phiC, gammaC, KC, GC, at_nrC, eHC = data.at[mC, 'volume'], data.at[mC, 'area'], data.at[mC, 'n'], data.at[mC, 'phi'], data.at[mC, 'gamma'], data.at[mC, 'K'], data.at[mC, 'G'], data.at[mC, 'at_nr'], data.at[mC, 'eH']

# dictionary of variables used to call properties of materials
dictInfo = {mA : {'C':cA, 'A':VA2_3, 'n':nA1_3, 'phi':phiA, 'y':gammaA, 'K':KA, 'G':GA, 'V':VA, 'EH': eHA},
            mB : {'C':cB, 'A':VB2_3, 'n':nB1_3, 'phi':phiB, 'y':gammaB, 'K':KB, 'G':GB, 'V':VB, 'EH': eHB},
            mC : {'C':cC, 'A':VC2_3, 'n':nC1_3, 'phi':phiC, 'y':gammaC, 'K':KC, 'G':GC, 'V':VC, 'EH': eHC}}

# not all adsorbation energies are in el_info, give warning when it is not in
if eHA == 0 or eHB == 0 or eHC == 0:
    print("ADSORPTION ENERGY IS NOT IN EL_INFO")

""" Constants """
Zl=6                    # coordination number within the layer
Zv=3                    # coordination number interlater
Z=Zl+2*Zv               # for FCC Zl = 6 and Zv = 3
P=12.35                 # Miedema constant
Q=115.62                # Miedema constant
R=47.97                 # Miedema constant
RR=0.00863              # gas constant [eV/atom/K]
kj_eV = 96.4853365      # kJ/mol to eV/atom conversion (google "ev per atom")
b=-4.9e-11              # temperature constant of surface energy
alpha=0.04              # constant
kB = 8.617333262145E-5  # Boltzmanns constant [eV/K]
g=4.56e8                # constant depending on the shape of the Wigner-Seitz

# compute usefull lattice ratios
sq2=np.sqrt(2)
sq3=np.sqrt(3)
sq2sq3=sq2/sq3

# FCC nnb positions 
nnb_coord=np.array([[ 0,         1.0 / sq3,         -sq2sq3 ],
                    [ -0.5,	    -0.5*1.0 / sq3,		-sq2sq3 ],
                    [ 0.5,      -0.5*1.0 / sq3,		-sq2sq3 ],
                    [ -0.5,      0.5*sq3,			0 ],
                    [ 0.5,       0.5*sq3,			0 ],
                    [ -1,	     0,					0 ],
                    [ 1,	  	 0,					0 ],
                    [ -0.5,	    -0.5*sq3,			0 ],
                    [ 0.5,	    -0.5*sq3,			0 ],
                    [ 0,		-1.0 / sq3,			sq2sq3 ],
                    [ -0.5,	     0.5*1.0 / sq3,		sq2sq3 ],
                    [ 0.5,	     0.5*1.0 / sq3,		sq2sq3 ]])
nnb_coord = lattice_parameter/sq2 * nnb_coord

""" initial calculations and/or conversion  """
atom = fcc111(mA, size=SC, vacuum=vacA)   #create fcc111 surface of material A

n_atoms=SC[0]*SC[1]*SC[2]   # number of atoms
n_atoms_surf = SC[0]*SC[1]  # number of surface atoms
n_layers = SC[2]            # number of layers from the SC

# all hollow coordinate positions to add H
hollow_coord_all =  list(product(range(SC[0]), repeat=2)) 

# writing the final results to the following source, if this folder exists, we do NOT overwrite
source = str(mA+str(int((cA*100)))+mB+str(int((cB*100)))+mC+str(int((1-cA-cB)*100))+"T"+str(Tend)+" "+str(SC)+str(coverage))   
try:
    Path(source).mkdir(parents=True, exist_ok=True) # exist_ok=True -> overwrite allowed, False not allowed and exception error
except:
    print(f"Foldername '{source}' aready exists,")
    print("change the folder name in the directory or delete the folder\n")
    sys.exit(1)
    
    
    

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
    
    VA_alloy, VB_alloy =  fsolve(solveVolume, (VA2_3, VB2_3)) # corrected values for volume
    cAS=(cAA*VA_alloy)/(cAA*VA_alloy+cBB*VB_alloy) # eq.10 with corrected volumes
    cBS=(cBB*VB_alloy)/(cAA*VA_alloy+cBB*VB_alloy)              # 1-cAS

    fBA=cBS*(1+8*((cAS**2*cBS**2)))     # eq.9 corrected value
    fAB=cAS*(1+8*((cAS**2*cBS**2)))
    
    # corerction factor proposed by Wang et al. (2007) AinB = BinA
    SxAinB = 1 - (0.5*cAA*cBB*abs(VAA-VBB))/(cAA*cAA*VAA+cBB*cBB*VBB)
    
    Hsol_AinB=SxAinB*VA_alloy/n_av*(-P*(delta_phi)**2+Q*(delta_n)**2) # eq.11
    Hsol_BinA=SxAinB*VB_alloy/n_av*(-P*(delta_phi)**2+Q*(delta_n)**2) # eq.11
    
    return Hsol_AinB, Hsol_BinA, fAB, fBA


def omega_calc(a1Type,a2Type,cType):
    """
    compute the omega parameter from the mixing enthalpy

    Parameters
    ----------
    a1Type : atom a1
    a2Type : atom a2
    cType : third atom

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

omegaAB = omega_calc(mA,mB,mC)
omegaAC = omega_calc(mA,mC,mB)
omegaBC = omega_calc(mB,mC,mA)
print(f"omegaAB, {omegaAB}")
print(f"omegaAC, {omegaAC}")
print(f"omegaBC, {omegaBC}")


def surfEnergy(a1Type,a2Type):
    """
    Compute the surface energy  

    Parameters
    ----------
    a1Type : The type of the a1 atom
    a2Type : Type of a2 atom

    Returns
    -------
    delta_gamma_sigma_AB : surface energy

    """
    sigA = dictInfo[a1Type]["A"]/1e4            #surface area of A converted to [m2/mol] (/1e4)
    sigB = dictInfo[a2Type]["A"]/1e4            #surface area of B converted to [m2/mol] (/1e4)
    Hvap_A = g*(dictInfo[a1Type]['y']*sigA+b*Tend)       #calculated Hvap of A
    Hvap_B = g*(dictInfo[a2Type]['y']*sigB+b*Tend)       #calculated Hvap of B
    gamma_sigma_A = 0.174*Hvap_A
    gamma_sigma_B = 0.174*Hvap_B
    delta_gamma_sigma_AB=(gamma_sigma_A-gamma_sigma_B)/kj_eV # surface energy
    return delta_gamma_sigma_AB


def Eelastic(a1Type,a2Type,cType):
    """
    Compute the elastic energy

    Parameters
    ----------
    a1Type : The type of the a1 atom
    a2Type : Type of a2 atom
    cType : Type of the final atom in the ternary system

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
    
    gam_A = cAA+delta_ij*cC 
    gam_B = cBB+delta_ji*cC   
    
    EelasticAinB=((2*KAA*GBB*(VAA-VBB)**2)/(3*KAA*VBB+4*GBB*VAA))/kj_eV 
    EelasticBinA=((2*KBB*GAA*(VBB-VAA)**2)/(3*KBB*VAA+4*GAA*VBB))/kj_eV 
    
    EelasticAB = gam_A * gam_B*(gam_B*EelasticAinB + gam_A * EelasticBinA)
    return EelasticAB


def RandomizeStructure(atom, SEED):
    """
    Randomize the structure of an atom object
    
    Parameters
    ----------
    atom : the atom that should be scrambled
    SEED : the seed of the randomizer

    Returns
    -------
    atom : randomized structure

    """
    comp={mA:int(np.around(cA*n_atoms)),mB:int(np.around(cB*n_atoms)),mC:int(np.around(cC*n_atoms))}  # make dict {material; the number of atoms of atoms, ...}
    s=sum(comp.values())                                                                # count the number of atoms in comp        
    
    # check if the number of atoms in comp matches n_atoms
    # otherwise remove or add an atom of material_B
    if s>n_atoms:
        comp[mB]=comp[mB]-1
    if s<n_atoms:
        comp[mB]=comp[mB]+1
    
    list_els=[]                             # create an empty list to store all the symbols in
    for els in comp:
        list_els=list_els+[els]*comp[els]   # add all symbols that are in the composition to list_els
    atom.set_chemical_symbols(list_els)     # set the symbols to the atom 
    
    pos=list(atom.positions)                # get the positions of all the atoms
    for i in range(0,3):
        SEED = SEED + i
        random.seed(SEED)
        random.shuffle(pos)                 # shuffle all atoms to create a randomized system
    atom.positions=np.array(pos)            # set the randomized positions to the atom
    return atom


def energy_calc(a1, a2, atom, H_offsets):
    """ 
    Compute the energy of an atom at a certain position using a localized
    Miedema's model which can be used in Monte Carlo
    
    Parameters
    ----------
    a1, a2 : atom where energy is computed, only nearest neighbors are accounted for

    Returns
    -------
    energy : energy of atom at certain position dependent on surrounding atoms

    """
    
    # wrap atoms around atom a1 and a2 to get pbc, pbc=110, no z-dir as we have a surface here
    wrap_pos_a1 = wrap_positions(atom.positions,atom.get_cell(),pbc=[1,1,0],center=atom.get_scaled_positions()[a1])
    wrap_pos_a2 = wrap_positions(atom.positions,atom.get_cell(),pbc=[1,1,0],center=atom.get_scaled_positions()[a2])
    
    # create a list where the atoms should be, these positions do not have to exist
    # e.g. atom on surface + z step -> in vacuum
    virtual_nb_a1 = nnb_coord+wrap_pos_a1[a1]
    virtual_nb_a2 = nnb_coord+wrap_pos_a2[a2]
    
    # subtract virtual neighbour from all positions, creating new array for every neighbor
    vecdista1 = np.linalg.norm(wrap_pos_a1 - virtual_nb_a1[:, np.newaxis], axis=-1)
    vecdista2 = np.linalg.norm(wrap_pos_a2 - virtual_nb_a2[:, np.newaxis], axis=-1)

    # get indices of values below threshold
    closeidxa1 = np.flatnonzero(vecdista1.min(axis=0) < 1e-5) 
    closeidxa2 = np.flatnonzero(vecdista2.min(axis=0) < 1e-5)
        
    # get the atomic numbers of the neighbors around atom position a1 and a2
    neighbortypes_a1 = atom.get_atomic_numbers()[closeidxa1]
    neighbortypes_a2 = atom.get_atomic_numbers()[closeidxa2]
                
    # count the nr of atoms of type a1 surrounding position a1 and a2 - resulting in symmetry
    # e.g. a1=A, nb: 3A,9B and a2=B, nb:5A,7B -> 3-5=-2
    # e.g. a1=B, nb: 3A,9B and a2=A, nb:5A,7B -> 9-7=2    
    # e.g. a1=A, nb: 12A,0B and a2=B, nb:3A,9B -> 12-3=9
    # e.g. a1=B, nb: 12A,0B and a2=A, nb:3A,9B -> 0-9=-9  

    ZA1 = np.count_nonzero(neighbortypes_a1 == atom.get_atomic_numbers()[a1])  #nr of A atoms around a1
    ZA2 = np.count_nonzero(neighbortypes_a2 == atom.get_atomic_numbers()[a1])  #nr of A atoms around a2 
    
    ZB1 = np.count_nonzero(neighbortypes_a1 == atom.get_atomic_numbers()[a2])  #nr of B atoms around a1  
    ZB2 = np.count_nonzero(neighbortypes_a2 == atom.get_atomic_numbers()[a2])  #nr of B atoms around a2  
    
    # if a1 and a2 are neighbors asymetry wil occur 
    # A-B-(A)-(B)-A-B  ZA1-ZA2=0-2, ZA1+1-ZA2=1-2 
    # A-B-(B)-(A)-A-B  ZA1-ZA2=1-1, ZA1+1-ZA2=2-1
     
    # A-B-(A)-(B)-A-B  ZB1-ZB2=2-0   ZB1-1-ZB2=1-0
    # A-B-(B)-(A)-A-B  ZB1-ZB2=1-1   ZB1-1-ZB2=0-1 
    if a1 in closeidxa2:
        ZA1 = ZA1 + 1 
        ZB1 = ZB1 - 1 
    
    # dA and dB are used in energy computation    
    dA = ZA1 - ZA2 
    dB = ZB1 - ZB2    

    
    # determine the type of all the atoms (a1,a2 and the other)
    a1Type = atom.get_chemical_symbols()[a1]
    a2Type = atom.get_chemical_symbols()[a2]
    allTypes = [mA,mB,mC]
    allTypes.remove(a1Type)
    allTypes.remove(a2Type)
    cType = allTypes[0]    
    
    # compute surface energy
    delta_gamma_sigma_AB = surfEnergy(a1Type,a2Type)
    
    # compute the omega values
    omegaAB = omega_calc(a1Type,a2Type,cType)
    omegaAC = omega_calc(a1Type,cType,a2Type)
    omegaBC = omega_calc(a2Type,cType,a1Type)   
    
    # computing elastic energy
    ElAinB = Eelastic(a1Type,a2Type,cType) # same as ElBinA
    ElCinA = Eelastic(cType,a1Type,a2Type)
    ElCinB = Eelastic(cType,a2Type,a1Type)
    cAA, cBB, cCC = dictInfo[a1Type]["C"], dictInfo[a2Type]["C"], dictInfo[cType]["C"]
    Eelastictot = cAA*ElAinB - cBB*ElAinB + cCC*(-ElCinA + ElCinB)
    
    #compute the adsorbation energy
    EHtot = AdsorptionEnergy(atom,a1,a2, H_offsets) 

    # if both atoms have Z neighbors: bulk-bulk switch
    if len(neighbortypes_a1) == Z and len(neighbortypes_a2) == Z:
        energy = dA*(omegaAB+omegaAC-omegaBC) + dB*(omegaAC-omegaAB-omegaBC)
        
    # if both atoms have Zl + Zv neighbors: surface-surface switch
    elif len(neighbortypes_a1) == Zl+Zv and len(neighbortypes_a2) == Zl+Zv:
        energy = dA*(omegaAB+omegaAC-omegaBC) + dB*(omegaAC-omegaAB-omegaBC) + EHtot
    
    # if Z(=12) neighbours around a1, bulk-surf switch
    elif len(neighbortypes_a1) == Z and len(neighbortypes_a2) == Zl+Zv: 
        energy = dA*(omegaAB+omegaAC-omegaBC) + dB*(omegaAC-omegaAB-omegaBC) + Zv*omegaBC - Zv*omegaAC + delta_gamma_sigma_AB + Eelastictot  + EHtot

    # if Zl+Zv(=9) neighbours around a1, surf-bulk switch
    elif len(neighbortypes_a1) == Zl+Zv and len(neighbortypes_a2) == Z: 
        energy = dA*(omegaAB+omegaAC-omegaBC) + dB*(omegaAC-omegaAB-omegaBC) - Zv*omegaBC + Zv*omegaAC - delta_gamma_sigma_AB - Eelastictot  + EHtot
        
    else:
        print(a1Type,a2Type, cType)
        print(len(neighbortypes_a1))
        print(len(neighbortypes_a2))
        energy = 100          
        
    return energy




def concentrationLayer(atom):
    """
    Compute the concentration at every layer for a certain configuration

    Parameters
    ----------
    atom : current atom object

    Returns
    -------
    concentrations : list of concentrations per layer [c_layer0, c_layer1, ...]

    """
    n_A_atoms_in_Layer = []     # empty list to store the concentrations
    n_B_atoms_in_Layer = []     # empty list to store the concentrations
    
    # get the positions of the atoms and the atomic numbers
    pos = atom.get_positions()
    atnr = atom.get_atomic_numbers()
    
    # we stitch these togethere and sort them by the Z-coordinate
    pos_atnr = np.hstack((pos, np.atleast_2d(atnr).T))
    possort = pos_atnr[pos_atnr[:,2].argsort()]
    
    # we only need the sorted atomic numbers (by z-dir) so that we can count
    # the amount of a certain species in a layer
    atnr_sort = possort[:,3]
    
    for layer in range(n_layers):    # go over all the layers
        xsA=0 #set counter for the nr of A atoms
        xsB=0 #set counter for the nr of A atoms
        
        # select a layer
        index_surf = atnr_sort[layer*n_atoms_surf:layer*n_atoms_surf+n_atoms_surf]
        
        for atoms in range(n_atoms_surf): # go over the amount of atoms that are in a layer
            xsA = xsA + np.count_nonzero([index_surf[atoms]] == at_nrA) # count the number of A atoms in the layer
            xsB = xsB + np.count_nonzero([index_surf[atoms]] == at_nrB) # count the number of A atoms in the layer
        n_A_atoms_in_Layer.append(xsA)    # append the number of A atoms per layer to concentrations
        n_B_atoms_in_Layer.append(xsB)    # append the number of A atoms per layer to concentrations

      # convert to np array and divide by the number of atoms in the surface to get the concentration

    concentrationsA = np.array(n_A_atoms_in_Layer)/n_atoms_surf
    concentrationsB = np.array(n_B_atoms_in_Layer)/n_atoms_surf
    return [concentrationsA, concentrationsB]

def pbc(i,n):
    """
    Periodic Boundary conditions, input is given in integer positions
    
    Parameters
    ----------
    i : single dimension position (i.e. x,y,z pos)
    n : maximum position possible in lattice
    
    Returns
    -------
    i : position after pbc
    
    Examples
    -------
    i=6, n=5
    i = 6 - (5+1)*int(floor(6/6)) = 6-6*floor(0) = 0
    i=0, n = 5
    i = 0 - (5+1)*int(floor(0/6)) = 0-6*floor(0) = 0
    i=3, n = 5
    i = 3 - (5+1)*int(floor(3/6)) = 3-3*floor(1/2) = 3
    """

    i = i - (n+1)*int(np.floor(i/(n+1)))
    return i

def CountHydrogenAtomsAroundPosition(position, H_offsets):
        """
        Using the position and the hydrogen adsorption energy of the a1 and a2
        atom we compute the hydrogen adsorbtion energy to this position before 
        switching the a1 and a2 atom and after the switch

        Parameters
        ----------
        position : List
            coordinates of position of atom [x,y,z]
            
        H_offsets : List
            List that holds the current positions where adsorbed atoms are stored
            e.g. [[1,0],[3,1],[2,1]]
            
        Returns
        -------
        EH_b : float
            energy associated to atom before switching
            
        EH_a : float
            energy associated to atom after switching

        """
        # first we convert the position to an XY position
        # then we convert the x and y position to integer positions, because 
        # the hydrogen adsorbtion spots are defined by integer positions instead
        # of xy positions.
        # e.g. adsorbate on (0,0) site is the first adsorbate encountered in x and in y direction
        position = np.delete(position, 2) 
        position[0] *= sq2/lattice_parameter
        position[1] *= sq2/lattice_parameter/sq3*2
        position[0]  = np.round(position[0] - position[1] *0.5,2)

        # create an array of the 3 hollow positions around the atom, 'position' is top right atom
        # so we have to add the atom to the left and below the top right atom in an FCC system
        hollow_site = np.round([position, position+[-1,0], position+[0,-1]],2)
        
        # go over the three positions contained in 'hollow_site' and apply pbc to the x and y dir 
        for i in range(len(hollow_site)):
            hollow_site[i] = [pbc(hollow_site[i][0], SC[0]-1), pbc(hollow_site[i][1], SC[1]-1)]
        
        # find the positions where hydrogen in hollow_site match H_offsets
        # i.e. we compare the three positions that could hold a hydrogen atom from hollow_site
        # to the H_offsets list, which holds all the positions that currently store hydrogen atoms
        vecdist = np.linalg.norm(H_offsets - hollow_site[:, np.newaxis], axis=-1 )
        
        # count the number of hydrogen atoms around the atom
        n_adsorbate_atoms = len(np.flatnonzero(vecdist.min(axis=0) < 1e-5) )
               
        return n_adsorbate_atoms

def AdsorptionEnergy(atom,a1,a2, H_offsets):
    """
    Check if one or both of the atoms are on the surface
    If this is the case, Check if one or both of the atoms have one or more hydrogen atoms
    adsorbed to the atom.
    e.g. atomic position of surface atom +- the directions to the hollow site.
    Count the number of times this position is in the array that holds the positions of the 
    hydrogen atoms.
    Dependent on this number, compute the energy

    Parameters
    ----------
    atom : current atom object
    a1 : index of atom a1 in atom
    a2 : index of atom a2 in atom

    Returns
    -------
    EHtot : energy as a result of H adsorbtion when a1 and a2 are switched from position

    """
    # if there are no hydrogen atoms on the surface-> EHtot is always zero
    if H_offsets == []:
        EHtot = 0
        return EHtot
    
    # set initial adsorption energy for position a1 and a2, before the switch
    # if no hydrogen atoms are found around the atomic position, the adsorbtion
    # energy is 0
    
    # set initial energy for position a1 and a2, before and after to 0
    EH_a1b = 0   # energy due to H adsorbtion of a1 before switching
    EH_a2b = 0   # energy due to H adsorbtion of a2 before switching
    EH_a1a = 0   # energy due to H adsorbtion of a1 after switching
    EH_a2a = 0   # energy due to H adsorbtion of a2 after switching 


    # set hydrogen adsorbtion energies to the right variables
    eH_a1 = dictInfo[atom.get_chemical_symbols()[a1]]["EH"]
    eH_a2 = dictInfo[atom.get_chemical_symbols()[a2]]["EH"]
            
    # get the positions of the atoms a1 and a2
    a1Pos = atom.get_positions()[a1]
    a2Pos = atom.get_positions()[a2]
    
    # if the number of atoms is equal to the number of metal atoms, no adsorbate atoms 
    # and thus the max Z position is that of the maximum Z atom object
    # else we havehHydrogen atoms which are 1A above the max Z position, so subtract this
    all_pos = atom.get_positions()
    if len(all_pos) == n_atoms:
        maxZ = max(all_pos[:,2]) 
    else:
        maxZ = max(all_pos[:,2])-1  
        
    # if the a1 atom is on the surface, compute energy associated to this position
    if abs(a1Pos[2] - maxZ) < 1e-4 :
        # count the number of adsorbate atoms around the a1 position
        n_adsorbate_atoms_a1 = CountHydrogenAtomsAroundPosition(a1Pos, H_offsets)
       
        # energy associated 
        EH_a1b = n_adsorbate_atoms_a1*eH_a1 / 3 
        EH_a1a = n_adsorbate_atoms_a1*eH_a2 / 3
        
    # if the a2 is on the surface, compute energy associated to this position
    if abs(a2Pos[2] - maxZ) < 1e-4 :
        # count the number of adsorbate atoms around the a1 position        
        n_adsorbate_atoms_a2 = CountHydrogenAtomsAroundPosition(a2Pos, H_offsets)
        EH_a2b = n_adsorbate_atoms_a2*eH_a2 / 3
        EH_a2a = n_adsorbate_atoms_a2*eH_a1 / 3  
    

    EHb = EH_a1b+EH_a2b # H energy before switch
    EHa = EH_a1a+EH_a2a # H energy after switch
    EHtot = -(EHa-EHb)
    return EHtot


def addHydrogenInitial(atom):
    """
    It is possible to set an initial coverage, with the "atomCoverage" parameter 
    defined at the beginning. As this is a fraction, we have to convert it to
    random positions on the surface first.

    Parameters
    ----------
    atom : object
        complete atom object

    Returns
    -------
    atom : object
        atom obect with hydrogen added to it
    H_offsets : list
        list which holds all the hydrogen integer positions

    """
    atomCoverage = int(np.floor(coverage*SC[0]*SC[1])) # how many adsorbate atoms are there
    H_offsets = sample(list(product(range(SC[0]), repeat=2)), k=atomCoverage) # create tuple of SC

    # go over the amount of adsorbate atoms and add them to their positions
    for i in range(atomCoverage):
        add_adsorbate(atom, 'H' , 1, 'fcc', offset=H_offsets[i])
        atom.center(vacuum=10.0, axis=2)
        
    return atom, H_offsets

def addHydrogenToAtom(atom, H_offsets):
    """
    Add Hydrogen atoms on all the positions that are stored in H_offsets


    Parameters
    ----------
    atom : object
        complete atom object
    H_offsets : list
        positions where we want to add hydrogen to the atom object

    Returns
    -------
    atom : object
        new atom object containing the hydrogen positions 
    """
    
    # go over all the position in H_offsets and add a hydrogen atom to this position
    for i in range(len(H_offsets)):
        add_adsorbate(atom, 'H' , 1, 'fcc', offset=H_offsets[i])
        atom.center(vacuum=10.0, axis=2)
        
    return atom


def PickHydrogenSite(H_offsets):
    """
    Pick a random site on the surface of the supercell and compute if it
    is energetically favourable to adsorb an hydrogen atom on this site.
        
    Parameters
    ----------
    H_offsets: current positions where hydrogen atoms are adsorbed

    Returns
    -------
    Eadsorb : TYPE
        DESCRIPTION.
    rand_hollow_site_i : TYPE
        DESCRIPTION.

    """
    
    # get all the positions of the atom so we can use this to find the adsorbate layer
    all_pos = atom.get_positions()  
    
    # check if there are atoms adsorbed in the first place,
    # if this is not the case the size of the supercell is smaller 
    if H_offsets == []:
        maxZ= max(all_pos[:,2])       # H adsorbed on top Z_layer
    else:
        maxZ= max(all_pos[:,2])-1       # H adsorbed on top Z_layer, minus 1, from the height of the adsorbate
        
    # pick a random atom from the surface
    rand_hollow_site_i = random.sample(hollow_coord_all, k=1)[0]
    rand_hollow_site_list = np.array(rand_hollow_site_i).astype(float)
    
    # coordinate transformation from xy position to integer position
    rand_hollow_site_list[0] *= lattice_parameter/sq2
    rand_hollow_site_list[0] += rand_hollow_site_i[1]*0.5*lattice_parameter/sq2
    rand_hollow_site_list[1] *= 0.5*sq3*lattice_parameter/sq2
    
    # transfer back to xyz pos
    rand_hollow_site_list = np.append(rand_hollow_site_list, maxZ)
    
    
    surf_idx1 = np.where(np.all(abs(all_pos - rand_hollow_site_list) <1e-5, axis=1))[0][0]
    
    hollow_nb_coord = np.array([[0.5, 0.5*sq3,0 ],     # array that stores the positions of the atom up right ... 
                      [1,0,0]])*lattice_parameter/sq2  # and right from surf_pos

    wrap_pos_ur = wrap_positions(atom.positions,atom.get_cell(),pbc=[1,1,0],center=atom.get_scaled_positions()[surf_idx1])
    hollow_nb = hollow_nb_coord+wrap_pos_ur[surf_idx1] # the upright and right neighbors surrounding the hollow site
    vecdist_H = np.linalg.norm(wrap_pos_ur - hollow_nb[:, np.newaxis], axis=-1) 
    closeidx_H = np.flatnonzero(vecdist_H.min(axis=0) < 1e-5) # find indices of atoms that surround hollow site
    neighbortypes_ah1 = atom.get_atomic_numbers()[closeidx_H] # get the atomic nrs of the atoms
    
    surf_type1= neighbortypes_ah1[0] #  up right atom 
    surf_type2 = neighbortypes_ah1[1] #  right atom

    surf_type_or = atom.get_atomic_numbers()[surf_idx1] # atom types of the original atom
    atnr_hollow = [surf_type1, surf_type2, surf_type_or]
    

    Eadsorb = 0
    for i in atnr_hollow:
        if i == at_nrA:
            Eadsorb += eHA
        elif i == at_nrB:
            Eadsorb += eHB
        elif i == at_nrC:
            Eadsorb += eHC
        else:
            print("Eadsorb error")    
    Eadsorb = -Eadsorb/3      
   
    
    return Eadsorb, rand_hollow_site_i

def LangmuirMcLean(cov):
    """
    Compute the surface composition using the Langmuir-McLean equation
    
    Parameters
    ----------
    q : [guess surface mA, guess surface mB, segregationEnthalpyGuess]
    
    Returns
    -------
    F: [final surface composition, final Hseg]
    
    """
    delta_gamma_sigma_AM = surfEnergy(mA,mC)
    delta_gamma_sigma_BM = surfEnergy(mB,mC)
    EelasticAM = Eelastic(mA,mC,mB)
    EelasticBM = Eelastic(mB,mC,mA)
    
    omegaAM=omega_calc(mA,mC,mB)
    omegaBM=omega_calc(mB,mC,mA)
    omegaAB=omega_calc(mA,mB,mC)
    
    omegaprime = omegaAB-omegaAM-omegaBM 

    def solveHseg(q):
        x = q[0] # surf cA
        y = q[1] # surf cB
        Q1 = q[2] # HsegA
        Q2 = q[3] # HsegB
        
        F = np.empty((4))
        F[0] = (cA/(1-cA-cB))*(np.exp(-Q1/(kB*Tend)))*(1-x-y) - x# langmuir mclean
        F[1] = (cB/(1-cA-cB))*(np.exp(-Q2/(kB *Tend)))*(1-x-y) - y# langmuir mclean
        F[2] = delta_gamma_sigma_AM + 2*omegaAM *(Zl*(cA-x)+Zv*(cA-1/2))+omegaprime*(Zl*(y-cB)-Zv*cB) - EelasticAM + cov*(eHA-eHC) - Q1
        F[3] = delta_gamma_sigma_BM + 2*omegaBM *(Zl*(cB-y)+Zv*(cB-1/2))+omegaprime*(Zl*(x-cA)-Zv*cA) - EelasticBM + cov*(eHB-eHC) - Q2
        
        return F
    
    
    HsegAGuess = delta_gamma_sigma_AM + 2*omegaAM *(Zv*(cA-1/2))+omegaprime*(-Zv*cB) - EelasticAM + cov*(eHA-eHC)
    HsegBGuess = delta_gamma_sigma_BM + 2*omegaBM *(Zv*(cB-1/2))+omegaprime*(-Zv*cA) - EelasticBM + cov*(eHB-eHC) 

    xs = fsolve(solveHseg,[0.2,0.75,HsegAGuess,HsegBGuess])
    return xs


# randomize the supercell
RandomizeStructure(atom, SEED)

# add the initial coverage of hydrogen
atom, H_offsets = addHydrogenInitial(atom)

# write the initialized randomized structure with hydrogen 
# io.write(source+"/initial.xyz",atom)         
io.write(source+"/state_0.xyz",atom)
       

""" Metropolis Monte Carlo  """
elist=np.zeros(N_STEPS)
concentration_listA = []
concentration_listB = []
energyT= 0 
energyH = 0
N_accepted = 0

# chemical potential vs temperature of hydrogen 
chem_pot = [-0.036, -0.064, -0.094, -0.126, -0.159, -0.194, -0.229, -0.266, -0.303, -0.341, -0.38, -0.42, -0.46, -0.5, -0.541, -0.583, -0.625, -0.667, -0.71, -0.754, -0.797, -0.841, -0.886, -0.93, -0.975, -1.021, -1.066, -1.112, -1.158, -1.205, -1.251, -1.299, -1.346, -1.393, -1.441, -1.489, -1.537, -1.586, -1.635]
T = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000]

# create function(T), so that we get the chemical potential at the temperature we need
f = interp1d(T, chem_pot)


for i in range(0,N_STEPS):              # go over the number of Monte Carlo Steps   
    if i < T_ramp:
        T = Tstart-(Tstart-Tend)/T_ramp * i
    elif i>= T_ramp:
        T = Tend
              
    p_h = random.random()
    
  ############ H adsorption ############ 
    # every step, a chance that adsorbated added/removed
    if p_h < p_adsorbate_addRemove:
        # pick a random site on the surface where hydrogen could adsorb (or is already there)
        # compute the energy associated to this spot by taking the three atoms that surround this position
        Eadsorb, Hsite = PickHydrogenSite(H_offsets)
        
        # chemical potential energy of hydrogen to be in gas phase, always negative
        EchemH = f(T) + kB*T*np.log(Press/Press0)

        # both Eadsorb and EchemH are negative values, the more negative value is preferred
        # if energy difference value is negative, adsorbtion is preferred 
        energyH = Eadsorb - EchemH 
        
        # If Hydrogen atom in the hollow site, compute if we should remove it (entropy) or keep it      
        if Hsite in H_offsets:
            # if the energy is smaller than 0 we remove the hydrogen atom depending on outcome of Metropolis
            if energyH <= 0 : 
                ph1=random.random()                  # get random float between 0 and 1
                
                # adsorbate atom is removed if this is True
                # energyH already negative, so energyH should not be -energyH
                if ph1<np.exp(energyH/kB/T):        # if  float is lower than energy/kB/T     
                    H_offsets.remove(Hsite)          # we accept the change and we remove the hydrogen atom
                    del atom[[atom.index for atom in atom if atom.symbol=='H']]
                    atom = addHydrogenToAtom(atom, H_offsets)
                    
                    # adsorbate atom is removed, leads to an increase of energy, 
                    # energyH is negative so minus energyH to increase energyT
                    energyT = energyT - energyH 
                    elist[i] = energyT
                
                # otherwise the adsorbate atom stays in place and energy stays the same
                else:
                    elist[i] = energyT   
                    
            #  if the energy change is larger than 0, remove H        
            elif energyH > 0:              
                H_offsets.remove(Hsite)
                del atom[[atom.index for atom in atom if atom.symbol=='H']]
                atom = addHydrogenToAtom(atom, H_offsets)
                
                # energyH is positive, removing it results in lower energy, so minus
                energyT = energyT - energyH
                elist[i] = energyT
                
        # If Hydrogen atom NOT in the hollow site, compute if we should add it           
        else:
            if energyH <= 0 : # if adding hydrogen atom results in lower energy, add H atom
                H_offsets.append(Hsite)
                del atom[[atom.index for atom in atom if atom.symbol=='H']]
                atom = addHydrogenToAtom(atom, H_offsets)
                
                # adsorbate atom is added, results in lower energy, energyH is negative so +
                energyT = energyT + energyH
                elist[i] = energyT
                
            # otherwise the adsorbate position stays empty and energy stays the same
            else:
                ph1=random.random()                  # get random float between 0 and 1
                if ph1<np.exp(-energyH/kB/T):        # if  float is lower than energy/kB/T  
                    H_offsets.append(Hsite)
                    del atom[[atom.index for atom in atom if atom.symbol=='H']]
                    atom = addHydrogenToAtom(atom, H_offsets)
                
                    # adsorbate atom is added, results in lower energy, energyH is negative so +
                    energyT = energyT + energyH
                    elist[i] = energyT
                else:
                    elist[i] = energyT

                
    ########## Switiching Atoms ########## 
    else:
        a1=random.randint(0,n_atoms-1)      # get random integer, which represents the first random atom
        while True:                         # start loop
            a2=random.randint(0,n_atoms-1)  # get random integer, which represents the second random atom
            if atom.get_atomic_numbers()[a2]!=atom.get_atomic_numbers()[a1]:    # check if the atoms are not of the same type
                break                                                           # as switching same type results in no energy change
   
        energy = energy_calc(a1, a2, atom, H_offsets)       # calculate energy before switch
 
    
        if  energy<=0:           
            # if the energy at the new position is lower
            atom_new=atom.copy()                    # copy the old system
            new=atom.get_atomic_numbers().copy()    # copy old atomic numbers
            new[a1],new[a2]=new[a2],new[a1]         # switch atom a1 and a2 around
            atom_new.set_atomic_numbers(new)        # copy switched atoms to new system
            atom=atom_new.copy()                # keep new system
            energyT = energyT+energy
            elist[i] = energyT
            N_accepted = N_accepted + 1
    
        else:
            p=random.random()                   # get random float between 0 and 1
            if p<np.exp(-energy/kB/T):        # if this float is lower than energy/kB/T
                atom_new=atom.copy()                    # copy the old system
                new=atom.get_atomic_numbers().copy()    # copy old atomic numbers
                new[a1],new[a2]=new[a2],new[a1]         # switch atom a1 and a2 around
                atom_new.set_atomic_numbers(new)        # copy switched atoms to new system
                atom=atom_new.copy()             # keep new system
                energyT = energyT+energy
                elist[i] = energyT
                N_accepted = N_accepted + 1
    
            else:                               # if the new energy is higher and not accepted, energy stays the same
                elist[i] = energyT

    if (i+1)%N_PRINT == 0:    
        io.write(source+"/state_"+str(i+1)+".xyz",atom)
        concentration_listA.append(concentrationLayer(atom)[0])
        concentration_listB.append(concentrationLayer(atom)[1])
        print(i+1, "/", N_STEPS, 100*(i+1) / N_STEPS, "%", "   T=",T)


print(f"omegaAB, {omegaAB}")
print(f"omegaAC, {omegaAC}")
print(f"omegaBC, {omegaBC}")

xsH = list(LangmuirMcLean(1))
xs = list(LangmuirMcLean(0))



"""writing away data"""
df_elist = pd.DataFrame(elist).to_csv(source+'/E.csv', index=False, header=False)
df_concentration_list = pd.DataFrame(concentration_listA).to_csv(source+'/CA.csv', index=False, header=False)
df_concentration_list = pd.DataFrame(concentration_listB).to_csv(source+'/CB.csv', index=False, header=False)



dict_to_text = {"cA": cA,"cB": cB, "cC": cC, "mA": mA,"mB": mB,"mC": mC ,"SC": SC,"N_STEPS": N_STEPS, "T": Tend, 
                 "AR":N_accepted/N_STEPS, "xs":xs, "xsH":xsH}

with open(source+'/data.txt', 'w') as f:
    print(dict_to_text, file=f)


""" Printing """
print(25*'*')
print(f'Conditions: {mA}{cA}{mB}{cB}{mC}{cC} {Tend}[K]')  
print(f'Acceptence ratio: {N_accepted/N_STEPS}')
print(f"source: {source}")






