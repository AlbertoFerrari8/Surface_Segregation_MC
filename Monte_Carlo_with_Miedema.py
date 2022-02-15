import numpy as np
import pandas as pd
from pathlib import Path
import sys 
from ase.build import fcc111
from ase import io
from itertools import product
import random
from scipy.interpolate import interp1d
from modules.geometry_misc import RandomizeStructure,pbc,concentrationLayer
from modules.miedema_model import omega_calc
from modules.constants import *
from ase.geometry import wrap_positions
from modules.energy_contributions import Eelastic,surfEnergy
from modules.LangmuirMcLean import LangmuirMcLean
from modules.adsorption import AdsorptionEnergy,addHydrogenInitial,addHydrogenToAtom,PickHydrogenSite

configFile = pd.read_csv('input/config.csv')  

mA = configFile['mA'][0]
mB = configFile['mB'][0]
mC = configFile['mC'][0]
cA = configFile['cA'][0]
cB = configFile['cB'][0]
Tend = configFile['T'][0]

vacA = 15     # vacuum thickness
SC = (3,3,7)  # supercell size

# set both to 0 for a vacuum environment ALSO comment set EHtot in energy calc
coverage = 1 # fraction coverage, e.g. coverage =0.5 and 24 surf atoms, 12 adsorbate atoms
p_adsorbate_addRemove = 0.01 # chance of adding/removing hydrogen atoms to the surface

Press = 1
Press0 = 1

""" Monte Carlo Parameters """
N_STEPS = 3000              # number of MC steps
N_PRINT = 300               # frequency for printing
SEED = 1                    # seed for randomizer
Tstart = 2000               # starting temperature [K], 
# Tend = 700                  # final temperature [K]  
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

omegaAB = omega_calc(mA,mB,mC,dictInfo)
omegaAC = omega_calc(mA,mC,mB,dictInfo)
omegaBC = omega_calc(mB,mC,mA,dictInfo)
print(f"omegaAB, {omegaAB}")
print(f"omegaAC, {omegaAC}")
print(f"omegaBC, {omegaBC}")

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
    delta_gamma_sigma_AB = surfEnergy(a1Type,a2Type,dictInfo,Tend)
    
    # compute the omega values
    omegaAB = omega_calc(a1Type,a2Type,cType,dictInfo)
    omegaAC = omega_calc(a1Type,cType,a2Type,dictInfo)
    omegaBC = omega_calc(a2Type,cType,a1Type,dictInfo)   
    
    # computing elastic energy
    ElAinB = Eelastic(a1Type,a2Type,cType,dictInfo) # same as ElBinA
    ElCinA = Eelastic(cType,a1Type,a2Type,dictInfo)
    ElCinB = Eelastic(cType,a2Type,a1Type,dictInfo)
    cAA, cBB, cCC = dictInfo[a1Type]["C"], dictInfo[a2Type]["C"], dictInfo[cType]["C"]
    Eelastictot = cAA*ElAinB - cBB*ElAinB + cCC*(-ElCinA + ElCinB)
    
    #compute the adsorbation energy
    EHtot = AdsorptionEnergy(atom,a1,a2, H_offsets,dictInfo,lattice_parameter,SC) 

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

# randomize the supercell
RandomizeStructure(atom,SEED,mA,mB,mC,dictInfo)

# add the initial coverage of hydrogen
atom, H_offsets = addHydrogenInitial(atom,coverage,SC)

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
        Eadsorb, Hsite = PickHydrogenSite(atom,H_offsets,mA,mB,mC,dictInfo,lattice_parameter,hollow_coord_all,at_nrA,at_nrB,at_nrC)
        
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
        concentration_listA.append(concentrationLayer(atom,SC,at_nrA,at_nrB)[0])
        concentration_listB.append(concentrationLayer(atom,SC,at_nrA,at_nrB)[1])
        print(i+1, "/", N_STEPS, 100*(i+1) / N_STEPS, "%", "   T=",T)


#print(f"omegaAB, {omegaAB}")
#print(f"omegaAC, {omegaAC}")
#print(f"omegaBC, {omegaBC}")

xsH = list(LangmuirMcLean(1,mA,mB,mC,dictInfo,Tend))
xs = list(LangmuirMcLean(0,mA,mB,mC,dictInfo,Tend))



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




