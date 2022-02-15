from modules.constants import *
import numpy as np
from modules.geometry_misc import pbc
from random import sample
from itertools import product
from ase.build import add_adsorbate
from ase.geometry import wrap_positions

def CountHydrogenAtomsAroundPosition(position, H_offsets,lattice_parameter,SC):
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
            
        lattice_parameter  : float
        
        SC : List
            supercell indices
            
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
    
def AdsorptionEnergy(atom,a1,a2, H_offsets,dictInfo,lattice_parameter,SC):
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
    dictInfo : a dictionary with all the materials parameters
    lattice_parameter : float
    SC : supercell indices

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
    if len(all_pos) == len(atom):
        maxZ = max(all_pos[:,2]) 
    else:
        maxZ = max(all_pos[:,2])-1  
        
    # if the a1 atom is on the surface, compute energy associated to this position
    if abs(a1Pos[2] - maxZ) < 1e-4 :
        # count the number of adsorbate atoms around the a1 position
        n_adsorbate_atoms_a1 = CountHydrogenAtomsAroundPosition(a1Pos, H_offsets,lattice_parameter,SC)
       
        # energy associated 
        EH_a1b = n_adsorbate_atoms_a1*eH_a1 / 3 
        EH_a1a = n_adsorbate_atoms_a1*eH_a2 / 3
        
    # if the a2 is on the surface, compute energy associated to this position
    if abs(a2Pos[2] - maxZ) < 1e-4 :
        # count the number of adsorbate atoms around the a1 position        
        n_adsorbate_atoms_a2 = CountHydrogenAtomsAroundPosition(a2Pos, H_offsets,lattice_parameter,SC)
        EH_a2b = n_adsorbate_atoms_a2*eH_a2 / 3
        EH_a2a = n_adsorbate_atoms_a2*eH_a1 / 3  
    

    EHb = EH_a1b+EH_a2b # H energy before switch
    EHa = EH_a1a+EH_a2a # H energy after switch
    EHtot = -(EHa-EHb)
    return EHtot

def addHydrogenInitial(atom,coverage,SC):
    """
    It is possible to set an initial coverage, with the "atomCoverage" parameter 
    defined at the beginning. As this is a fraction, we have to convert it to
    random positions on the surface first.

    Parameters
    ----------
    atom : object
        complete atom object
    
    coverage : float
        initial coverage of adsorbant
        
    SC : List
       supercell indices

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

def PickHydrogenSite(atom,H_offsets,mA,mB,mC,dictInfo,lattice_parameter,hollow_coord_all,at_nrA,at_nrB,at_nrC):
    """
    Pick a random site on the surface of the supercell and compute if it
    is energetically favourable to adsorb an hydrogen atom on this site.
        
    Parameters
    ----------
    atom : an ASE atom object
    H_offsets: current positions where hydrogen atoms are adsorbed
    mA,mB,mC : elements
    dictInfo : a dictionary with all the materials parameters
    lattice_parameter : float
    hollow_coord_all : coordinates of adsorption sites
    at_nrA,at_nrB,at_nrC : atomic numbers
    
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
    rand_hollow_site_i = sample(hollow_coord_all, k=1)[0]
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
    
    eHA=dictInfo[mA]["EH"]
    eHB=dictInfo[mB]["EH"]
    eHC=dictInfo[mC]["EH"]

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