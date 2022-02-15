import numpy as np
import random

def RandomizeStructure(atom,SEED,mA,mB,mC,dictInfo):
    """
    Randomize the structure of an atom object
    
    Parameters
    ----------
    atom : the atom that should be scrambled
    SEED : the seed of the randomizer
    mA,mB,mC : elements
    dictInfo : dictionary with materials information

    Returns
    -------
    atom : randomized structure

    """
    cA=dictInfo[mA]['C']
    cB=dictInfo[mB]['C']
    cC=dictInfo[mC]['C']
    n_atoms=len(atom)
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

def concentrationLayer(atom,SC,at_nrA,at_nrB):
    """
    Compute the concentration at every layer for a certain configuration

    Parameters
    ----------
    atom : current atom object
    SC : supercell indices
    at_nrA, at_nrB : atomic numbers

    Returns
    -------
    concentrations : list of concentrations per layer [c_layer0, c_layer1, ...]

    """
    
    n_atoms=SC[0]*SC[1]*SC[2]   # number of atoms
    n_atoms_surf = SC[0]*SC[1]  # number of surface atoms
    n_layers = SC[2]            # number of layers from the SC
    
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