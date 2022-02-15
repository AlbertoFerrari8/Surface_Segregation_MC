import numpy as np

""" Constants """
P=12.35                 # Miedema constant
Q=115.62                # Miedema constant
R=47.97                 # Miedema constant
alpha=0.04              # constant
Zl=6                    # coordination number within the layer
Zv=3                    # coordination number interlater
Z=Zl+2*Zv               # for FCC Zl = 6 and Zv = 3

RR=0.00863              # gas constant [eV/atom/K]
kj_eV = 96.4853365      # kJ/mol to eV/atom conversion (google "ev per atom")
b=-4.9e-11              # temperature constant of surface energy
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