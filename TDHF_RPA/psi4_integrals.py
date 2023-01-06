import psi4
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import sys

psi4.set_memory("1GB")
h2 = psi4.geometry("""
0 1
H 0 0 0
H 0 0 0.7414
symmetry c1
"""
)

h2o = psi4.geometry("""
0 1 
O 0 0 0
H 0 0.866812 -0.677228
H 0 -0.866812 -0.677228
symmetry c1
"""
)

ch4 = psi4.geometry("""
0 1
C 0 0 0 
H 0.6276 0.6276 0.6276
H 0.6276 -0.6276 -0.6276
H -0.6276 0.6276 -0.6276
H -0.6276 -0.6276 0.6276
symmetry c1
"""
)

h2o_opt = psi4.geometry("""
0 1
O 0 0 0.1173
H 0 0.7572 -0.4692
H 0 -0.7572 -0.4692
symmetry c1
"""
)

o2 = psi4.geometry("""
0 1
O 0 0 0
O 0 0 1.2075
symmetry c1
"""
)

co = psi4.geometry("""
0 1
C 0 0 0 
O 0 0 1.1282
symmetry c1
"""
)

ch3f = psi4.geometry("""
0 1
C 0 0 -0.6244
F 0 0 0.7402
H 0 1.0245 -0.9718
H 0.8873 -0.5123 -0.9718
H -0.8873 -0.5123 -0.9718
symmetry c1
"""
)


psi4.set_options({'basis':'sto-3g'})
wfn = psi4.core.Wavefunction.build(ch4, psi4.core.get_global_option('basis'))

# Initialize MintsHelper with wavefunction's basis set
mints = psi4.core.MintsHelper(wfn.basisset())

nuc_rep = ch4.nuclear_repulsion_energy()

#Calculating 1e and 2e integrals in AO basis
# Overlap
S = np.asarray(mints.ao_overlap())
# 1e Hamiltonian
#h = np.asarray(mints.ao_kinetic())+np.asarray(mints.ao_potential())
t = np.asarray(mints.ao_kinetic())
v = np.asarray(mints.ao_potential())
# ERI -2e integrals
g = np.asarray(mints.ao_eri())
print(g.shape[0])
# Dipole Integrals
#mu = np.asarray([np.asarray(i) for i in mints.ao_dipole()])
#Electric Field
#ote to self: create a class that generalizes the electric field object
#np.savetxt('./enuc.txt', nuc_rep)
file = open('enuc.txt','w')
file.write(str(nuc_rep))
file.close()
#np.savetxt('./S.txt',S) 
#np.savetxt('./t.txt',t)
#np.savetxt('./v.txt',v)
#np.savetxt('./eri.txt',g.reshape((g.shape[0]**2,-1))) #two electron integrals

file = open('S.txt', 'w')
for ii in range(S.shape[0]):
    for jj in range(S.shape[1]):
        if(ii>=jj):
            file.write(str(ii+1) + "\t" + str(jj +1) +"\t %.15f \n" % S[ii, jj])
file.close()


file = open('t.txt', 'w')
for ii in range(t.shape[0]):
    for jj in range(t.shape[1]):
        if(ii>=jj):
            file.write(str(ii+1) + "\t" + str(jj +1) +"\t %.15f \n" % t[ii, jj])
file.close()

file = open('v.txt', 'w')
for ii in range(v.shape[0]):
    for jj in range(v.shape[1]):
        if(ii>=jj):
            file.write(str(ii+1) + "\t" + str(jj +1) +"\t %.15f \n" % v[ii,jj])
file.close()

file = open('eri.txt', 'w')
for ii in range(g.shape[0]):
    for jj in range(g.shape[1]):
        for kk in range(g.shape[2]):
            for ll in range(g.shape[3]):
                ij = ii*(ii+1)/2 + jj
                kl = kk*(kk+1)/2 + ll
                if (ii>=jj and kk>=ll and ij >= kl and abs(g[ii, jj, kk, ll])>0.000001):
                #if (ii>=jj and kk>=ll and ij >= kl):
                    file.write(str(ii + 1) + "\t" + str(jj + 1) + "\t" + str(kk + 1) + "\t" + str(ll + 1) + "\t" + str(g[ii, jj, kk, ll])+ "\n")

file.close()

#np.savetxt('./mux.txt',mu[0])
#np.savetxt('./muy.txt',mu[1])
#np.savetxt('./muz.txt',mu[2])
