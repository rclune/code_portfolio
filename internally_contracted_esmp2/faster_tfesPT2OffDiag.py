"""
ESMF-MP2 correction with an internally contracted first order correction to the wave function
Last updated: 09/03/2019 by Rachel Clune
Contributors: Rachel Clune, Jacqueline A. R. Shea, Eric Neuscamman
"""

import time

# importing the modules I created to wrap the tcontractor outputs
# LHS (linear translation modules)
import r_aaaa_to_aa
#print (r_aaaa_to_aa.aaaa_to_aa.__doc__)

#import r_aaaa_to_aaaa_1
#print(r_aaaa_to_aaaa_1.aaaa_to_aaaa_1.__doc__)

#import r_aaaa_to_aaaa_2
#print(r_aaaa_to_aaaa_2.aaaa_to_aaaa_2.__doc__)

import r_aaaa_to_aabb
#print(r_aaaa_to_aabb.aaaa_to_aabb.__doc__)

import r_aaaa_to_bb
#print(r_aaaa_to_bb.aaaa_to_bb.__doc__)

import r_aaaa_to_bbaa
#print(r_aaaa_to_bbaa.aaaa_to_bbaa.__doc__)

import r_aa_to_aa
#print(r_aa_to_aa.aa_to_aa.__doc__)

import r_aa_to_aabb
#print(r_aa_to_aabb.aa_to_aabb.__doc__)

import r_aa_to_bb
#print(r_aa_to_bb.aa_to_bb.__doc__)

import r_aa_to_bbaa
#print(r_aa_to_bbaa.aa_to_bbaa.__doc__)

import r_aabb_to_aa
#print(r_aabb_to_aa.aabb_to_aa.__doc__)

import r_aabb_to_aabb
#print(r_aabb_to_aabb.aabb_to_aabb.__doc__)

import r_aabb_to_bb
#print(r_aabb_to_bb.aabb_to_bb.__doc__)

import r_aabb_to_bbaa
#print(r_aabb_to_bbaa.aabb_to_bbaa.__doc__)

# pt2 energy modules
import r_pt2_e_a
#print(r_pt2_e_a.pt2_e_a.__doc__)

import r_pt2_e_ab
#print(r_pt2_e_ab.pt2_e_ab.__doc__)

#LHS Overlap
import r_ovrlp_aa_to_aa
#print(r_ovrlp_aa_to_aa.ovrlp_aa_to_aa.__doc__)

import r_ovrlp_aa_to_aabb
#print(r_ovrlp_aa_to_aabb.ovrlp_aa_to_aabb.__doc__)

import r_ovrlp_aa_to_bb
#print(r_ovrlp_aa_to_bb.ovrlp_aa_to_bb.__doc__)

import r_ovrlp_aa_to_bbaa
#print(r_ovrlp_aa_to_bbaa.ovrlp_aa_to_bbaa.__doc__)

import r_ovrlp_aaaa_to_aa
#print(r_ovrlp_aaaa_to_aa.ovrlp_aaaa_to_aa.__doc__)

import r_ovrlp_aaaa_to_aabb
#print(r_ovrlp_aaaa_to_aabb.ovrlp_aaaa_to_aabb.__doc__)

import r_ovrlp_aaaa_to_bbaa
#print(r_ovrlp_aaaa_to_bbaa.ovrlp_aaaa_to_bbaa.__doc__)

import r_ovrlp_aabb_to_aa
#print(r_ovrlp_aabb_to_aa.ovrlp_aabb_to_aa.__doc__)

import r_ovrlp_aabb_to_aabb
#print(r_ovrlp_aabb_to_aabb.ovrlp_aabb_to_aabb.__doc__)

import r_ovrlp_aabb_to_bb
#print(r_ovrlp_aabb_to_bb.ovrlp_aabb_to_bb.__doc__)

import r_ovrlp_aabb_to_bbaa
#print(r_ovrlp_aabb_to_bbaa.ovrlp_aabb_to_bbaa.__doc__)

#RHS
# import r_rhs_aa
# #print(r_rhs_aa.rhs_aa_sub.__doc__)

# import r_rhs_aabb
# #print(r_rhs_aabb.rhs_aabb_sub.__doc__)

import r_rhs_aa
#print(r_rhs_aa.rhs_aa.__doc__)

import r_rhs_aabb
#print(r_rhs_aabb.rhs_aabb.__doc__)

#import r_rhs_aabb_f2py
#print(r_rhs_aabb_f2py.rhs_aabb_f2py.__doc__)

# these do not include a factor of 0.25
import r_pt2_e_aa
# print(r_pt2_e_aa.pt2_e_aa.__doc__)

import r_rhs_aaaa
#print(r_rhs_aaaa.rhs_aaaa.__doc__

# import r_rhs_aaaa
# #print(r_rhs_aaaa.rhs_aaaa_sub.__doc__)

# import r_aaaa_to_aaaa
# #print(r_aaaa_to_aaaa.aaaa_to_aaaa.__doc__)
#
# import r_aa_to_aaaa
# #print(r_aa_to_aaaa.aa_to_aaaa.__doc__)
#
# import r_aa_to_bbbb
# #print(r_aa_to_bbbb.aa_to_bbbb.__doc__)
#
# import r_aabb_to_aaaa
# #print(r_aabb_to_aaaa.aabb_to_aaaa.__doc__)
#
# import r_aabb_to_bbbb
# #print(r_aabb_to_bbbb.aabb_to_bbbb.__doc__)
#
# import r_ovrlp_aa_to_aaaa
# #print(r_ovrlp_aa_to_aaaa.ovrlp_aa_to_aaaa.__doc__)
#
# import r_ovrlp_aaaa_to_aaaa
# #print(r_ovrlp_aaaa_to_aaaa.ovrlp_aaaa_to_aaaa.__doc__)
#
# import r_ovrlp_aabb_to_aaaa
# #print(r_ovrlp_aabb_to_aaaa.ovrlp_aabb_to_aaaa.__doc__)
#
# import r_ovrlp_aabb_to_bbbb
# #print(r_ovrlp_aabb_to_bbbb.ovrlp_aabb_to_bbbb.__doc__)

# # these do include a factor of 0.25
# import r_pt2_e_aa_025 as r_pt2_e_aa
# #print(r_pt2_e_aa.pt2_e_aa.__doc__)

# import r_rhs_aaaa_025 as r_rhs_aaaa
# #print(r_rhs_aaaa.rhs_aaaa_sub.__doc__)

import r_aaaa_to_aaaa_025 as r_aaaa_to_aaaa
#print(r_aaaa_to_aaaa.aaaa_to_aaaa.__doc__)

import r_aa_to_aaaa_025 as r_aa_to_aaaa
#print(r_aa_to_aaaa.aa_to_aaaa.__doc__)

import r_aa_to_bbbb_025 as r_aa_to_bbbb
#print(r_aa_to_bbbb.aa_to_bbbb.__doc__)

import r_aabb_to_aaaa_025 as r_aabb_to_aaaa
#print(r_aabb_to_aaaa.aabb_to_aaaa.__doc__)

import r_aabb_to_bbbb_025 as r_aabb_to_bbbb
#print(r_aabb_to_bbbb.aabb_to_bbbb.__doc__)

import r_ovrlp_aa_to_aaaa_025 as r_ovrlp_aa_to_aaaa
#print(r_ovrlp_aa_to_aaaa.ovrlp_aa_to_aaaa.__doc__)

import r_ovrlp_aaaa_to_aaaa_025 as r_ovrlp_aaaa_to_aaaa
#print(r_ovrlp_aaaa_to_aaaa.ovrlp_aaaa_to_aaaa.__doc__)

import r_ovrlp_aabb_to_aaaa_025 as r_ovrlp_aabb_to_aaaa
#print(r_ovrlp_aabb_to_aaaa.ovrlp_aabb_to_aaaa.__doc__)

import r_ovrlp_aabb_to_bbbb_025 as r_ovrlp_aabb_to_bbbb
#print(r_ovrlp_aabb_to_bbbb.ovrlp_aabb_to_bbbb.__doc__)

#RHS Overlap


from tfesMinRes import genMinRes

# get starting time of the program
startTime = time.time()

import sys
import os
import string
import operator
import numpy as np
import scipy
import scipy.optimize
#import tensorflow as tf

from tfesReadFCIDUMP import my_ints
#from tfesMinRes import genMinRes
from tfesUtils import string_to_bool

# print time
print("")
print("Finished imports, elapsed time in seconds = ", ( time.time() - startTime ))
print("")
sys.stdout.flush()

print ("")
print ("Starting execution at ", time.asctime())
print ("")
sys.stdout.flush()

########################################################################################################################
# Functions for calculating the reference energies and reading in input files
########################################################################################################################

# function to print 2-d matrices nicely
def print_mat(x, ut=False):
    for i in range(x.shaoe[0]):
        for j in range(x.shape[1]):
            if j>=i or not ut:
                print("%16.8f" % x[i,j], end="")
                #print("%16.8f" % x[i, j])
            else:
                print("%16s" % " ", end="")
                #print "%16s" % " ",
        print ("")

# read the input file for locations of constant and integral files and store values in a dictionary
# expected input file has variable/matrix name then location of respective file
# comments in the input file are denoted by # at the start of the line
def get_user_input(filename):
    # default values
    d = {'diagonal'         : 'false',
         'coeff_thresh'     : '0.0',
         'comment'          : None,
         }

    f = open(filename, 'r')
    line = f.readline()
    line_number = 0

    while line:
        line_number += 1
        stripped = line.split('#')[0]  # removes commented lines
        split_line = stripped.split(maxsplit=1)
        #stripped = string.split(line, sep = '#')[0] # removes commented lines
        #split_line = string.split(stripped, maxsplit = 1)
        if len(split_line) == 1:
            raise RuntimeError("Only one word in input line %d" % line_number)
        elif len(split_line) == 2:
            d[split_line[0]] = split_line[1].strip()
            #d[split_line[0]] = string.strip(split_line[1])
        elif len(split_line) > 2:
            raise RuntimeError("Unexpectedly found more than two pieces in input line %d" % line_number)

        line = f.readline()

    # convert from strings to usable booleans
    d['diagonal'] = string_to_bool(d['diagonal'])
    d['coeff_thresh']  = float(d['coeff_thresh'])

    return d


# given a truncation threshold remove any irrelevant values in the Ca and Cb inputs
def truncate_coeffs(threshold, coeff_matrix):
    # turns the input matrix to a 1-d array
    coeff_array = np.reshape(np.copy(coeff_matrix), [coeff_matrix.size])
    for ii in range(coeff_array.size):
        if np.abs(coeff_array[ii])<threshold:
            coeff_array[ii] = 0.0
    return np.reshape(coeff_array, coeff_matrix.shape)


# Taylor Series expansion of exp(X)
# n_terms should be at least 10 for accuracy purposes
def create_U(Xmat, norb, n_terms):
    taylorTerms = []
    taylorTerms.append(np.identity(norb))
    U = np.zeros((Xmat.shape))
    for ii in range(n_terms):
        temp = np.matmul(taylorTerms[-1], Xmat/(1.0+ii))
        taylorTerms.append(temp)

    for jj in range(len(taylorTerms)):
        U = np.add(U, taylorTerms[jj])

    return U


# for getting the RHF energy
# E_RHF = 2 \sum_a h_aa + \sum_ab 2J_ab - K_ab
def get_det_energy(enuc, oei, tei, ova, ovb):
    de = 1.0 * enuc
    for p in ova:
        de = de + oei[p,p]
        for q in ova:
            de = de + 0.5 * (tei[p, p, q, q] - tei[p, q, p, q])
        for q in ovb:
            de = de + tei[p, p, q, q]
    for p in ovb:
        de = de + oei[p, p]
        for q in ovb:
            de = de + 0.5 * (tei[p, p, q, q] - tei[p, q, p, q])

    return de


########################################################################################################################
# Reading in and organizing the input values
########################################################################################################################

# Note that throughout this section the use of the labels AO and MO do not actually denote atomic and molecular orbitals
# but instead are for before and after the orbital rotations are accounted for, respectively.

# process input file
user_inputs = get_user_input(sys.argv[1])
print(user_inputs)

# print out user inputs
print("User input dictionary: ")
for key in user_inputs:
    print("%30s :        %30s" % (key, user_inputs[key]))
print("")
sys.stdout.flush()

# numpy settings
np.random.seed(123456)
np.set_printoptions(linewidth=250)

# read in one and two electron integrals in the atomic orbital basis from the pyscf fcidump file
integral_file = user_inputs['ints'] # should be fcidump.txt
print("Reading electron integrals from %s" % integral_file)
MyInts = my_ints(integral_file)

norb = MyInts.norb      # number of orbitals
nocc = int(MyInts.nocc)      # number of occupied orbitals
nvir = int(MyInts.nvir)      # number of virtual orbitals
nelec = int(MyInts.nelec)    # number of electrons

enuc = MyInts.nucRep    # nuclear repulsion energy

AO_oints = MyInts.one_elec_ints     # one-electron integrals before orbital rotation
AO_tints = MyInts.two_elec_ints     # two-electron integrals before orbital rotation

# get the matrices of coefficients
C0 = np.reshape(np.loadtxt(user_inputs['c0']), [1])
print(C0[0])
#Ca = np.reshape( np.loadtxt(user_inputs['ca']), [nocc*nvir] )
Ca = np.reshape(np.loadtxt(user_inputs['ca']), (nocc, nvir))
#Cb = np.reshape( np.loadtxt(user_inputs['ca']), [nocc*nvir] )
Cb = np.reshape(np.loadtxt(user_inputs['cb']), (nocc, nvir))
#X_ov_array = np.reshape( np.loadtxt(user_inputs['xut']), [nocc*nvir] )
X_ov_block = np.reshape(np.loadtxt(user_inputs['xut']), [nocc, nvir])

#for rhs cases
Cb = Ca

# getting the full X matrix, currently only the ov block is formed
oo_block = np.zeros((nocc, nocc))
bottom_half = np.zeros((nvir, norb))
temp = np.concatenate((oo_block, X_ov_block), axis=1) # end up with an nocc by norb matrix, the first nocc columns are 0
Xut = np.concatenate((temp, bottom_half), axis=0) # end up with an norb by norb matrix

# get the full antisymmetric matrix
Xmat = Xut - np.transpose(Xut)

print('C0 = ', C0)

#print ('shape of Xvec = ', X_ov_array.shape)

# to determine if the norm of the reference wave function is one
wfnNormCheck = np.sqrt(np.sum(np.square(C0)) + np.sum(np.square(Ca)) + np.sum(np.square(Cb)))

print("")
print("initial wfn norm = ", wfnNormCheck)

# enforcing that the norm of the reference wave function is one

C0 = C0 / wfnNormCheck
Ca = Ca / wfnNormCheck
Cb = Cb / wfnNormCheck

wfnNormCheck = np.sqrt( np.sum(np.square(C0)) + np.sum(np.square(Ca)) + np.sum(np.square(Cb)))

print("")
print("fixed wfn norm   = ", wfnNormCheck)

# truncate small coefficients
print("")
print("truncating Ca and Cb coefficients smaller than %.8f" % user_inputs['coeff_thresh'])
tCa = truncate_coeffs(user_inputs['coeff_thresh'], Ca)
tCb = truncate_coeffs(user_inputs['coeff_thresh'], Cb)


########################################################################################################################
# Getting reference energies, density, and storing the integrals
########################################################################################################################

# note that * between two matrices does not result in matrix multiplication, for example
# [[1,2],[3,4]] * [[5,6],[7,8]] becomes [[5,12],[21,32]]
# note that the names AO and MO do not actually mean atomic and molecular orbitals
# the orbitals being read in are already MO's they just haven't been rotated into their optimal positions
# will probably rename these later
# also the energy calculated below should match the energy calculated by the CIS code.
# should change this so that instead of recalculating it is just included in the input file, though for now it's just a way to check the numbers

# Using a 13-term Taylor series expansion for the representation of exp(X)
rotation_operator = create_U(Xmat, norb, 13)

# rotating the one-electron integrals into the MO basis
# (the rotation operator should be purely real so the adjoint of the rotation operator is the transpose)
# think F_pq -> F_PQ via U_Pp F_pq U_qQ^T
MO_oints = np.matmul(rotation_operator, np.matmul(AO_oints, np.transpose(rotation_operator)))

# rotating the two-electron integrals into the MO basis
# since the two-electron integrals are stored as a 4 dimensional tensor and the rotation operator is a matrix,
# this is not a straightforward process.
# think G_pqrs -> G_PQRS via G_pqrs U_pP U_qQ U_rR, U_sS, but need rearranging to get the indices to match correctly.

# G_pqrs -> G_pqr,s -> sum_s G_pqr,s (U_sS)^T -> G_pqr,S -> G_pqrS
temp = np.matmul(np.reshape(AO_tints, (norb * norb * norb, norb)), np.transpose(rotation_operator))
temp = np.reshape(temp, [norb, norb, norb, norb])
# G_pqrS -> G_pqSr
temp = np.transpose(temp, (0, 1, 3, 2))
# G_pqSr -> G_pqS,r -> sum_r G_pqS,r (U_rR)^T -> G_pqS,R -> G_pqSR
temp = np.matmul(np.reshape(temp, (norb * norb * norb, norb)), np.transpose(rotation_operator))
temp = np.reshape(temp, [norb, norb, norb, norb])
# G_pqSR -> G_qpRS
temp = np.transpose(temp, (1, 0, 3, 2))
# G_qpRS -> G_q,pRS -> sum_q U_qQ G_q,pRS -> G_Q,pRS -> G_QpRS
temp = np.matmul(rotation_operator, np.reshape(temp, (norb, norb * norb * norb)))
temp = np.reshape(temp, (norb, norb, norb, norb))
# G_QpRS -> G_pQRS
temp = np.transpose(temp, (1, 0, 2, 3))
# G_pQRS -> G_p,QRS _> sum_p U_pP G_p,QRS -> G_PQRS
temp = np.matmul(rotation_operator, np.reshape(temp, (norb, norb * norb * norb)))
MO_tints = np.reshape(temp, (norb, norb, norb, norb))

# splitting the one electron integrals into the occupied and virtual components
MO_oints_occ = MO_oints[:nocc, :nocc]
MO_oints_vir = MO_oints[nocc:, nocc:]

# one-electron density matrix
# if the density matrix was for a normal Fock operator, only the oo block would be nonzero and it would be an identity matrix
# The reason the oo block is not is because some of the occupied orbital occupations are going towards occupying some tof the virtual obritals
# the ov block comes from the occupied and virtual states being coupled because of the inclusion of excited states in the reference space
# similar for the vo block
# T the vv block is just the opposite of the oo block, it's the occupation of the virtual orbitals in the reference space

rho_oo = np.identity(nocc) - np.matmul(Ca, np.transpose(Ca))
rho_ov = C0 * Ca
rho_vo = C0 * np.transpose(Ca)
rho_vv = np.matmul(np.transpose(Ca), Ca)
rho_top = np.concatenate((rho_oo, rho_ov), 1)
rho_bot = np.concatenate((rho_vo, rho_vv), 1)
rho_tot = np.concatenate((rho_top, rho_bot), 0)

# Fock matrix, just the standard closed-shell restricted HF expression
Fock_mat = MO_oints + 2.0 * np.sum(MO_tints * np.reshape(rho_tot, (1, 1, norb, norb)), (2, 3))\
                    - 1.0 * np.sum(MO_tints * np.reshape(rho_tot, (1, norb, 1, norb)), (1, 3))

print ("Fock matrix dimensions:")
print (Fock_mat.shape)
print (" ")

# these are just to simplify the energy expressions a tiny bit
# \sum_p^{Nocc} h_{pp}
sum_p_nocc_hpp = np.trace(MO_oints[0:nocc, 0:nocc])

# \sum_{pr}^{Nocc} (pp|rr)
sum_pr_nocc_Vpprr = np.trace(np.trace(MO_tints[0:nocc, 0:nocc, 0:nocc, 0:nocc], axis1=2, axis2=3))
# \sum_{pr}^{Nocc} (pr|rp)
sum_pr_nocc_Vprrp = np.trace(np.trace(np.transpose(MO_tints[0:nocc, 0:nocc, 0:nocc, 0:nocc], (0,2,1,3)), axis1=2, axis2=3))

ground_state_energy = 2.0 * sum_p_nocc_hpp + 2.0 * sum_pr_nocc_Vpprr - sum_pr_nocc_Vprrp

# identity matrices of two sizes, necessary for getting the correct matrix dimensions
occ_I = np.reshape(np.identity(nocc), (nocc, 1, nocc, 1))
vir_I = np.reshape(np.identity(nvir), (1, nvir, 1, nvir))

# <phi_i^a| H | phi_j^b>
# term1: hab, term2: hij, term3: E_0 or SCF energy, term4: (ai|jb) term5: (ab|ij), term6: \sum_p (ij|pp)
# term7: \sum_p (ip|pj), term8: \sum_p (ab|pp), term 9: \sum_p (ap|pb)
# the traces are written such that the inner most matrices are done first
CIS_matrix_aa = occ_I * np.reshape(MO_oints[nocc:, nocc:], (1, nvir, 1, nvir)) \
           - vir_I * np.reshape(MO_oints[:nocc, :nocc], (nocc, 1, nocc, 1)) \
           + vir_I * occ_I * ground_state_energy \
           + MO_tints[:nocc, nocc:, :nocc, nocc:] \
           - np.transpose(MO_tints, (1,3,0,2))[:nocc, nocc:, :nocc, nocc:] \
           - 2.0 * vir_I * np.reshape(np.trace(MO_tints[0:nocc, 0:nocc, 0:nocc, 0:nocc], axis1=2, axis2=3), (nocc, 1, nocc, 1)) \
           +       vir_I * np.reshape(np.trace(np.transpose(MO_tints[0:nocc, 0:nocc, 0:nocc, 0:nocc],(0,3,1,2)), axis1=2, axis2=3), (nocc, 1, nocc, 1))\
           + 2.0 * occ_I * occ_I * np.reshape(np.trace(MO_tints[nocc:, nocc:, 0:nocc, 0:nocc], axis1=2, axis2=3), (1, nvir, 1, nvir))\
           -       occ_I * np.reshape(np.trace(np.transpose(MO_tints, (0,3,1,2))[nocc:, nocc:, 0:nocc, 0:nocc], axis1=2, axis2=3), (1, nvir, 1, nvir))

CIS_matrix_aa = np.reshape(CIS_matrix_aa, [nocc*nvir, nocc*nvir])

CIS_matrix_ab = np.reshape(MO_tints[0:nocc, nocc:norb, 0:nocc, nocc:norb], [nocc*nvir, nocc*nvir])

# <phi_0| H |phi_i^a> and <phi_i^a| H |phi_0>
cross_term_hia = MO_oints[:nocc, nocc:] + 2.0 * np.trace(MO_tints[:nocc, nocc:, :nocc, :nocc], axis1=2, axis2=3) - np.trace(np.transpose(MO_tints[:nocc, :nocc, nocc:, :nocc], (0,2,1,3)), axis1=2, axis2=3)

CIS_Energy_mat = np.matmul(np.matmul(np.reshape(Ca, (1, nocc*nvir)), CIS_matrix_aa), np.reshape(Ca, (nocc*nvir, 1)))\
           + np.matmul(np.matmul(np.reshape(Cb, (1, nocc*nvir)), CIS_matrix_aa), np.reshape(Cb, (nocc*nvir, 1)))\
           + np.matmul(np.matmul(np.reshape(Ca, (1, nocc*nvir)), CIS_matrix_ab), np.reshape(Cb, (nocc*nvir, 1)))\
           + np.matmul(np.matmul(np.reshape(Cb, (1, nocc*nvir)), CIS_matrix_ab), np.reshape(Ca, (nocc*nvir, 1)))\
           + C0[0] * C0[0] * (2.0*np.trace(MO_oints_occ) + 2.0*sum_pr_nocc_Vpprr - sum_pr_nocc_Vprrp) \
           + 2.0 * C0[0] * np.sum((Ca + Cb) * cross_term_hia)


# this should technically be one though
wfn_norm = np.sum(np.square(Ca)) + np.sum(np.square(Cb)) + np.sum(np.square(C0))

CIS_Energy_mat = CIS_Energy_mat/wfn_norm + enuc

# Calculating E_0 for the purposes of the perturbative expansion
# \sum_p F_pp, to be used later
sum_p_nocc_fpp = np.trace(Fock_mat[:nocc, :nocc])

# Fab - Fij + \sum_p F_pp + \sum_p F_pp
# First three terms from <\phi_0|F_pq i^\dagger a p^\dagger  q b^\dagger j|\phi_0>
# Last term from <\phi_0|F_pq i^\dagger a \bar{p}^\dagger \bar{q} b^\dagger j|\phi_0>
Fock_matrix_aa = occ_I * np.reshape(Fock_mat[nocc:, nocc:], (1, nvir, 1, nvir))\
               - vir_I * np.reshape(Fock_mat[:nocc, :nocc], (nocc, 1, nocc, 1))\
               + occ_I * vir_I * 2.0 * sum_p_nocc_fpp

Fock_matrix_aa = np.reshape(Fock_matrix_aa, (nocc*nvir, nocc*nvir))

# <\phi_0|F_pq p^\dagger q b^\dagger j|\phi_0>
Fock_cross_term = Fock_mat[:nocc, nocc:]

# 2.0 * the C0*C0 term comes from the alpha p^\dagger q and beta p^\dagger q giving the same thing
# 2.0 * last term: <\phi_0|F_pq P^\dagger q|\phi_i^a> = <\phi_i^a|F_pq p^\dagger q|phi_0>
fockE_mat = np.matmul(np.matmul(np.reshape(Ca, (1, nocc*nvir)), Fock_matrix_aa), np.reshape(Ca, (nocc*nvir, 1)))\
      + np.matmul(np.matmul(np.reshape(Cb, (1, nocc*nvir)), Fock_matrix_aa), np.reshape(Cb, (nocc*nvir, 1)))\
      + C0[0]*C0[0] * 2.0 * np.trace(Fock_mat[:nocc, :nocc])\
      + 2.0*C0[0] * np.sum((Ca + Cb) * Fock_cross_term)


fockE_mat = fockE_mat / wfn_norm

refE = np.sum(CIS_Energy_mat)
refE0 = np.sum(fockE_mat)

# This is for the denominators in the PT2 theory, our 'Fock' matrix is not the normal Fock matrix so there are extra terms to consider.
refE0X = refE0 - 2.0 * np.trace(Fock_mat[:nocc, :nocc])

# For the ground state HF energy
rhfDetEnergy = get_det_energy(enuc, MO_oints, MO_tints, range(nocc), range(nocc))

# for the ground state MP2 energy
mp2Energy = rhfDetEnergy
for i in range(nocc):
    for j in range(nocc):
        for a in range(nocc, norb):
            for b in range(nocc, norb):
                mp2Energy = mp2Energy + (2.0 * MO_tints[i, a, j, b] ** 2 - MO_tints[i, a, j, b] * MO_tints[i, b, j, a])/(Fock_mat[i, i] + Fock_mat[j, j] - Fock_mat[a, a] - Fock_mat[b, b])

print ("MP2ENERGY")
print (mp2Energy)
########################################################################################################################
# ESMF-MP2
########################################################################################################################

########################################################################################################################
# creating the temp variables, initializing tensors, Fotran ordering all tensors
########################################################################################################################

# transforming all matrices to be f ordered:
F_fock_mat = np.copy(Fock_mat, order = 'F')

#print("fock_mat", Fock_mat)

fock_oo = np.copy(Fock_mat[:nocc, :nocc])
fock_ov = np.copy(Fock_mat[:nocc, nocc:])
fock_vo = np.copy(Fock_mat[nocc:, :nocc])
fock_vv = np.copy(Fock_mat[nocc:, nocc:])

f_oo = np.copy(fock_oo, order = 'F')
f_ov = np.copy(fock_ov, order = 'F')
f_vo = np.copy(fock_vo, order = 'F')
f_vv = np.copy(fock_vv, order = 'F')

# to_2 = np.zeros((nvir, nocc), order = 'F')
# to_4 = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
F_MO_oints = np.copy(MO_oints, order = 'F')
F_MO_tints = np.copy(MO_tints, order = 'F')

#C0_coeff = C0[0]
#for testing purposes: (makes RHS overlap matrix elements become 0 for all excitations)
C0_coeff = 0.0000
C0_coeff_2 = C0_coeff**2
Ca_coeff = np.copy(tCa, order = 'F')
Cb_coeff = np.copy(tCb, order = 'F')

#print("Ca_coeff: ", Ca_coeff)

Cstart = 1
Cend = nocc
Vstart = 1
Vend = nvir
Hstart = 1
Hend = norb

# traces and temp variables I need for the FORTRAN modules
# the temps will just be copied when used as parameters for the modules so I don't need
# 400 temp variables declared at the beginning of the program
int_temp = 0.0

temp_vv = np.zeros((nvir, nvir), order = 'F')
temp_ov = np.zeros((nocc, nvir), order = 'F')
temp_vo = np.zeros((nvir, nocc), order = 'F')
temp_oo = np.zeros((nocc, nocc), order = 'F')

temp_vvvv = np.zeros((nvir, nvir, nvir, nvir), order = 'F')
temp_vvvo = np.zeros((nvir, nvir, nvir, nocc), order = 'F')
temp_vvov = np.zeros((nvir, nvir, nocc, nvir), order = 'F')
temp_vovv = np.zeros((nvir, nocc, nvir, nvir), order = 'F')
temp_ovvv = np.zeros((nocc, nvir, nvir, nvir), order = 'F')
temp_vvoo = np.zeros((nvir, nvir, nocc, nocc), order = 'F')
temp_vovo = np.zeros((nvir, nocc, nvir, nocc), order = 'F')
temp_ovvo = np.zeros((nocc, nvir, nvir, nocc), order = 'F')
temp_voov = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
temp_ovov = np.zeros((nocc, nvir, nocc, nvir), order = 'F')
temp_oovv = np.zeros((nocc, nocc, nvir, nvir), order = 'F')
temp_vooo = np.zeros((nvir, nocc, nocc, nocc), order = 'F')
temp_ovoo = np.zeros((nocc, nvir, nocc, nocc), order = 'F')
temp_oovo = np.zeros((nocc, nocc, nvir, nocc), order = 'F')
temp_ooov = np.zeros((nocc, nocc, nocc, nvir), order = 'F')
temp_oooo = np.zeros((nocc, nocc, nocc, nocc), order = 'F')

tf_oo = np.trace(fock_oo)
tf_ov = np.trace(fock_ov)
tf_vo = np.trace(fock_vo)
tf_vv = np.trace(fock_vv)

F_g_oooo = np.copy(MO_tints[:nocc, :nocc, :nocc, :nocc], order = 'F')
F_g_ooov = np.copy(MO_tints[:nocc, :nocc, :nocc, nocc:], order = 'F')
F_g_oovo = np.copy(MO_tints[:nocc, :nocc, nocc:, :nocc], order = 'F')
F_g_ovoo = np.copy(MO_tints[:nocc, nocc:, :nocc, :nocc], order = 'F')
F_g_vooo = np.copy(MO_tints[nocc:, :nocc, :nocc, :nocc], order = 'F')
F_g_oovv = np.copy(MO_tints[:nocc, :nocc, nocc:, nocc:], order = 'F')
F_g_ovov = np.copy(MO_tints[:nocc, nocc:, :nocc, nocc:], order = 'F')
F_g_voov = np.copy(MO_tints[nocc:, :nocc, :nocc, nocc:], order = 'F')
F_g_ovvo = np.copy(MO_tints[:nocc, nocc:, nocc:, :nocc], order = 'F')
F_g_vovo = np.copy(MO_tints[nocc:, :nocc, nocc:, :nocc], order = 'F')
F_g_vvoo = np.copy(MO_tints[nocc:, nocc:, :nocc, :nocc], order = 'F')
F_g_ovvv = np.copy(MO_tints[:nocc, nocc:, nocc:, nocc:], order = 'F')
F_g_vovv = np.copy(MO_tints[nocc:, :nocc, nocc:, nocc:], order = 'F')
F_g_vvov = np.copy(MO_tints[nocc:, nocc:, :nocc, nocc:], order = 'F')
F_g_vvvo = np.copy(MO_tints[nocc:, nocc:, nocc:, :nocc], order = 'F')
F_g_vvvv = np.copy(MO_tints[nocc:, nocc:, nocc:, nocc:], order = 'F')

g_oooo = np.copy(MO_tints[:nocc, :nocc, :nocc, :nocc])
g_ooov = np.copy(MO_tints[:nocc, :nocc, :nocc, nocc:])
g_oovo = np.copy(MO_tints[:nocc, :nocc, nocc:, :nocc])
g_ovoo = np.copy(MO_tints[:nocc, nocc:, :nocc, :nocc])
g_vooo = np.copy(MO_tints[nocc:, :nocc, :nocc, :nocc])
g_oovv = np.copy(MO_tints[:nocc, :nocc, nocc:, nocc:])
g_ovov = np.copy(MO_tints[:nocc, nocc:, :nocc, nocc:])
g_voov = np.copy(MO_tints[nocc:, :nocc, :nocc, nocc:])
g_ovvo = np.copy(MO_tints[:nocc, nocc:, nocc:, :nocc])
g_vovo = np.copy(MO_tints[nocc:, :nocc, nocc:, :nocc])
g_vvoo = np.copy(MO_tints[nocc:, nocc:, :nocc, :nocc])
g_ovvv = np.copy(MO_tints[:nocc, nocc:, nocc:, nocc:])
g_vovv = np.copy(MO_tints[nocc:, :nocc, nocc:, nocc:])
g_vvov = np.copy(MO_tints[nocc:, nocc:, :nocc, nocc:])
g_vvvo = np.copy(MO_tints[nocc:, nocc:, nocc:, :nocc])
g_vvvv = np.copy(MO_tints[nocc:, nocc:, nocc:, nocc:])

tg_jjkb = np.copy(np.trace(g_ooov, axis1 = 0, axis2 = 1), order = 'F')
tg_jkjb = np.copy(np.trace(g_ooov, axis1 = 0, axis2 = 2), order = 'F')
tg_jbkk = np.copy(np.trace(g_ovoo, axis1 = 2, axis2 = 3), order = 'F')
tg_jkbk = np.copy(np.trace(g_oovo, axis1 = 1, axis2 = 3), order = 'F')
tg_iija = np.copy(np.trace(g_ooov, axis1 = 0, axis2 = 1), order = 'F')
tg_iiaj = np.copy(np.trace(g_oovo, axis1 = 0, axis2 = 1), order = 'F')
tg_ijia = np.copy(np.trace(g_ooov, axis1 = 0, axis2 = 2), order = 'F')
tg_iaij = np.copy(np.trace(g_ovoo, axis1 = 0, axis2 = 2), order = 'F')
tg_ijjk = np.copy(np.trace(g_oooo, axis1 = 1, axis2 = 2), order = 'F')
tg_ijkk = np.copy(np.trace(g_oooo, axis1 = 2, axis2 = 3), order = 'F')
tg_ajjb = np.copy(np.trace(g_voov, axis1 = 1, axis2 = 2), order = 'F')
tg_abjj = np.copy(np.trace(g_vvoo, axis1 = 2, axis2 = 3), order = 'F')
tg_jjbk = np.copy(np.trace(g_oovo, axis1 = 0, axis2 = 1), order = 'F')
tg_jbjk = np.copy(np.trace(g_ovoo, axis1 = 0, axis2 = 2), order = 'F')
tg_jjkk = np.trace(np.trace(g_oooo, axis1 = 0, axis2 = 1))
tg_jkjk = np.trace(np.trace(g_oooo, axis1 = 0, axis2 = 2))

# t2_aa = np.zeros((nvir, nocc), order = 'F')
# t2_bb = np.zeros((nvir, nocc), order = 'F')
# t4_aaaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
# t4_aabb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
# t4_bbaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
# t4_bbbb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')

########################################################################################################################
# ESMF MP2 functions (restricted, C0=0 currently)
########################################################################################################################
# set this boolean to True if you want to include the bbaa terms
# they are excluded as they should just cause over counting as they are the same as the aabb terms?

include_bbaa = False
include_025_RHS = False

# making use of the symmetry of the rank-4 aaaa or bbbb tensors, should aid in convergence 
def symmetrize_aijb_aaaa(t):
    for ii in range(t.shape[0]):
        for aa in range(t.shape[1]):
            for bb in range(aa+1, t.shape[2]):
                for jj in range(ii+1, t.shape[3]):
                    x = t[ii, aa, bb, jj] - t[jj, aa, bb, ii] - t[ii, bb, aa, jj] + t[jj, bb, aa, ii]
                    t[ii, aa, bb, jj] = 0.25 * x
                    t[jj, aa, bb, ii] = -0.25 * x
                    t[ii, bb, aa, jj] = -0.25 * x
                    t[jj, bb, aa, ii] = 0.25 * x

def pack_pt2_amps(array_of_tensors):
    # this method takes an array of tensors and converts them to a vector
    # the created vectors are then concatenated together to return one long vector
    # also returns an array with arrays of the dimensions of each of the tensors fed in
    # returns the packed array and an array containing the dimensions of the original tensors in the order
    #   that the original tensors were read into the function
    # for this code the order is assumed to be aa, bb, aaaa, aabb, bbaa, bbbb where applicable
    dimensions = [array_of_tensors[0].shape]
    temp = np.reshape(array_of_tensors[0], [array_of_tensors[0].size, 1])

    for ii in range(len(array_of_tensors)-1):
        temp = np.concatenate([temp, np.reshape(array_of_tensors[ii+1], [array_of_tensors[ii+1].size, 1])], 0)
        dimensions.append(array_of_tensors[ii+1].shape)
    return temp, dimensions

# returns a 1-d array version of the inputted tensors
# the dimensions inputted here should be the same as the ones returned by the pack_pt2_amps function
def unpack_pt2_amps(dimensions, packed_t):
    unpacked_t = []
    m = 0
    for dimens in dimensions:
        length = 1
        for dim in dimens:
            length = length * dim
        unpacked_t.append(np.reshape(1.0 * packed_t[m:m+length, 0], dimens))
        m = m + length
    return unpacked_t

# def prepare_aaaa_guess(no, nv, x):
#     t = np.zeros((nv, no, no, nv), order = 'F')
#     for i in range(t.shape[0]):
#         for j in range(t.shape[1]):
#             for k in range(j+1, t.shape[2]):
#                 for l in range(i+1, t.shape[3]):
#                     t[i, j, k, l]=4.0*x
#     symmetrize_aijb_aaaa(t)
#     return t

def pt2_lin_trans(nvir, nocc, norb, Fock_mat, Ca_coeff, C0_coeff, packed_t, dimensions):
    # <I|F t(orbitals involved in the excitations) H0 excitations off of the reference wave function |\Phi_0>

    # constants used in the fortran codes, fortran starts counting at 1, not 0
    # I decided to just split the fock and two-electron tensors into their occ or vir components
    # however, being clever with these values could have been done instead (have Vstart be nocc+1 and Vend be norb)
    Cstart = 1
    Vstart = 1
    Hstart = 1
    Cend = nocc
    Vend = nvir
    Hend = norb

    f = 1.0 * fockE_mat
    
    C0_coeff_2 = C0_coeff ** 2
    Cb_coeff = np.copy(tCb, order='F')

    F_fock_mat = np.copy(Fock_mat, order='F')

    # splitting the ESMF Fock matrix into it's occupied and virtual portions
    fock_oo = np.copy(Fock_mat[:nocc, :nocc])
    fock_ov = np.copy(Fock_mat[:nocc, nocc:])
    fock_vo = np.copy(Fock_mat[nocc:, :nocc])
    fock_vv = np.copy(Fock_mat[nocc:, nocc:])

    f_oo = np.copy(fock_oo, order='F')
    f_ov = np.copy(fock_ov, order='F')
    f_vo = np.copy(fock_vo, order='F')
    f_vv = np.copy(fock_vv, order='F')

    # initializing the ouput vectors
    to_2_aa = np.zeros((nvir, nocc), order = 'F')
    to_2_bb = np.zeros((nvir, nocc), order = 'F')
    to_4_aaaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    to_4_aabb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    if include_bbaa:
        to_4_bbaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    to_4_bbbb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')

    # for the values/tensors that need to be stored in the tensor contractions when there are broken maps
    int_temp = 0

    temp_vv = np.zeros((nvir, nvir), order='F')
    temp_ov = np.zeros((nocc, nvir), order='F')
    temp_vo = np.zeros((nvir, nocc), order='F')
    temp_oo = np.zeros((nocc, nocc), order='F')

    temp_vvvv = np.zeros((nvir, nvir, nvir, nvir), order='F')
    temp_vvvo = np.zeros((nvir, nvir, nvir, nocc), order='F')
    temp_vvov = np.zeros((nvir, nvir, nocc, nvir), order='F')
    temp_vovv = np.zeros((nvir, nocc, nvir, nvir), order='F')
    temp_ovvv = np.zeros((nocc, nvir, nvir, nvir), order='F')
    temp_vvoo = np.zeros((nvir, nvir, nocc, nocc), order='F')
    temp_vovo = np.zeros((nvir, nocc, nvir, nocc), order='F')
    temp_ovvo = np.zeros((nocc, nvir, nvir, nocc), order='F')
    temp_voov = np.zeros((nvir, nocc, nocc, nvir), order='F')
    temp_ovov = np.zeros((nocc, nvir, nocc, nvir), order='F')
    temp_oovv = np.zeros((nocc, nocc, nvir, nvir), order='F')
    temp_vooo = np.zeros((nvir, nocc, nocc, nocc), order='F')
    temp_ovoo = np.zeros((nocc, nvir, nocc, nocc), order='F')
    temp_oovo = np.zeros((nocc, nocc, nvir, nocc), order='F')
    temp_ooov = np.zeros((nocc, nocc, nocc, nvir), order='F')
    temp_oooo = np.zeros((nocc, nocc, nocc, nocc), order='F')

    # Fock matrix traces
    tf_oo = np.trace(fock_oo)
    tf_ov = np.trace(fock_ov)
    tf_vo = np.trace(fock_vo)
    tf_vv = np.trace(fock_vv)

    # reading in the excitation constants for the excitations acting to the left in the matrix element
    if include_bbaa:
        t2_aa, t2_bb, t4_aaaa, t4_aabb, t4_bbaa, t4_bbbb = unpack_pt2_amps(dimensions, packed_t)
    else:
        t2_aa, t2_bb, t4_aaaa, t4_aabb, t4_bbbb = unpack_pt2_amps(dimensions, packed_t)
    symmetrize_aijb_aaaa(t4_aaaa)
    symmetrize_aijb_aaaa(t4_bbbb)
    
    #functions producing the aa excitation constants
    r_aa_to_aa.aa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(temp_vv), np.copy(int_temp),
                    np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(temp_oo),
                    C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_aa, tf_oo, tf_oo, tf_oo, tf_oo,
                    tf_oo, tf_oo, to_2_aa)
    r_aa_to_aaaa.aa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(int_temp),
                        np.copy(int_temp), np.copy(temp_ov), np.copy(int_temp), C0_coeff, C0_coeff_2, Ca_coeff,
                        Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_aaaa, tf_oo, to_2_aa)
    r_aa_to_aabb.aa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(temp_vo), np.copy(temp_vo),
                        np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), C0_coeff, C0_coeff_2, Ca_coeff,
                        Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_aabb, tf_oo, to_2_aa)
    r_aa_to_bb.aa_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(int_temp), C0_coeff,
                        Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_bb, tf_oo, to_2_aa)
    if include_bbaa:
        r_aa_to_bbaa.aa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(temp_ov),
                            np.copy(int_temp), np.copy(temp_ov), np.copy(temp_ov), np.copy(int_temp), C0_coeff, C0_coeff_2,
                            Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_bbaa, tf_oo, to_2_aa)
    r_aa_to_bbbb.aa_to_bbbb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), Ca_coeff, Ca_coeff, f_ov, t4_bbbb, to_2_aa)

    #functions producing the bb excitation constants
    r_aa_to_aa.aa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(temp_vv), np.copy(int_temp),
                    np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(temp_oo),
                    C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_bb, tf_oo, tf_oo, tf_oo, tf_oo,
                    tf_oo, tf_oo, to_2_bb)
    r_aa_to_aaaa.aa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(int_temp),
                        np.copy(int_temp), np.copy(temp_ov), np.copy(int_temp), C0_coeff, C0_coeff_2, Ca_coeff,
                        Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_bbbb, tf_oo, to_2_bb)
    if include_bbaa:
        r_aa_to_aabb.aa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(temp_vo), np.copy(temp_vo),
                            np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), C0_coeff, C0_coeff_2, Ca_coeff,
                            Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_bbaa, tf_oo, to_2_bb)
    r_aa_to_bb.aa_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(int_temp), C0_coeff,
                        Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_aa, tf_oo, to_2_bb)
    r_aa_to_bbaa.aa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(temp_ov),
                        np.copy(int_temp), np.copy(temp_ov), np.copy(temp_ov), np.copy(int_temp), C0_coeff, C0_coeff_2,
                        Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_aabb, tf_oo, to_2_bb)
    r_aa_to_bbbb.aa_to_bbbb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), Ca_coeff, Ca_coeff, f_ov, t4_aaaa, to_2_bb)

    #functions producing the aaaa excitation constants
    r_aaaa_to_aa.aaaa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vv), np.copy(temp_oo),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_vv), np.copy(temp_oo), np.copy(temp_vv),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_vv),
                            np.copy(temp_oo), np.copy(temp_oo), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov,
                            f_vo, f_vv, t2_aa, tf_oo, tf_oo, tf_oo, tf_oo, to_4_aaaa)
    r_aaaa_to_aaaa.aaaa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_voov), np.copy(temp_voov),
                                np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_ov),
                                np.copy(temp_vovo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo),
                                np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vovo), np.copy(temp_vo),
                                np.copy(temp_vo),np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vooo),
                                np.copy(temp_vovv), np.copy(temp_vovv), np.copy(temp_vooo), np.copy(temp_voov),
                                np.copy(temp_vo), np.copy(temp_voov), np.copy(temp_vovo), np.copy(temp_vovo),
                                np.copy(temp_ov), np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                                f_ov, f_vo, f_vv, t4_aaaa, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo,
                                tf_oo, tf_oo, tf_oo, to_4_aaaa)
    # r_aaaa_to_aaaa_1.aaaa_to_aaaa_1(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp),np.copy(int_temp), np.copy(temp_ov),
    #                                 np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_vo), np.copy(int_temp),
    #                                 np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_ov),
    #                                 np.copy(int_temp), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo),
    #                                 np.copy(temp_vo), np.copy(temp_vvoo), np.copy(temp_ovov), np.copy(temp_ovov),
    #                                 np.copy(temp_vvoo), np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_vo),
    #                                 C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv,
    #                                 t4_aaaa, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, to_4_aaaa)
    # r_aaaa_to_aaaa_2.aaaa_to_aaaa_2(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_voov), np.copy(temp_voov),
    #                                 np.copy(temp_ovov), np.copy(temp_ovov), C0_coeff, C0_coeff_2, Ca_coeff,
    #                                 Ca_coeff, f_oo, f_ov, f_vo, f_vv, to_4_aaaa, tf_oo, tf_oo, to_4_aaaa)
    r_aaaa_to_aabb.aaaa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(temp_vo),
                            np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo),
                            np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_ov),
                            np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_vo),
                            np.copy(temp_vo), np.copy(temp_vo), C0_coeff, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv,
                            t4_aabb, tf_oo, tf_oo, tf_oo, tf_oo, to_4_aaaa)
    r_aaaa_to_bb.aaaa_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                            f_ov, f_vo, f_vv, t2_bb, to_4_aaaa)
    if include_bbaa:
        r_aaaa_to_bbaa.aaaa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_ov), np.copy(temp_ov),
                                    np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_vo),
                                    np.copy(temp_ov), np.copy(temp_ov), np.copy(temp_ov), np.copy(temp_ov),
                                    np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_ov),
                                    np.copy(temp_vo), np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov,
                                    f_vo, f_vv, t4_bbaa, tf_oo, tf_oo, tf_oo, tf_oo, to_4_aaaa)
    
    # functions producing the bbbb exciation constants
    r_aaaa_to_aa.aaaa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vv), np.copy(temp_oo),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_vv), np.copy(temp_oo), np.copy(temp_vv),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_vv),
                            np.copy(temp_oo), np.copy(temp_oo), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov,
                            f_vo, f_vv, t2_bb, tf_oo, tf_oo, tf_oo, tf_oo, to_4_bbbb)
    r_aaaa_to_aaaa.aaaa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_voov), np.copy(temp_voov),
                                np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_ov),
                                np.copy(temp_vovo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo),
                                np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vovo), np.copy(temp_vo),
                                np.copy(temp_vo),np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vooo),
                                np.copy(temp_vovv), np.copy(temp_vovv), np.copy(temp_vooo), np.copy(temp_voov),
                                np.copy(temp_vo), np.copy(temp_voov), np.copy(temp_vovo), np.copy(temp_vovo),
                                np.copy(temp_ov), np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                                f_ov, f_vo, f_vv, t4_bbbb, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo,
                                tf_oo, tf_oo, tf_oo, to_4_bbbb)
    # r_aaaa_to_aaaa_1.aaaa_to_aaaa_1(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), np.copy(temp_ov),
    #                                 np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_vo), np.copy(int_temp),
    #                                 np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_ov),
    #                                 np.copy(int_temp), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo),
    #                                 np.copy(temp_vo), np.copy(temp_vvoo), np.copy(temp_ovov), np.copy(temp_ovov),
    #                                 np.copy(temp_vvoo), np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_vo),
    #                                 C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv,
    #                                 t4_bbbb, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, to_4_bbbb)
    # r_aaaa_to_aaaa_2.aaaa_to_aaaa_2(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_voov), np.copy(temp_voov),
    #                                 np.copy(temp_ovov), np.copy(temp_ovov), C0_coeff, C0_coeff_2, Ca_coeff,
    #                                 Ca_coeff, f_oo, f_ov, f_vo, f_vv, to_4_bbbb, tf_oo, tf_oo, to_4_bbbb)
    if include_bbaa:
        r_aaaa_to_aabb.aaaa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vo), np.copy(temp_vo),
                                    np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_vo),
                                    np.copy(temp_vo), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_ov),
                                    np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_vo),
                                    np.copy(temp_vo), np.copy(temp_vo), C0_coeff, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo,
                                    f_vv, t4_bbaa, tf_oo, tf_oo, tf_oo, tf_oo, to_4_bbbb)
    r_aaaa_to_bb.aaaa_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                            f_ov, f_vo, f_vv, t2_aa, to_4_bbbb)
    # r_aaaa_to_bbaa.aaaa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_ov), np.copy(temp_ov),
    #                             np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_vo),
    #                             np.copy(temp_ov), np.copy(temp_ov), np.copy(temp_ov), np.copy(temp_ov),
    #                             np.copy(temp_ov), np.copy(temp_vo), np.copy(temp_ov), np.copy(temp_ov),
    #                             np.copy(temp_vo), np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov,
    #                             f_vo, f_vv, t4_aabb, tf_oo, tf_oo, tf_oo, tf_oo, to_4_bbbb)

    #functions producing the aabb excitation constants
    r_aabb_to_aa.aabb_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vv), np.copy(temp_vv),
                            np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_vv), np.copy(temp_oo), C0_coeff,
                            C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_aa, tf_oo, to_4_aabb)
    r_aabb_to_aaaa.aabb_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_ov), np.copy(temp_ovov), np.copy(temp_vo),
                   np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_aaaa, tf_oo, to_4_aabb)
    r_aabb_to_aabb.aabb_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_oovv), np.copy(temp_ov),
                                np.copy(temp_ovov), np.copy(temp_ovov), np.copy(temp_vovv), np.copy(temp_oovo),
                                np.copy(temp_oovv), np.copy(temp_ov), np.copy(temp_voov), np.copy(temp_vo),
                                np.copy(temp_ovov), np.copy(temp_ov), np.copy(temp_ovov), np.copy(temp_ovov),
                                np.copy(temp_ov), np.copy(temp_vooo), np.copy(temp_vovv), np.copy(temp_voov),
                                np.copy(temp_ov), np.copy(temp_ovov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                                f_ov, f_vo, f_vv, t4_aabb, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo,
                                to_4_aabb)
    r_aabb_to_bb.aabb_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vv), np.copy(temp_ov),
                            np.copy(temp_oo), np.copy(temp_vv), np.copy(temp_ov), np.copy(temp_vo), C0_coeff, C0_coeff_2,
                            Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_bb, tf_oo, to_4_aabb)
    if include_bbaa:
        r_aabb_to_bbaa.aabb_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(temp_vo),
                                    np.copy(int_temp), np.copy(temp_vo), np.copy(temp_vvoo), np.copy(temp_ovov),
                                    np.copy(int_temp), np.copy(temp_ov), np.copy(int_temp), np.copy(temp_ov),
                                    np.copy(int_temp), np.copy(temp_vo), np.copy(temp_ov), np.copy(int_temp), np.copy(temp_ov),
                                    np.copy(temp_ovov), np.copy(temp_voov), np.copy(int_temp), np.copy(temp_vo),
                                    np.copy(int_temp), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv,
                                    t4_bbaa, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, to_4_aabb)
    r_aabb_to_bbbb.aabb_to_bbbb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vvoo), np.copy(temp_ovov),
                                np.copy(temp_vo), np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                                f_ov, f_vo, f_vv, t4_bbbb, tf_oo,to_4_aabb)

    #functions that produce the bbaa excitation constants
    if include_bbaa:
        r_aabb_to_aa.aabb_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vv), np.copy(temp_vv),
                                np.copy(temp_oo), np.copy(temp_oo), np.copy(temp_vv), np.copy(temp_oo), C0_coeff,
                                C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_bb, tf_oo, to_4_bbaa)
        r_aabb_to_aaaa.aabb_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_ov), np.copy(temp_ovov), np.copy(temp_vo),
                       np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t4_bbbb, tf_oo, to_4_bbaa)
        r_aabb_to_aabb.aabb_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_oovv), np.copy(temp_ov),
                                    np.copy(temp_ovov), np.copy(temp_ovov), np.copy(temp_vovv), np.copy(temp_oovo),
                                    np.copy(temp_oovv), np.copy(temp_ov), np.copy(temp_voov), np.copy(temp_vo),
                                    np.copy(temp_ovov), np.copy(temp_ov), np.copy(temp_ovov), np.copy(temp_ovov),
                                    np.copy(temp_ov), np.copy(temp_vooo), np.copy(temp_vovv), np.copy(temp_voov),
                                    np.copy(temp_ov), np.copy(temp_ovov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                                    f_ov, f_vo, f_vv, t4_bbaa, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo,
                                    to_4_bbaa)
        r_aabb_to_bb.aabb_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vv), np.copy(temp_ov),
                                np.copy(temp_oo), np.copy(temp_vv), np.copy(temp_ov), np.copy(temp_vo), C0_coeff, C0_coeff_2,
                                Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, t2_aa, tf_oo, to_4_bbaa)
        r_aabb_to_bbaa.aabb_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(temp_vo),
                                    np.copy(int_temp), np.copy(temp_vo), np.copy(temp_vvoo), np.copy(temp_ovov),
                                    np.copy(int_temp), np.copy(temp_ov), np.copy(int_temp), np.copy(temp_ov),
                                    np.copy(int_temp), np.copy(temp_vo), np.copy(temp_ov), np.copy(int_temp), np.copy(temp_ov),
                                    np.copy(temp_ovov), np.copy(temp_voov), np.copy(int_temp), np.copy(temp_vo),
                                    np.copy(int_temp), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv,
                                    t4_aabb, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, tf_oo, to_4_bbaa)
        r_aabb_to_bbbb.aabb_to_bbbb(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(temp_vvoo), np.copy(temp_ovov),
                                    np.copy(temp_vo), np.copy(temp_ov), C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff, f_oo,
                                    f_ov, f_vo, f_vv, t4_aaaa, tf_oo,to_4_bbaa)

    symmetrize_aijb_aaaa(to_4_aaaa)
    symmetrize_aijb_aaaa(to_4_bbbb)
    if include_bbaa:
        to_be_returned, dims = pack_pt2_amps([to_2_aa, to_2_bb, to_4_aaaa, to_4_aabb, to_4_bbaa, to_4_bbbb])
    else:
        to_be_returned, dims = pack_pt2_amps([to_2_aa, to_2_bb, to_4_aaaa, to_4_aabb, to_4_bbbb])
    
    return to_be_returned

# print("here")

def lhs_ovrlps(nvir, nocc, norb, Ca_coeff, C0_coeff, packed_t, dimensions, ref_E0):

    # -E0 * <I| (single and double excitations with constants denoted by the t's)|\Phi_0>

    # start and end points of the tensors
    Cstart = 1
    Vstart = 1
    Hstart = 1
    Cend = nocc
    Vend = nvir
    Hend = norb

    # needed as an input for the Fotran modules
    C0_coeff_2 = C0_coeff * C0_coeff

    # initializing the output tensors
    to_2_aa = np.zeros((nvir, nocc), order = 'F')
    to_2_bb = np.zeros((nvir, nocc), order = 'F')
    to_4_aaaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    to_4_aabb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    if include_bbaa:
        to_4_bbaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    to_4_bbbb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')

    # getting the input tensors
    if include_bbaa:
        t2_aa, t2_bb, t4_aaaa, t4_aabb, t4_bbaa, t4_bbbb = unpack_pt2_amps(dimensions, packed_t)
    else:
        t2_aa, t2_bb, t4_aaaa, t4_aabb, t4_bbbb = unpack_pt2_amps(dimensions, packed_t)
    symmetrize_aijb_aaaa(t4_aaaa)
    symmetrize_aijb_aaaa(t4_bbbb)

    #print("t2_aa: ", t2_aa)

    # functions that produce aa excitation constants
    r_ovrlp_aa_to_aa.ovrlp_aa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff,
                                    t2_aa, to_2_aa)
    r_ovrlp_aa_to_aaaa.ovrlp_aa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff,
                                        t4_aaaa, to_2_aa)
    r_ovrlp_aa_to_aabb.ovrlp_aa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff,
                                        t4_aabb, to_2_aa)
    r_ovrlp_aa_to_bb.ovrlp_aa_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff,
                                    t2_bb, to_2_aa)
    if include_bbaa:
        r_ovrlp_aa_to_bbaa.ovrlp_aa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff,
                                            t4_bbaa, to_2_aa)

    #functions that produce bb excitation constants
    r_ovrlp_aa_to_aa.ovrlp_aa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff,
                                    t2_bb, to_2_bb)
    r_ovrlp_aa_to_aaaa.ovrlp_aa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff,
                                        Ca_coeff, t4_bbbb, to_2_bb)
    if include_bbaa:
        r_ovrlp_aa_to_aabb.ovrlp_aa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                            Ca_coeff, t4_bbaa, to_2_bb)
    r_ovrlp_aa_to_bb.ovrlp_aa_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff, Ca_coeff,
                                    t2_aa, to_2_bb)
    r_ovrlp_aa_to_bbaa.ovrlp_aa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                        Ca_coeff,t4_aabb, to_2_bb)

    #functions that produce aaaa excitation constants
    r_ovrlp_aaaa_to_aa.ovrlp_aaaa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff, t2_aa,
                                        to_4_aaaa)
    r_ovrlp_aaaa_to_aaaa.ovrlp_aaaa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                            Ca_coeff, t4_aaaa, to_4_aaaa)
    r_ovrlp_aaaa_to_aabb.ovrlp_aaaa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_aabb,
                                            to_4_aaaa)
    if include_bbaa:
        r_ovrlp_aaaa_to_bbaa.ovrlp_aaaa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_bbaa,
                                                to_4_aaaa)

    # functions that produce bbbb excitation constatns
    r_ovrlp_aaaa_to_aa.ovrlp_aaaa_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff, t2_bb,
                                        to_4_bbbb)
    r_ovrlp_aaaa_to_aaaa.ovrlp_aaaa_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                            Ca_coeff, t4_bbbb, to_4_bbbb)
    if include_bbaa:
        r_ovrlp_aaaa_to_aabb.ovrlp_aaaa_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_bbaa,
                                                to_4_bbbb)
    r_ovrlp_aaaa_to_bbaa.ovrlp_aaaa_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_aabb,
                                            to_4_bbbb)

    # functions that produce aabb excitation constants
    r_ovrlp_aabb_to_aa.ovrlp_aabb_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff, t2_aa,
                                        to_4_aabb)
    r_ovrlp_aabb_to_aaaa.ovrlp_aabb_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_aaaa,
                                            to_4_aabb)
    r_ovrlp_aabb_to_aabb.ovrlp_aabb_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                            Ca_coeff, t4_aabb, to_4_aabb)
    r_ovrlp_aabb_to_bb.ovrlp_aabb_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff, t2_bb,
                                        to_4_aabb)
    if include_bbaa:
        r_ovrlp_aabb_to_bbaa.ovrlp_aabb_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                                Ca_coeff, t4_bbaa, to_4_aabb)
    r_ovrlp_aabb_to_bbbb.ovrlp_aabb_to_bbbb(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_bbbb,
                                             to_4_aabb)

    #functions that produce bbaa excitation constants
    if include_bbaa:
        r_ovrlp_aabb_to_aa.ovrlp_aabb_to_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff, t2_bb,
                                            to_4_bbaa)
        r_ovrlp_aabb_to_aaaa.ovrlp_aabb_to_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_bbbb,
                                                to_4_bbaa)
        r_ovrlp_aabb_to_aabb.ovrlp_aabb_to_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                                Ca_coeff, t4_bbaa, to_4_bbaa)
        r_ovrlp_aabb_to_bb.ovrlp_aabb_to_bb(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, Ca_coeff, Ca_coeff, t2_aa,
                                            to_4_bbaa)
        r_ovrlp_aabb_to_bbaa.ovrlp_aabb_to_bbaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, C0_coeff, C0_coeff_2, Ca_coeff,
                                                Ca_coeff, t4_aabb, to_4_bbaa)
        r_ovrlp_aabb_to_bbbb.ovrlp_aabb_to_bbbb(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, t4_aaaa,
                                                 to_4_bbaa)

    # making use of the symmetries the all-alpha and all-beta rank-4 tensors should have
    symmetrize_aijb_aaaa(to_4_aaaa)
    symmetrize_aijb_aaaa(to_4_bbbb)

    # multiplying these elements by -E0 as the lhs lin trans terms are positive in this code
    to_2_aa = -0.5 * ref_E0 * to_2_aa
    to_2_bb = -0.5 * ref_E0 * to_2_bb
    to_4_aaaa = -0.5 * ref_E0 * to_4_aaaa
    to_4_aabb = -0.5 * ref_E0 * to_4_aabb
    if include_bbaa:
        to_4_bbaa = -0.5 * ref_E0 * to_4_bbaa
    to_4_bbbb = -0.5 * ref_E0 * to_4_bbbb

    #to_2_aa = -1.0 * refE0X * to_2_aa
    #to_2_bb = -1.0 * refE0X * to_2_bb
    #to_4_aaaa = -1.0 * refE0X * to_4_aaaa

    #to_4_aabb = -1.0 * refE0X * to_4_aabb
    #to_4_bbaa = -1.0 * refE0X * to_4_bbaa
    #to_4_bbbb = -1.0 * refE0X * to_4_bbbb

    if include_bbaa:
        return pack_pt2_amps([to_2_aa, to_2_bb, to_4_aaaa, to_4_aabb, to_4_bbaa, to_4_bbbb])
    else:
        return pack_pt2_amps([to_2_aa, to_2_bb, to_4_aaaa, to_4_aabb, to_4_bbbb])

def rhs_outputs(num_occ, num_vir, num_orbs, matC, oei, tei, tei_trace_pqii, tei_trace_piqi):
    
    # -<I|V|\Phi_0> = -<I|H|\Phi_0> when C0==0

    # These functions were done without needing to split the Fock matrix by being careful with the start and end points
    # used in each contracted elements

    # initializing the output tensors
    rhs_aa = np.zeros((num_occ, num_vir), order = 'F')
    rhs_aaaa = np.zeros((num_occ, num_vir, num_occ, num_vir), order = 'F')
    rhs_aabb = np.zeros((num_occ, num_vir, num_occ, num_vir), order = 'F')
    rhs_bb = np.zeros((num_occ, num_vir), order='F')
    rhs_bbbb = np.zeros((num_occ, num_vir, num_occ, num_vir), order='F')
    if include_bbaa:
        rhs_bbaa = np.zeros((num_occ, num_vir, num_occ, num_vir), order='F')
    check = np.zeros((3), order = 'F')

    # creating the necessary traces
    #tei_trace_pqii = np.copy(np.trace(tei[:, :, :num_occ, :num_occ], axis1 = 2, axis2 = 3), order = 'F')
    #tei_trace_piqi = np.copy(np.trace(tei[:, :num_occ, :, :num_occ], axis1 = 1, axis2 = 3), order = 'F')

    r_rhs_aa.rhs_aa_sub(num_occ+1, matC, oei, tei, tei_trace_pqii, tei_trace_piqi, rhs_aa, [num_occ, num_orbs])

    r_rhs_aaaa.rhs_aaaa_sub(num_occ+1, matC, tei, rhs_aaaa, [num_occ, num_orbs])

    r_rhs_aabb.rhs_aabb_sub(num_occ+1, matC, tei, rhs_aabb)

    r_rhs_aa.rhs_aa_sub(num_occ + 1, matC, oei, tei, tei_trace_pqii, tei_trace_piqi, rhs_bb, [num_occ, num_orbs])

    r_rhs_aaaa.rhs_aaaa_sub(num_occ + 1, matC, tei, rhs_bbbb, [num_occ, num_orbs])

    if include_bbaa:
        r_rhs_aabb.rhs_aabb_sub(num_occ + 1, matC, tei, rhs_bbaa, [num_occ, num_orbs])

    # getting final value of the RHS terms
    # rhs_aa = 1.0 * rhs_aa
    # rhs_bb = 1.0 * rhs_bb
    # rhs_aaaa = 1.0 * rhs_aaaa
    # rhs_aabb = 1.0 * rhs_aabb
    # if include_bbaa:
    #     rhs_bbaa = 1.0 * rhs_bbaa
    # rhs_bbbb = 1.0 * rhs_bbbb

    #rhs_aaaa = -1.0 * np.zeros((nocc, nvir, nocc, nvir), order = 'F')
    #rhs_aabb = -1.0 * np.zeros((nocc, nvir, nocc, nvir), order = 'F')
    #rhs_bbaa = -1.0 * np.zeros((nocc, nvir, nocc, nvir), order = 'F')
    #rhs_bbbb = -1.0 * np.zeros((nocc, nvir, nocc, nvir), order = 'F')

    

    #print("rhs_aa = \t", rhs_aa)
    #print("rhs_bb = \t", rhs_bb)
    #print("rhs_aaaa = \t", rhs_aaaa)
    #print("rhs_bbbb = \t", rhs_bbbb)
    #print("rhs_aabb = \t", rhs_aabb)
    #print("rhs_bbaa = \t", rhs_bbaa)

    transposed_aa = np.copy(np.transpose(np.copy(rhs_aa, order = 'C')), order = 'F')
    transposed_bb = np.copy(np.transpose(np.copy(rhs_bb, order = 'C')), order = 'F')
    transposed_aaaa = np.copy(np.transpose(np.copy(rhs_aaaa, order = 'C'), (1,0,2,3)), order = 'F')
    transposed_aabb = np.copy(np.transpose(np.copy(rhs_aabb, order = 'C'), (1,0,2,3)), order = 'F')
    if include_bbaa:
        transposed_bbaa = np.copy(np.transpose(np.copy(rhs_bbaa, order = 'C'), (1,0,2,3)), order = 'F')
    transposed_bbbb = np.copy(np.transpose(np.copy(rhs_bbbb, order = 'C'), (1,0,2,3)), order = 'F')

    symmetrize_aijb_aaaa(transposed_aaaa)
    symmetrize_aijb_aaaa(transposed_bbbb)

    #transposed_aaaa = 0.25 * transposed_aaaa
    #transposed_bbbb = 0.25 * transposed_bbbb:
    # return pack_pt2_amps([np.transpose(rhs_aa), np.transpose(rhs_bb), np.transpose(rhs_aaaa, (1, 0, 2, 3)), transposed_aabb,
    #                       transposed_bbaa, np.transpose(rhs_bbbb, (1, 0, 2, 3))])

    if include_bbaa:
        return pack_pt2_amps([transposed_aa, transposed_bb, transposed_aaaa, transposed_aabb, transposed_bbaa, transposed_bbbb])
    else:
        return pack_pt2_amps([transposed_aa, transposed_bb, transposed_aaaa, transposed_aabb, transposed_bbbb])


def rhs_outputs_RAC(nvir, nocc, norb, oei, tei, Ca_coeff):
    oei_ov = np.copy(oei[:nocc, nocc:], order = 'F')
    oei_vo = np.copy(oei[nocc:, :nocc], order = 'F')

    tei_ooov = np.copy(tei[:nocc, :nocc, :nocc, nocc:], order = 'F')
    tei_ovoo = np.copy(tei[:nocc, nocc:, :nocc, :nocc], order = 'F')
    tei_vooo = np.copy(tei[nocc:, :nocc, :nocc, :nocc], order = 'F')
    tei_vovv = np.copy(tei[nocc:, :nocc, nocc:, nocc:], order = 'F')
    tei_vvov = np.copy(tei[nocc:, nocc:, :nocc, nocc:], order = 'F')
    tei_voov = np.copy(tei[nocc:, :nocc, :nocc, nocc:], order = 'F')
    tei_vovo = np.copy(tei[nocc:, :nocc, nocc:, :nocc], order = 'F')

    ttei_cakaii = np.copy(np.trace(tei[nocc:, :nocc, :nocc, :nocc], axis1=2, axis2=3), order='F')
    ttei_caikai = np.copy(np.trace(tei[nocc:, :nocc, :nocc, :nocc], axis1=1, axis2=3), order='F')
    ttei_caiij = np.copy(np.trace(tei[nocc:, :nocc, :nocc, :nocc], axis1=1, axis2=2), order='F')
    ttei_caijj = np.copy(np.trace(tei[nocc:, :nocc, :nocc, :nocc], axis1=2, axis2=3), order='F')
    ttei_kaiia = np.copy(np.trace(tei[:nocc, :nocc, :nocc, nocc:], axis1=1, axis2=2), order='F')
    ttei_kaaii = np.copy(np.trace(tei[:nocc, nocc:, :nocc, :nocc], axis1=2, axis2=3), order='F')
    ttei_iija = np.copy(np.trace(tei[:nocc, :nocc, :nocc, nocc:], axis1=0, axis2=1), order='F')
    ttei_ijia = np.copy(np.trace(tei[:nocc, :nocc, :nocc, nocc:], axis1=0, axis2=2), order='F')

    Cstart = 1
    Vstart = 1
    Hstart = 1
    Cend = nocc
    Vend = nvir
    Hend = norb

    rhs_aa_out = np.zeros((nvir, nocc), order = 'F')
    rhs_aaaa_out = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    rhs_aabb_out = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    rhs_bb_out = np.zeros((nvir, nocc), order = 'F')
    rhs_bbbb_out = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
    if include_bbaa:
        rhs_bbaa_out = np.zeros((nvir, nocc, nocc, nvir), order = 'F')

    r_rhs_aa.rhs_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, oei_ov, oei_vo, tei_ooov, tei_ovoo,
                    tei_vooo, tei_vovv, tei_vvov, ttei_iija, ttei_ijia, ttei_kaiia, ttei_kaaii, ttei_caijj,
                    ttei_caiij, ttei_cakaii, ttei_caikai, rhs_aa_out)
    r_rhs_aaaa.rhs_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, tei_voov, tei_vovo, rhs_aaaa_out)
    r_rhs_aabb.rhs_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, tei_voov, tei_vovo, rhs_aabb_out)
    r_rhs_aa.rhs_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, oei_ov, oei_vo, tei_ooov, tei_ovoo,
                    tei_vooo, tei_vovv, tei_vvov, ttei_iija, ttei_ijia, ttei_kaiia, ttei_kaaii, ttei_caijj,
                    ttei_caiij, ttei_cakaii, ttei_caikai, rhs_bb_out)
    r_rhs_aaaa.rhs_aaaa(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, tei_voov, tei_vovo, rhs_bbbb_out)
    if include_bbaa:
        r_rhs_aabb.rhs_aabb(Cstart, Cend, Vstart, Vend, Hstart, Hend, Ca_coeff, Ca_coeff, tei_voov, tei_vovo, rhs_bbaa_out)

    if include_025_RHS:
        rhs_aaaa_out = 0.25 * rhs_aaaa_out
        rhs_bbbb_out = 0.25 * rhs_bbbb_out

    symmetrize_aijb_aaaa(rhs_aaaa_out)
    symmetrize_aijb_aaaa(rhs_bbbb_out)

    if include_bbaa:
        return pack_pt2_amps([rhs_aa_out, rhs_bb_out, rhs_aaaa_out, rhs_aabb_out, rhs_bbaa_out, rhs_bbbb_out])
    else:
        return pack_pt2_amps([rhs_aa_out, rhs_bb_out, rhs_aaaa_out, rhs_aabb_out, rhs_bbbb_out])


def gen_min_res_transform(nvir, nocc, norb, Fock_mat, Ca_coeff, C0_coeff, packed_t, dimensions, ref_E0):

    # combining the LHS lin trans and overlaps into the full expression for the LHS of the equation
    # technically I could have just had a single function to start with, but this made the code easier to read and debug

    # calling the LHS lin trans and LHS overlap functions defined above
    lhs_packed_vec = pt2_lin_trans(nvir, nocc, norb, Fock_mat, Ca_coeff, C0_coeff, packed_t, dimensions)
    lhs_ovrlp_packed_vec, ovrlp_dims = lhs_ovrlps(nvir, nocc, norb, Ca_coeff, C0_coeff, packed_t, dimensions, ref_E0)

    # final_packed_vec = lhs_packed_vec + lhs_ovrlp_packed_vec
    if include_bbaa:
        lhs_t_aa, lhs_t_bb, lhs_t_aaaa, lhs_t_aabb, lhs_t_bbaa, lhs_t_bbbb = unpack_pt2_amps(dimensions, lhs_packed_vec)
        lhs_ovrlp_aa, lhs_ovrlp_bb, lhs_ovrlp_aaaa, lhs_ovrlp_aabb, lhs_ovrlp_bbaa, lhs_ovrlp_bbbb = unpack_pt2_amps(dimensions, lhs_ovrlp_packed_vec)
    else:
        lhs_t_aa, lhs_t_bb, lhs_t_aaaa, lhs_t_aabb, lhs_t_bbbb = unpack_pt2_amps(dimensions, lhs_packed_vec)
        lhs_ovrlp_aa, lhs_ovrlp_bb, lhs_ovrlp_aaaa, lhs_ovrlp_aabb, lhs_ovrlp_bbbb = unpack_pt2_amps(dimensions, lhs_ovrlp_packed_vec)


    final_t_aa = lhs_t_aa
    final_t_bb = lhs_t_bb
    final_t_aaaa = lhs_t_aaaa
    final_t_aabb = lhs_t_aabb
    if include_bbaa:
        final_t_bbaa = lhs_t_bbaa
    final_t_bbbb = lhs_t_bbbb

    # final_t_aa = lhs_t_aa + lhs_ovrlp_aa
    # final_t_bb = lhs_t_bb + lhs_ovrlp_bb
    # final_t_aaaa = lhs_t_aaaa + lhs_ovrlp_aaaa
    # final_t_aabb = lhs_t_aabb + lhs_ovrlp_aabb
    # if include_bbaa:
    #     final_t_bbaa = lhs_t_bbaa + lhs_ovrlp_bbaa
    # final_t_bbbb = lhs_t_bbbb + lhs_ovrlp_bbbb
    #
    # for ii in range(final_t_aaaa.shape[0]):
    #     for jj in range(final_t_aaaa.shape[1]):
    #         for kk in range(final_t_aaaa.shape[2]):
    #             for ll in range(final_t_aaaa.shape[3]):
    #                 if final_t_aaaa[ii, jj, kk, ll] != final_t_bbbb[ii, jj, kk, ll]:
    #                     print("the aaaa and bbbb excitation coefficients have diverged")
    #                     #print("%d \t %d \t %d \t %d: \t aaaa_val: %5.10f, \t bbbb_val: %5.10f" %(ii, jj, kk, ll, final_t_aaaa[ii, jj, kk, ll], final_t_bbbb[ii, jj, kk, ll]))
    #
    # for ii in range(final_t_aa.shape[0]):
    #     for jj in range(final_t_aa.shape[1]):
    #         if final_t_aa[ii, jj] != final_t_bb[ii, jj]:
    #             print("The aa and bb excitation coefficients have diverged. ")

    if include_bbaa:
        final_packed, dimensions = pack_pt2_amps([final_t_aa, final_t_bb, final_t_aaaa, final_t_aabb, final_t_bbaa, final_t_bbbb])
    else:
        final_packed, dimensions = pack_pt2_amps([final_t_aa, final_t_bb, final_t_aaaa, final_t_aabb, final_t_bbbb])

    return final_packed

#def rhs_outputs(num_occ, num_vir, num_orbs, matC, oei, tei):
#    rhs_aa = np.ones((num_vir, num_occ), order='F')
#    rhs_aaaa = np.ones((num_vir, num_occ, num_occ, num_vir), order = 'F')
#    rhs_aabb = np.ones((num_vir, num_occ, num_occ, num_vir), order = 'F')
#    rhs_bb = np.ones((num_vir, num_occ), order='F')
#    rhs_bbbb = np.ones((num_vir, num_occ, num_occ, num_vir), order='F')
#    rhs_bbaa = np.ones((num_vir, num_occ, num_occ, num_vir), order='F')
#    return pack_pt2_amps([rhs_aa, rhs_bb, rhs_aaaa, rhs_aabb, rhs_bbaa, rhs_bbbb])

# this as a guess caused weird issues with the genMinRes stuff 
# guess_aa = np.zeros((nvir, nocc), order = 'F')
# guess_aaaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
# guess_aabb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
# guess_bb = np.zeros((nvir, nocc), order = 'F')
# guess_bbbb = np.zeros((nvir, nocc, nocc, nvir), order = 'F')
# guess_bbaa = np.zeros((nvir, nocc, nocc, nvir), order = 'F')

# guess_aa = np.ones((nvir, nocc), order = 'F') * 0.00001
# guess_aaaa = np.ones((nvir, nocc, nocc, nvir), order = 'F') * 0.00001
# guess_aabb = np.ones((nvir, nocc, nocc, nvir), order = 'F') * 0.00001
# guess_bb = np.ones((nvir, nocc), order = 'F') * 0.00001
# guess_bbbb = np.ones((nvir, nocc, nocc, nvir), order = 'F') * 0.00001
# guess_bbaa = np.ones((nvir, nocc, nocc, nvir), order = 'F') * 0.00001

guess_aa = np.ones((nvir, nocc), order = 'F')
guess_aaaa = np.ones((nvir, nocc, nocc, nvir), order = 'F')
guess_aabb = np.ones((nvir, nocc, nocc, nvir), order = 'F')
guess_bb = np.ones((nvir, nocc), order = 'F')
guess_bbbb = np.ones((nvir, nocc, nocc, nvir), order = 'F')
if include_bbaa:
    guess_bbaa = np.ones((nvir, nocc, nocc, nvir), order = 'F')

# if include_bbaa:
#     packed_guess, packed_dims = pack_pt2_amps([guess_aa, guess_bb, guess_aaaa, guess_aabb, guess_bbaa, guess_bbbb])
# else:
#     packed_guess, packed_dims = pack_pt2_amps([guess_aa, guess_bb, guess_aaaa, guess_aabb, guess_bbbb])

#print("packed dimensions: \n", packed_dims)

rhs_terms, dims = rhs_outputs_RAC(nvir, nocc, norb, MO_oints, MO_tints, Ca_coeff)
packed_guess = rhs_terms
packed_dims = dims

# a better guess: (following the old MP2 code
# diag_aa = np.ones([nvir, nocc])
# diag_aaaa = np.ones([nvir, nocc, nocc, nvir])
# diag_aabb = np.ones([nvir, nocc, nocc, nvir])
#
# for a in range(nocc, norb):
#     for i in range(nocc):
#         diag_aa[a-nocc, i] = 1.0/(Fock_mat[a-nocc, a-nocc] - Fock_mat[i, i] - refE0X)
#
# for a in range(nocc, norb):
#     for i in range(nocc):
#         for j in range(i+1, nocc):
#             for b in range(a+1, norb):
#                 diag_aaaa[a-nocc, i, j, b-nocc] = 1.0 / (Fock_mat[a-nocc, a-nocc] + Fock_mat[b-nocc, b-nocc] - Fock_mat[i, i] - Fock_mat[j, j] - refE0X)
#                 diag_aaaa[b-nocc, i, j, a-nocc] = 1.0 / (Fock_mat[a-nocc, a-nocc] + Fock_mat[b-nocc, b-nocc] - Fock_mat[i, i] - Fock_mat[j, j] - refE0X)
#                 diag_aaaa[b-nocc, j, i, a-nocc] = 1.0 / (Fock_mat[a-nocc, a-nocc] + Fock_mat[b-nocc, b-nocc] - Fock_mat[i, i] - Fock_mat[j, j] - refE0X)
#                 diag_aaaa[a-nocc, j, i, b-nocc] = 1.0 / (Fock_mat[a-nocc, a-nocc] + Fock_mat[b-nocc, b-nocc] - Fock_mat[i, i] - Fock_mat[j, j] - refE0X)
#
# for a in range(nocc, norb):
#     for i in range(nocc):
#         for j in range(nocc):
#             for b in range(nocc, norb):
#                 diag_aabb[a-nocc, i, j, b-nocc] = 1.0/(Fock_mat[a-nocc, a-nocc] + Fock_mat[b-nocc, b-nocc] - Fock_mat[i, i] - Fock_mat[j, j] - refE0X)
#
# packed_diag, packed_dims = pack_pt2_amps([diag_aa, diag_aa, diag_aaaa, diag_aabb, diag_aabb, diag_aaaa])
#
# packed_guess = rhs_terms * packed_diag

# actually getting the values of the excitation constants via a Krylov subspace solver
# packed_pt2_amps = genMinRes(rhs_terms, packed_guess,
#                             lambda x: gen_min_res_transform(nvir, nocc, norb, Fock_mat, Ca_coeff, C0_coeff, x, dims, refE0),
#                             thresh = 1.0e-6,
#                             precondition = lambda x: x*packed_diag,
#                             maxiter = 400)

packed_pt2_amps = genMinRes(rhs_terms, packed_guess,
                            lambda x: gen_min_res_transform(nvir, nocc, norb, Fock_mat, Ca_coeff, C0_coeff, x, dims, refE0),
                            thresh = 1.0e-6,
                            maxiter = 200)

if include_bbaa:
    t2_aa, t2_bb, t4_aaaa, t4_aabb, t4_bbaa, t4_bbbb = unpack_pt2_amps(packed_dims, packed_pt2_amps)
else:
    t2_aa, t2_bb, t4_aaaa, t4_aabb, t4_bbbb = unpack_pt2_amps(packed_dims, packed_pt2_amps)
symmetrize_aijb_aaaa(t4_aaaa)
symmetrize_aijb_aaaa(t4_bbbb)

# initializing the energy values
# f2py for some reason made the expected output a single valued array instead of just a real
# still am not sure why it did that
e_a = np.zeros((1))
e_b = np.zeros((1))
e_aa = np.zeros((1))
e_bb = np.zeros((1))
e_ab = np.zeros((1))
e_ba = np.zeros((1))

g_oooo = np.copy(MO_tints[:nocc, :nocc, :nocc, :nocc], order = 'F')
g_ooov = np.copy(MO_tints[:nocc, :nocc, :nocc, nocc:], order = 'F')
g_oovo = np.copy(MO_tints[:nocc, :nocc, nocc:, :nocc], order = 'F')
g_ovoo = np.copy(MO_tints[:nocc, nocc:, :nocc, :nocc], order = 'F')
g_vooo = np.copy(MO_tints[nocc:, :nocc, :nocc, :nocc], order = 'F')
g_oovv = np.copy(MO_tints[:nocc, :nocc, nocc:, nocc:], order = 'F')
g_ovov = np.copy(MO_tints[:nocc, nocc:, :nocc, nocc:], order = 'F')
g_voov = np.copy(MO_tints[nocc:, :nocc, :nocc, nocc:], order = 'F')
g_ovvo = np.copy(MO_tints[:nocc, nocc:, nocc:, :nocc], order = 'F')
g_vovo = np.copy(MO_tints[nocc:, :nocc, nocc:, :nocc], order = 'F')
g_vvoo = np.copy(MO_tints[nocc:, nocc:, :nocc, :nocc], order = 'F')
g_ovvv = np.copy(MO_tints[:nocc, nocc:, nocc:, nocc:], order = 'F')
g_vovv = np.copy(MO_tints[nocc:, :nocc, nocc:, nocc:], order = 'F')
g_vvov = np.copy(MO_tints[nocc:, nocc:, :nocc, nocc:], order = 'F')
g_vvvo = np.copy(MO_tints[nocc:, nocc:, nocc:, :nocc], order = 'F')
g_vvvv = np.copy(MO_tints[nocc:, nocc:, nocc:, nocc:], order = 'F')

r_pt2_e_a.pt2_e_a(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), np.copy(int_temp),
                          np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp),
                          np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), C0_coeff, C0_coeff_2,
                          Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, g_oooo, g_ooov, g_oovo, g_oovv, g_ovoo,
                          g_ovov, g_ovvo, g_ovvv, g_vooo, g_voov, g_vovo, g_vovv, g_vvoo, g_vvov, g_vvvo, g_vvvv, t2_aa,
                          tf_oo, tg_jjkk, tg_jkjk, tg_ijkk, tg_ijjk, tg_iija, tg_iija, tg_ijia, tg_ijia, tg_iiaj,
                          tg_iiaj, tg_iiaj, tg_iiaj, tg_jkbk, tg_iaij, tg_iaij, tg_iaij, tg_iaij, tg_jbkk, tg_ajjb,
                          tg_abjj, e_a)
r_pt2_e_a.pt2_e_a(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), np.copy(int_temp),
                          np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp),
                          np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), np.copy(int_temp), C0_coeff, C0_coeff_2,
                          Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, g_oooo, g_ooov, g_oovo, g_oovv, g_ovoo,
                          g_ovov, g_ovvo, g_ovvv, g_vooo, g_voov, g_vovo, g_vovv, g_vvoo, g_vvov, g_vvvo, g_vvvv, t2_bb,
                          tf_oo, tg_jjkk, tg_jkjk, tg_ijkk, tg_ijjk, tg_iija, tg_iija, tg_ijia, tg_ijia, tg_iiaj,
                          tg_iiaj, tg_iiaj, tg_iiaj, tg_jkbk, tg_iaij, tg_iaij, tg_iaij, tg_iaij, tg_jbkk, tg_ajjb,
                          tg_abjj, e_b)
r_pt2_e_aa.pt2_e_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), C0_coeff,
                          C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, g_oooo, g_ooov, g_oovo, g_oovv, g_ovoo,
                          g_ovov, g_ovvo, g_ovvv, g_vooo, g_voov, g_vovo, g_vovv, g_vvoo, g_vvov, g_vvvo, g_vvvv, t4_aaaa,
                          tg_jjkb, tg_jkjb, e_aa)
r_pt2_e_aa.pt2_e_aa(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), C0_coeff,
                          C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, g_oooo, g_ooov, g_oovo, g_oovv, g_ovoo,
                          g_ovov, g_ovvo, g_ovvv, g_vooo, g_voov, g_vovo, g_vovv, g_vvoo, g_vvov, g_vvvo, g_vvvv, t4_bbbb,
                          tg_jjkb, tg_jkjb, e_bb)
r_pt2_e_ab.pt2_e_ab(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), C0_coeff,
                            C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, g_oooo, g_ooov, g_oovo, g_oovv, g_ovoo,
                            g_ovov, g_ovvo, g_ovvv, g_vooo, g_voov, g_vovo, g_vovv, g_vvoo, g_vvov, g_vvvo, g_vvvv, t4_aabb,
                            tg_jjkb, tg_jkjb, tg_jjbk, tg_jbjk, e_ab)
if include_bbaa:
    r_pt2_e_ab.pt2_e_ab(Cstart, Cend, Vstart, Vend, Hstart, Hend, np.copy(int_temp), np.copy(int_temp), C0_coeff,
                                C0_coeff_2, Ca_coeff, Ca_coeff, f_oo, f_ov, f_vo, f_vv, g_oooo, g_ooov, g_oovo, g_oovv, g_ovoo,
                                g_ovov, g_ovvo, g_ovvv, g_vooo, g_voov, g_vovo, g_vovv, g_vvoo, g_vvov, g_vvvo, g_vvvv, t4_bbaa,
                                tg_jjkb, tg_jkjb, tg_jjbk, tg_jbjk, e_ba)

print("\nRHF det. energy  = %20.12f" % rhfDetEnergy )
print("MP2 energy       = %20.12f" % mp2Energy)
print("reference energy = %20.12f" % refE)
print("reference E0    =%20.12f" %refE0)
print("e_a = %20.12f" %e_a)
print("e_b = %20.12f" %e_b)
print("e_aa = %20.12f" %e_aa)
print("e_ab = %20.12f" %e_ab)
if include_bbaa:
    print("e_ba = %20.12f" %e_ba)
print("e_bb = %20.12f" %e_bb)

if include_bbaa:
    pt2_tot_E = refE + e_a + e_b + e_aa + e_bb + e_ab + e_ba
else:
    pt2_tot_E = refE + e_a + e_b + e_aa + e_bb + e_ab


print("Total perturbation energy = %20.12f" %pt2_tot_E[0])


print("aa abs average")
check = 0
for ii in range(t2_aa.shape[0]):
    for jj in range(t2_aa.shape[1]):
        check = check + abs(t2_aa[ii, jj])
print(check/t2_aa.size)

print("bb abs average")
check = 0
for ii in range(t2_bb.shape[0]):
    for jj in range(t2_bb.shape[1]):
        check = check + abs(t2_bb[ii, jj])
print(check/t2_bb.size)



