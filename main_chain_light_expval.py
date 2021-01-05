# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 09:25:27 2020

@author: giaco
"""
import tenpy
import copy
import sys
import numpy as np
import numpy.linalg as alg
from tenpy import models
from tenpy.networks.site import SpinSite
from tenpy.networks.site import FermionSite
from tenpy.networks.site import BosonSite
from tenpy.models.model import CouplingModel
from tenpy.models.model import CouplingMPOModel
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Lattice
from tenpy.tools.params import get_parameter
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPO, MPOEnvironment
import tenpy.linalg.charges as charges
from tenpy.models.lattice import Chain
from scipy.linalg import expm
from tenpy.models.fermions_spinless import FermionModel
from tenpy.algorithms.tebd import Engine


def sites(L,Nmax):
 FSite=FermionSite(None, filling=0.5)
 BSite=BosonSite(Nmax=Nmax,conserve=None, filling=0 )
 sites=[]
 sites.append(BSite)
 for i in range(L):
     sites.append(FSite)
 return sites

def product_state(L):
    ps=['vac']
    for i in range(int(L/2)):
        ps.append('empty')
        ps.append('full')
    return ps

def psi(sites,ps):
    psi=MPS.from_product_state(sites, ps)
    return psi
class Boson_operator:
    def __init__(self, dim):
        B = np.zeros([dim+1, dim+1], dtype=np.float)  # destruction/annihilation operator
        for n in range(1, dim+1):
          B[n - 1, n] = np.sqrt(n)
        self.Id=np.identity(dim+1)
        self.B=B
        self.Bd = np.transpose(B)  # .conj() wouldn't do anything
        # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
        Ndiag=np.arange(dim+1, dtype=np.float)
        self.Ndiag = Ndiag
        self.N = np.diag(Ndiag)
        self.NN = np.diag(Ndiag**2)
        return None
    
class Fermion_operator:
    def __init__(self, h):
        JW = np.array([[1., 0.], [0., -1.]])
        self.JW = JW
        C = np.array([[0., 1.], [0., 0.]])
        self.C = C
        
        self.Cd = np.array([[0., 0.], [1., 0.]])
        self.N = np.array([[0., 0.], [0., 1.]])
        self.Id=np.identity(2)
        return None


def Ev_bond(Nmax, dt, g, t):
       bos=Boson_operator(Nmax)
       B=bos.B
       Bd=bos.Bd
       A=1j*g*(B+Bd)
       Ad=-1j*g*(B+Bd)
       P = expm(A)
       Pd= expm(Ad)
       H=np.zeros((Nmax+1,2,2,Nmax+1,2,2), dtype=complex)
       H[:,1,0,:,0,1]=t*P
       H[:,0,1,:,1,0]=t*Pd
       H = np.reshape(H, [(Nmax+1)*4, (Nmax+1)*4])
       U = expm(-dt * H)
       U_bond=npc.Array.from_ndarray_trivial(np.reshape(U, [(Nmax+1), 2, 2, (Nmax+1),2,2]))
       return U_bond
   
def Update_bond_right(i,psi,U_bond):
    theta=psi.get_theta(i,n=3)
    Utheta = npc.tensordot(U_bond, theta, axes=([3, 4, 5], [1, 2, 3]))  
    Utheta.itranspose([3,0,1,2,4])
    Utheta.iset_leg_labels(['vL','p0','p1','p2','vR'])
    pstemp=MPS.from_full([sites[0],sites[1],sites[1]],Utheta,form='B',cutoff=1e-12, normalize=False, bc='finite', outer_S=None)
    for j in range(3):
        psi.set_B(j,pstemp.get_B(j))
        psi.set_SL(j, pstemp.get_SL(j))
        psi.set_SL(j, pstemp.get_SR(j))
    psi.swap_sites(i)    
   
    


def Bos_ferm_bond(sites, Nmax, L, g, U, Omega, J, Nfer, dt, d, options, trunc_param, err):
    
    bos=Boson_operator(Nmax)
    FId=np.identity(2)
    FId=np.reshape(FId, [1,1,2,2])
    B=bos.B
    Bd=bos.Bd
    Nb=Omega*bos.N
    A=1j*g*(B+Bd)
    Ad=-1j*g*(B+Bd)
    P = expm(A)
    Pd= expm(Ad)
    H=np.zeros((Nmax+1,2,2,Nmax+1,2,2), dtype=complex)
    for i in range(2):
      for j in range(2):
        for k in range(2):
            for l in range(2):
                H[:,i,j,:,k,l]=Nb

    H[:,1,0,:,0,1]=(-J*P)+Nb
    H[:,0,1,:,1,0]=(-J*Pd)+Nb
    H = np.reshape(H, [(Nmax+1)*4, (Nmax+1)*4])
    U_bond = expm(-(dt/(2*L)) * H)
    U_bond = np.reshape(U_bond, [(Nmax+1), 2, 2, (Nmax+1),2,2])
    U_bond=np.transpose(U_bond, [0,3,1,4,2,5])
    U_bond = np.reshape(U_bond, [(Nmax+1)*(Nmax+1),16])
    U, s, V=alg.svd(U_bond, full_matrices=False)
    r=s.shape[0]
    for i in range(s.shape[0]):
                  D=r-i 
                  if s[-i-1]>err:
                     
                     break
                        
    s=s[0:D]
    U=U[:,0:D]
    V=V[0:D,:]
    W1=copy.deepcopy(U)
    V1=copy.deepcopy(V)
    for i in range(D):
       W1[:,i]=U[:,i]*np.sqrt(s[i])
       V1[i,:]=V[i,:]*np.sqrt(s[i])
    V1=np.reshape(V, [D*4, 4])
    W1=np.reshape(W1, [1,Nmax+1,Nmax+1,D])
    W1=np.transpose(W1, [0,3,1,2])
    print(W1.shape)

    U, s, V=alg.svd(V1, full_matrices=False)
    r=s.shape[0]
    for i in range(s.shape[0]):
                  D=r-i 
                  if s[-i-1]>err:
                     
                     break
    s=s[0:D]
    U=U[:,0:D]
    V=V[0:D,:]
    W2=copy.deepcopy(U)
    W3=copy.deepcopy(V)
 
    for i in range(D):
      W2[:,i]=U[:,i]*np.sqrt(s[i])
      W3[i,:]=V[i,:]*np.sqrt(s[i])
    W2=np.reshape(W2, [W1.shape[1], 2, 2, D])
    W2=np.transpose(W2, [0,3,1,2])
    W3=np.reshape(W3,[D, 2, 2, 1] )
    W3=np.transpose(W3, [0,3,1,2])
    
    chinfo = npc.ChargeInfo([1], ['N'])
    Bqflat=[[0]]*Nmax
    idl=[0]*(4)
    Ws=[npc.Array.from_ndarray_trivial(W1, dtype=complex, labels=['wL', 'wR', 'p', 'p*']),npc.Array.from_ndarray_trivial(W2, dtype=complex, labels=['wL', 'wR', 'p', 'p*']),npc.Array.from_ndarray_trivial(W3, dtype=complex, labels=['wL', 'wR', 'p', 'p*'])]
    M=MPO([sites[0],sites[1],sites[2]], Ws,IdL=idl,IdR=idl, bc='segment')
    return M


def ferm_bond(sites, U, dt):
    H=np.zeros((2,2,2,2), dtype=complex)
    H[1,1,1,1]=U
    H = np.reshape(H, [4, 4])
    U_bond = expm(-(dt/(2)) * H)
    U_bond = np.reshape(U_bond, [2, 2, 2, 2])
    U_bond=np.transpose(U_bond, [0, 2, 1, 3])
    U_bond = np.reshape(U_bond, [4,4])
    U, s, V=alg.svd(U_bond, full_matrices=False)
    W1=copy.deepcopy(U)
    V1=copy.deepcopy(V)
    for i in range(s.shape[0]):
       W1[:,i]=U[:,i]*np.sqrt(s[i])
       V1[i,:]=V[i,:]*np.sqrt(s[i])
    W2=np.reshape(V, [s.shape[0],2,2,1])
    W1=np.reshape(W1, [1,2,2,s.shape[0]])
    W1=np.transpose(W1, [0,3,1,2])
    W2=np.transpose(W2, [0,3,1,2])
    
    chinfo = npc.ChargeInfo([1], ['N'])
    Bqflat=[[0]]*Nmax
    idl=[0]*(3)
    Ws=[npc.Array.from_ndarray_trivial(W1, dtype=complex, labels=['wL', 'wR', 'p', 'p*']),npc.Array.from_ndarray_trivial(W2, dtype=complex, labels=['wL', 'wR', 'p', 'p*'])]
    M=MPO([sites[1],sites[2]], Ws,IdL=idl,IdR=idl, bc='segment')
    return M


def Fermion_swap(psi, MF, trunc_param, options):
    for i in range(int(L/2)):
             theta=psi.get_theta(1+(2*i),n=2)
             ptheta=MPS.from_full([sites[1], sites[2]], theta, form='B', bc='segment', outer_S=(psi.get_SL(1+(2*i)),psi.get_SR((2*i)+2)))
             MF.apply_naively(ptheta)
             for j in range(2):
                    psi.set_B(1+(2*i)+j, ptheta.get_B(j,form='B'))
                    psi.set_SL(1+(2*i)+j, ptheta.get_SL(j))
                    psi.set_SR(1+(2*i)+j, ptheta.get_SR(j))
    for i in range(int(L/2)-1):
             theta=psi.get_theta(2+(2*i),n=2)
             ptheta=MPS.from_full([sites[1], sites[2]], theta, form='B', bc='segment', outer_S=(psi.get_SL(2+(2*i)),psi.get_SR(3+(2*i))))
             MF.apply_naively(ptheta)
             for j in range(2):
                    psi.set_B(2+(2*i)+j, ptheta.get_B(j,form='B'))
                    psi.set_SL(2+(2*i)+j, ptheta.get_SL(j))
                    psi.set_SR(2+(2*i)+j, ptheta.get_SR(j))
    psi.compress(options)

def right_swap(psi, M, L, trunc_param, options):
  for i in range(L-2):
     "action along all the fermion chain of the bond MPO M"
     
     theta=psi.get_theta(i,n=3)
     ptheta=MPS.from_full(sites[0:3], theta, form='B', bc='segment', outer_S=(psi.get_SL(i),psi.get_SR(i+2)))
     M.apply_naively(ptheta)
     for j in range(3):
        psi.set_B(i+j, ptheta.get_B(j,form='B'))
        psi.set_SL(i+j, ptheta.get_SL(j))
        psi.set_SR(i+j, ptheta.get_SR(j))
     psi.swap_sites(i, swap_op='auto', trunc_par=trunc_param)
  "The last two sites are treated separately 'cause there are no more swaps."
  theta=psi.get_theta(L-2,n=3)
  ptheta=MPS.from_full(sites[0:3], theta, form='B', bc='segment', outer_S=(psi.get_SL(L-2),psi.get_SR(L)))
  M.apply_naively(ptheta)
  for j in range(3):
        psi.set_B((L-2)+j, ptheta.get_B(j,form='B'))
        psi.set_SL((L-2)+j, ptheta.get_SL(j))
        psi.set_SR((L-2)+j, ptheta.get_SR(j))
  psi.compress(options) 
  print("right swap done")
  





def left_swap(psi, M, L, trunc_param, options):
      for i in range(L-2):
        
        theta=psi.get_theta((L-2)-i,n=3)
        ptheta=MPS.from_full(sites[0:3], theta, form='B', bc='segment', outer_S=(psi.get_SL((L-2)-i),psi.get_SR(L-i)))
        M.apply_naively(ptheta)
        
        for j in range(3):
          psi.set_B((L-2)-i+j, ptheta.get_B(j,form='B'))
          psi.set_SL((L-2)-i+j, ptheta.get_SL(j))
          psi.set_SR((L-2)-i+j, ptheta.get_SR(j))
        psi.swap_sites((L-3)-i, swap_op='auto', trunc_par=trunc_param)
        theta=psi.get_theta(0,n=3)
        
      ptheta=MPS.from_full(sites[0:3], theta, form='B', bc='segment', outer_S=(psi.get_SL(0),psi.get_SR(2)))
      M.apply_naively(ptheta)
      for j in range(3):
          psi.set_B(j, ptheta.get_B(j,form='B'))
          psi.set_SL(j, ptheta.get_SL(j))
          psi.set_SR(j, ptheta.get_SR(j))
      psi.compress(options)
      print("Left Swap done")
        
    

Nmax, L, g, U, Omega, t, Nfer, dt, d =5, 50, 1, 0.5, 1, 1, 10, 0.05, 2
err=0.0000001
steps=5
sites = sites(L,Nmax)
ps= product_state(L)
psi=psi(sites,ps)

verbose=False
trunc_param={'svd_min': 0.0000001, 'verbose': verbose}
options={
            'compression_method': 'SVD',
            'trunc_param': trunc_param,
            'verbose': verbose 
            }



    

print("Initial exp value at site ", psi.expectation_value('N', sites=[2]))

"""
def main(sites, Nmax, L, g, U, Omega, t, Nfer, dt, d, options, trunc_param, err, steps, psi):
    M=Bos_ferm_bond(sites, Nmax, L, g, U, Omega, t, Nfer, dt, d, options, trunc_param, err)
    MF=ferm_bond(sites, U, dt)
    for i in range(steps):
        
        right_swap(psi, M, L, trunc_param, options)
        left_swap(psi, M, L, trunc_param, options)
        
        print("Iteration n:", i, "psi:", psi)
    return psi

        
main(sites, Nmax, L, g, U, Omega, t, Nfer, dt, d, options, trunc_param, err, steps, psi)    


print("Initial exp value at site 2 after the swep:", psi.expectation_value('N', sites=[0]))
"""



ccd=npc.outer(psi.sites[0].B.replace_labels(['p', 'p*'], ['p0', 'p0*']),psi.sites[1].Cd.replace_labels(['p', 'p*'], ['p1', 'p1*']))

print(1j*g*(psi.sites[0].B+psi.sites[0].Bd))










  





       
   






    
   
    
    



    
    
    