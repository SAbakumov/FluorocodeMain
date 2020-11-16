# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:55:53 2020

@author: Sergey
"""

from Bio import Restriction
from Core.Noise import perlin
import Core.SIMTraces
import numpy as np
import sys

from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z



def GetGauss1d(size,sigma,pixelsz):
    x = np.linspace(-np.round(size/2),np.round(size/2), size+1)*pixelsz
    Gauss = 20*np.exp(-np.power(x,2)/(2*np.power(sigma,2)))
    Gauss = Gauss[0:len(Gauss)-1]
    return Gauss

def GetPerlinNoise(size):
    noise=perlin.Perlin(50)

    time=[i for i in range(size)]
    values=np.array([noise.valueAt(i) for i in time])
    return values


def GetGauss(sigma,pixelsz):
    x = np.linspace(-18,17)*pixelsz
    y = np.linspace(-18,17)*pixelsz
    
    xv,yv = np.meshgrid(x,y)
    Gauss = 20*np.multiply(np.exp(-np.power(xv,2)/(2*np.power(sigma,2))),np.exp(-np.power(yv,2)/(2*np.power(sigma,2))))
    return Gauss
    
def GetFilename(File):
    
    Filename = File.rpartition('\\')[-1]
    return Filename
    
def update_progress(progress):
   barLength = 1 # Modify this to change the length of the progress bar
   status = ""
   if isinstance(progress, int):
       progress = float(progress)
   if not isinstance(progress, float):
       progress = 0
       status = "error: progress var must be float\r\n"
   if progress < 0:
       progress = 0
       status = "Halt...\r\n"
   if progress >= 1:
       progress = 1
       status = "Done...\r\n"
   block = int(round(barLength*progress))
   text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
   sys.stdout.write(text)
   sys.stdout.flush()
   

def GetFWHM(wavelength,NA,ResEnhancement):
    FWHM =  0.61*wavelength/(NA)/ResEnhancement
    return FWHM
    
def FWHMtoSigma(FWHM):
    sigma = FWHM/2.3548
    return sigma 

def SigmatoFWHM(Sigma):
    FWHM = Sigma*2.3548
    return FWHM
    
def rebasecuts(Enzyme, Strand):
    batch = Restriction.RestrictionBatch()
    batch.add(Enzyme)
    enzyme = batch.get(Enzyme)    
    
    Sites = enzyme.search(Strand)
    
    return Sites



def kbToPx(arr,args):
    if type(args)==list:
        stretch, nmbp, pixelsz = args[0] , args[1] ,  args[2]
    else:
        stretch, nmbp, pixelsz = args.Stretch , args.BPSize ,  args.PixelSize

        
    arr  =   (arr*stretch*nmbp)/pixelsz
    return arr

def PxTokb(arr,args):
    if type(args)==list:
        stretch, nmbp, pixelsz = args[0] , args[1] ,  args[2]
    elif type(args)==Core.SIMTraces.TSIMTraces:
        stretch, nmbp, pixelsz = args.stretch , args.nmbp ,  args.pixelsz
    else:
        print('Unsupported data type in kbToPx, aborting execution')
    
    
    arr  =   (arr/stretch/nmbp)*pixelsz
    return arr
    
def ZScoreTransform(trace):
    trace = (trace - np.mean(trace))/np.std(trace)
    return trace