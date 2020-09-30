# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:55:53 2020

@author: Sergey
"""

from Bio import Restriction
import Core.SIMTraces
import numpy as np


def GetGauss(sigma,pixelsz):
    x = np.linspace(-18,17)*pixelsz
    y = np.linspace(-18,17)*pixelsz
    
    xv,yv = np.meshgrid(x,y)
    Gauss = 20*np.multiply(np.exp(-np.power(xv,2)/(2*np.power(sigma,2))),np.exp(-np.power(yv,2)/(2*np.power(sigma,2))))
    return Gauss
    
    
    
    

def GetFWHM(wavelength,NA):
    FWHM = wavelength/(2*NA)
    return FWHM
    
def FWHMtoSigma(FWHM):
    sigma = FWHM/2.355
    return sigma 

def SigmatoFWHM(Sigma):
    FWHM = Sigma*2.355
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
    
    