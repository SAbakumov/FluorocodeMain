
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:58:18 2020

@author: Boris
"""

"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import Core.RandomTraceGenerator as RTG
import numpy as np
# from ImGen.ImGen import TrainImageGenerator
from Core.DataHandler import DataConverter
from Core.DataHandler import DataLoader
from Core.Noise import perlin
from datetime import date
import matplotlib.pyplot as plt
import scipy.io
import random

Params = {"Wavelength" : 576,
        "NA" : 1.4,
        "FragmentSize" : 250,
        "PixelSize" : 39.68,
        "ResEnhancement":2,
        "GeneratorType" : "FromLags",
        "numsamples" : 5000,
        "ArtificialDyeNumber" : 15462,
        "ArtificialGenomeLen" : 66811,
        "NumTransformations"  : 60,
        "StretchingFactor" : 1.68,
        "LowerBoundEffLabelingRate" : 0.7,
        "UpperBoundEffLabelingRate" : 0.9,
        "shift" : 12,
        "step" : 2,
        "Date" : date.today()}






# genomes = ['Other','CP014051','CP039296','CP000948']
# genomes = ['NC_000913.3','Other','CP014051','CP014787']


file = scipy.io.loadmat('D:\Sergey\FluorocodeMain\FluorocodeMain\Traces.mat')
traces = file["Traces"]

file = scipy.io.loadmat('D:\Sergey\FluorocodeMain\FluorocodeMain\Lags.mat')
lags = file["Lags"]

file = scipy.io.loadmat('D:\Sergey\FluorocodeMain\FluorocodeMain\Lengs.mat')
lengs = file["Lengs"]



    
    
NoiseProfiles = np.load("D:/Sergey/FluorocodeMain/FluorocodeMain/Core/Noise/NoiseProfiles.npy")

genomes = ['NC_000913.3','NZ_AP018795.1','NZ_CP034060.1','NZ_CP033888.1']
# genomes = ['NC_000913.3','Artificial_1']


AllLabels = []
AllProfiles = []
    
Dt = DataConverter()
Ds = DataLoader()



for genome in genomes:
    

    
    
    
    EffLabeledTraces = []
    Profiles = []

    SIMTRC     = SIMTraces.TSIMTraces(genome,Params["StretchingFactor"],0.34,0,'TaqI',Params["PixelSize"] )  
    Gauss  = Misc.GetGauss1d(Params["FragmentSize"] ,  Misc.FWHMtoSigma(Misc.GetFWHM(Params["Wavelength"],Params["NA"],Params["ResEnhancement"])),Params["PixelSize"] )

    if not 'Artificial' in genome:
        ReCuts     = SIMTRC.GetTraceRestrictions()
        ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(ReCuts)
    else:
        try:
            ReCutsInPx = np.empty(Params["ArtificialDyeNumber"])
            for i in range(0,Params["ArtificialDyeNumber"]):
                ReCutsInPx[i] = random.uniform(0,Params["ArtificialGenomeLen"]  ) 
        except:
            assert "AritificialDyeNumber" in locals() or  "ArtificialGenomeLen" in locals(), "The length or number of dyes of artificial genome is not specified."                 
        
    R = RTG.RandomTraceGenerator(Misc.kbToPx(100000,SIMTRC),Misc.kbToPx(20000,SIMTRC),Params["numsamples"])
    
        
        
        
        
    AllTrc = [];
    for transform in range(0,Params["NumTransformations"]):
        print(transform)
        
        FullTrace = SIMTRC.GetFullProfile(R.GetEffLabelingRate([ReCutsInPx ],random.uniform(Params["LowerBoundEffLabelingRate"],Params["UpperBoundEffLabelingRate"])))
        FullTraceProfile = SIMTRC.GetFluorocodeProfile(FullTrace,Gauss,FullTrace.shape[0])
 
    
        if Params["GeneratorType"]=="FromLags":
            
            
            for lag in range(0,len(lags)):
                assert "shift" in Params , "No shift around lag is defined"
                
                for offset in range(-Params["shift"],Params["shift"],Params["step"]):
                    for lag in range(0,len(lags)):
                        
                        if lag>len(FullTraceProfile):
                            lags[lag] = random.randint(0,len(FullTraceProfile)-Params["FragmentSize"])
                            
                        fullag = lags[lag].item()-lengs[lag].item()+offset
                        trc =FullTraceProfile[fullag:fullag+Params["FragmentSize"]]+NoiseProfiles[np.random.randint(0,NoiseProfiles.shape[0]),:]
                        EffLabeledTraces.append(trc)
                        
                        
        elif Params["GeneratorType"]=="FromFull":
            
            for offset in range(0,len(FullTraceProfile)-Params["FragmentSize"],Params["step"]):
                trc =FullTraceProfile[offset:offset+Params["FragmentSize"]]+NoiseProfiles[np.random.randint(0,NoiseProfiles.shape[0]),:]
                EffLabeledTraces.append(trc)
    
 
    

        # for offset in range(0,len(FullTraceProfile)-250,3):

                # fullag = lags[lag].item()-lengs[lag].item()+offset
                # fullag = offset
                # trc =np.random.uniform(0.95,1.25)*FullTraceProfile[fullag:fullag+250]+NoiseProfiles[np.random.randint(0,NoiseProfiles.shape[0]),:]
                # trc =FullTraceProfile[fullag:fullag+250]+NoiseProfiles[np.random.randint(0,NoiseProfiles.shape[0]),:]
                # trcref = FullTrace[fullag:fullag+250]
                # trcexp = Misc.ZScoreTransform(traces[lag,:])
                # plt.plot(trc)
                # plt.plot(trcexp)
                # plt.plot(trcref)

                # plt.show()
                # EffLabeledTraces.append(trc)
            
                
            
            
            
            
            
            

    
    
    progress = 0
    for x in EffLabeledTraces:  
        if genome!='Other':
            Profiles.append(x) 
            # Profiles.append(np.flipud(x))
        else:
            prf = SIMTRC.GetFluorocodeProfile(np.squeeze(x),Gauss,Params["FragmentSize"])
            Profiles.append(prf) 


        progress+=1
        print('\r'+ str(progress) + " from " + str(len(EffLabeledTraces)) + " done", end = ""  )
        
        
    
    Label   = np.zeros(len(genomes))
    Label[genomes.index(genome)] = 1
    Labels  =  [ Label for x in Profiles]
    
    AllLabels.append(Labels)
    AllProfiles.append(Profiles)
    
    
Profiles= Dt.ToNPZ1D(AllProfiles,2,datatype='data')
Labels= Dt.ToNPZ1D(AllLabels,2,datatype='label')

print("Saving...")
Ds.SaveTrainingData(Profiles,Labels ,path="D:\Sergey\FluorocodeMain\FluorocodeMain\DataForTraining1D.npz")    
    

    
    
    
    

    
    
    
    
    
    
    
    
    
    
