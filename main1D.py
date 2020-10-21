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


Wavelength = 510 
NA = 1.4
size = 512

genomes = ['CP014787','CP000948']
AllLabels = []
AllProfiles = []
numsamples =3000

Dt = DataConverter()
Ds = DataLoader()



for genome in genomes:
    EffLabeledTraces = []

    SIMTRC     = SIMTraces.TSIMTraces(genome,1.75,0.34,0,'TaqI',80)  
    Profiles = []

    
    ReCuts     = SIMTRC.GetTraceRestrictions()
    ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(ReCuts)
    # ReCutsInPx = ReCutsInPx[1:10000]

    
    R      = RTG.RandomTraceGenerator(Misc.kbToPx(100000,SIMTRC),Misc.kbToPx(20000,SIMTRC),numsamples)

    Traces = R.GetRandomFixedLengthTraces(np.asarray(ReCutsInPx),size)
    for trace in Traces:
        for transform in range(0,30):
            EffLabeledTraces.extend(R.GetEffLabelingRate([trace],0.75))
        # EffLabeledTraces.append(np.array(trace))
    
    

    
    Gauss  = Misc.GetGauss1d(size,  Misc.FWHMtoSigma(Misc.GetFWHM(Wavelength,NA)),80)
    
    progress = 0
    for x in EffLabeledTraces:      
        Profiles.append(SIMTRC.GetTraceProfile(x,Gauss,size,EffLabeledTraces)) 
        progress+=1
        print('\r'+ str(progress) + " from " + str(numsamples) + " done", end = ""  )
        
        
    
    Label   = np.zeros(len(genomes))
    Label[genomes.index(genome)] = 1
    Labels  =  [ Label for x in EffLabeledTraces]
    
    AllLabels.append(Labels)
    AllProfiles.append(Profiles)
    
    
Profiles= Dt.ToNPZ1D(AllProfiles,2,datatype='data')
Labels= Dt.ToNPZ1D(AllLabels,2,datatype='label')

print("Saving...")
Ds.SaveTrainingData(Profiles,Labels ,path="D:\Sergey\FluorocodeMain\FluorocodeMain\DataForTraining1D.npz")    
    

    
    
    
    

    
    
    
    
    
    
    
    
    
    
