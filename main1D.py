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


Wavelength = 576
NA = 1.4
size =340
pixelsz = 39.68
ResEnhancement=2
# genomes = ['Other','CP014051','CP039296','CP000948']
genomes = ['Other','NC_000913.3','CP014051','CP014787']
# genomes = ['CP014787']
# genomes = ['CP000948']
# genomes = ['CP014051','CP039296', 'CP009467.1','CP014787','CP000948']

AllLabels = []
AllProfiles = []
numsamples =2000

Dt = DataConverter()
Ds = DataLoader()



for genome in genomes:
    EffLabeledTraces = []
    SIMTRC     = SIMTraces.TSIMTraces(genome,1.66,0.34,0,'TaqI',pixelsz )  

    if genome!='Other':

        Profiles = []
        ReCuts     = SIMTRC.GetTraceRestrictions()
        ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(ReCuts)
        R      = RTG.RandomTraceGenerator(Misc.kbToPx(100000,SIMTRC),Misc.kbToPx(20000,SIMTRC),numsamples)
    else:
        R      = RTG.RandomTraceGenerator([],[],numsamples)
        
        Profiles = []
    if genome!='Other':
        Traces = R.GetRandomFixedLengthTraces(np.asarray(ReCutsInPx),size)
        for trace in Traces:
            for transform in range(0,40):
                EffLabeledTraces.extend(R.GetEffLabelingRate([trace],0.75))
    else:
        Traces = R.GetRandomTraces(85,55,380,60*numsamples)
        for trace in Traces:
            EffLabeledTraces.append(trace)

        # EffLabeledTraces.append(np.array(trace))
    
    

    
    Gauss  = Misc.GetGauss1d(size,  Misc.FWHMtoSigma(Misc.GetFWHM(Wavelength,NA,ResEnhancement)),pixelsz)
    
    progress = 0
    for x in EffLabeledTraces:  
        if genome!='Other':
            prf = SIMTRC.GetTraceProfile(x,Gauss,size,EffLabeledTraces)
            Profiles.append(prf) 
            Profiles.append(np.flipud(prf))
        else:
            prf = SIMTRC.GetFluorocodeProfile(np.squeeze(x),Gauss,size)
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
    

    
    
    
    

    
    
    
    
    
    
    
    
    
    
