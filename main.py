# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import Core.RandomTraceGenerator as RTG
import numpy as np
from ImGen.ImGen import TrainImageGenerator
from Core.DataHandler import DataConverter

genomes = ['CP000948','CP015409']

ImagesForAllGenomes = []
LabelsForAllGenomes = []

Dt = DataConverter()
for genome in genomes:
    SIMTRC     = SIMTraces.TSIMTraces(genome,1.75,0.34,0,'TaqI',40)  
    
    
    ReCuts     = SIMTRC.GetTraceRestrictions()
    ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(ReCuts)
    
    
    R      = RTG.RandomTraceGenerator(Misc.kbToPx(60000,SIMTRC),Misc.kbToPx(5000,SIMTRC),1000)
    Traces = R.stratsample(np.asarray(ReCutsInPx))
    
    EffLabeledTraces = R.GetEffLabelingRate(Traces,0.75)
    
    IMGEN = TrainImageGenerator('D:\Sergey\TrainDirectory', 500, 30, 512, 510 , 1.4, SIMTRC.PixelSize )
    AllImages, AllLabels   = IMGEN.Generate(LabeledTraces=EffLabeledTraces,numclass=genomes.index(genome))
    
    
    
    ImagesForAllGenomes.append(AllImages)
    LabelsForAllGenomes.append([Dt.ToOneHot(x,0,3) for x in AllLabels])
    
TrainingImages = Dt.ToNPZ(ImagesForAllGenomes)
TrainingLabels = Dt.ToNPZ(LabelsForAllGenomes)
    
    











