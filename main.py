# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:59:10 2020

@author: Sergey
"""
import Core.Misc as Misc
import Core.SIMTraces as SIMTraces
import Core.RandomTraceGenerator as RTG
import numpy as np

SIMTRC     = SIMTraces.TSIMTraces('CP000948',1.75,0.34,0,'TaqI',40)  


ReCuts     = SIMTRC.GetTraceRestrictions()
ReCutsInPx = SIMTRC.GetDyeLocationsInPixel(ReCuts)


R      = RTG.RandomTraceGenerator(Misc.kbToPx(40000,SIMTRC),Misc.kbToPx(5000,SIMTRC),10000)
Traces = R.stratsample(np.asarray(ReCutsInPx))

EffLabeledTraces = R.GetEffLabelingRate(Traces,0.75)





