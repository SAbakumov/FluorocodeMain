"""
Created on Sun Nov  8 22:07:08 2020

@author: Sergey
"""

import Core.Misc as Misc
import numpy as np

size = 250
numprofiles = 20000
Profiles = []
for i in range(0,numprofiles):
   prof = np.random.uniform(0.1,0.35)*  Misc.GetPerlinNoise(size)
   Profiles.append(prof)
   
np.save("D:/Sergey/FluorocodeMain/FluorocodeMain/Core/Noise/NoiseProfiles.npy", Profiles)
   
   
   