##########################################################################################################
#                                                                                                        #
#                               Bachelor’s Thesis [Pompeu Fabra University]                              #
#           Psychoacoustic Correction of Binaural Audio Localization based on Head Orientation           #
#                                                                                                        #
#                                       ~ Pablo Rodríguez Cuenca ~                                       #
#                                                                                                        #
##########################################################################################################

import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
import time as tttt
from scipy.stats import circmean
from pysofaconventions import *
from cmath import e, pi, sin, cos

def circmedian(angs, unit='rad'): # From invalid index to scalar variable.
    #Radians!
    pdists = angs[np.newaxis, :2000] - angs[:2000, np.newaxis] #Discard of higher time and frequency values to improve performance and results (as there are a lot of outliers in those zones)
    if unit == 'rad':
        pdists = (pdists + np.pi) % (2 * np.pi) - np.pi
    elif unit == 'deg':
        pdists = (pdists +180) % (360.) - 180.
    pdists = np.abs(pdists).sum(1)

    if len(angs) % 2 != 0: # If angs is odd, take the center value
        return angs[np.argmin(pdists)]
    else: # If even, take the mean between the two minimum values
        index_of_min = np.argmin(pdists)
        min1 = angs[index_of_min]
        new_pdists = np.delete(pdists, index_of_min) # Remove minimum element from array and recompute
        new_angs = np.delete(angs, index_of_min) # Remove minimum element from array and recompute
        min2 = new_angs[np.argmin(new_pdists)]
        if unit == 'rad':
            return scipy.stats.circmean([min1, min2], high=np.pi, low=-np.pi)
        elif unit == 'deg':
            return scipy.stats.circmean([min1, min2], high=180., low=-180.)

t1 = tttt.time()  
print('Starting...\n ---------------')
##########################################################################################################
#FILE
#Set audio files directory
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic.sofa','r') 
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic_0_5m.sofa','r')
sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic_1m.sofa','r')
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic_2m.sofa','r')
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic_3m.sofa','r')

if sofa.isValid(): #Check validity
    print("SOFA file is valid")
else:
    print("Sofa file is _NOT_ valid")
    exit()
sfreq = int(sofa.getSamplingRate()) 
sourcePositions = sofa.getVariableValue('SourcePosition') 
angles = sourcePositions[:,0]
audios = sofa.getDataIR() # Read the data and create data array (8802, 2, 256)
numangles, numchannels, numsamples = audios.shape
PHI = numangles # Number of files


#CODEBOOK CREATION of HRTFs, ordered by audiofile
codebook = np.zeros((numangles, numchannels, numsamples//2+1),dtype=complex)
codebook[:] = np.fft.rfft(audios[:])

#RENDER WITH A FILE TO CREATE BINAURAL AUDIO
'''#"audios" index HOTKEYS:        #QU_KEMAR1m: 270 for 90º,  225 for 45º,  135 for -45º, 315 for 135º
#Original:                                      2211 for 90º, 1123 for 45º, 5523 for -45º, 3103 for 135º
#QU_KEMARfull:                                  990 for 90º,  945 for 45º,  875 for -45º
#NFHRIR:                                        90 for 90º
'''
ANGLE = 5  #/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
HTangle = 185 #Fake headtracker input angle /////////////////////////////////////////////////////////////////////////////////////////////////////

inputidx = list(sourcePositions[:,0]).index(ANGLE)

#data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\audiocheck.net_whitenoisegaussian.wav') #Open a mono wav gaussian noise.
data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\LNG_VocalLaugh_25.wav') #Open a mono wav file. I got this one from freesound https://freesound.org/people/Ryntjie/sounds/365061/ 
binaural_L = scipy.signal.fftconvolve(data,audios[inputidx,0,:]) #Convolve it with the hrtf of xxº azimuth and 0º elevation 
binaural_R = scipy.signal.fftconvolve(data,audios[inputidx,1,:]) #Convolve it with the hrtf of xxº azimuth and 0º elevation
binaural = np.asarray([binaural_L, binaural_R]) #.swapaxes(-1,0) to put the L/R channel first
sf.write('C:/Users/pablo/Desktop/audio_' + str(sourcePositions[inputidx,0]) + '.wav', binaural.swapaxes(-1,0), samplerate) #Save into a WAV file
print('Input audio convolved to ' + str(sourcePositions[inputidx,0]) + 'º')


#DFT WINDOWING
MU, LAMBDA, Y = scipy.signal.stft(binaural, sfreq, nperseg = numsamples) # Y to stft domain convertion with the same window size used above



t2 = tttt.time()  
print('Elapsed time = ' + str(t2-t1) + '\n ---------------')
print('Codebook ILDs and IPDs calculation...')
##########################################################################################################
#DIRECTION ESTIMATION
#Codebook ILDs and IPDs
ILDcb = np.transpose(np.abs(codebook[:,1,:])/np.abs(codebook[:,0,:])) # ILDcb has shape len(MU), numangles
IPDcb = np.transpose(np.angle(codebook[:,1,:]) - np.angle(codebook[:,0,:])) # IPDcb has shape len(MU), numangles
IPDcb = IPDcb % (2*np.pi) # Normalization between (0,2π)

t3 = tttt.time()  
print('Elapsed time = ' + str(t3-t1) + '\n ---------------')
print('Input file ILDs and IPDs...')
#Input file ILDs and IPDs + direction estimation
F = np.zeros((PHI))
phi_orig = np.zeros((len(LAMBDA), len(MU)), dtype=int)
ILDy = np.abs(Y[1, :,:])/np.abs(Y[0, :, :])
IPDy = np.angle(Y[1, :,:]) - np.angle(Y[0, :, :])
for lamb in range(len(LAMBDA)): # Time index
    for mu in range(len(MU)): # Frequency index
        F[:] = ILDcb[mu, :] / ILDy[mu, lamb] + ILDy[mu, lamb] / ILDcb[mu, :] - 2*np.cos( IPDy[mu, lamb] - IPDcb[mu, :])
        phi_orig[lamb, mu] = sourcePositions[int(np.argmin(F)), 0] # Phi index which gives the minimum F


t4 = tttt.time()    
print('Elapsed time = ' + str(t4-t1) + '\n ---------------')
print('Phi origin calculation...')
#PHI ORIGIN CALCULATION
ta1 = tttt.time()    
angle_mean = round(np.rad2deg( circmean(np.deg2rad(phi_orig[:])) ),1) # Rounded to 1 decimal
ta2 = tttt.time()
print('Estimated direction by mean: ' + str(angle_mean) + 'º' + ' in ' + str(ta2-ta1) + ' seconds')

ta3 = tttt.time()
p_o = np.round(phi_orig.flatten(),1) #2D phi_orig array into 1D (with 1 decimal)
angle_median = float(circmedian(p_o, 'deg'))
ta4 = tttt.time()
print('Estimated direction by median: ' + str(angle_median) + 'º' + ' in ' + str(ta4-ta3) + ' seconds')

ta5 = tttt.time()    
hist, _, _ = plt.hist(p_o, bins=360)
plt.close() #To delete hist plot
angle_hist = float(np.argmax(hist))
ta6 = tttt.time()
print('Estimated direction by histogram: ' + str(angle_hist) + 'º' + ' in ' + str(ta6-ta5) + ' seconds')

'''#PLOTS phi_orig
plt.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig, 0, 1)) #Inverted axis to have the plot we want
plt.title('pcolormesh KU100 indexes')
plt.xlabel('Time (λ)')
plt.ylabel('Frequency (μ)')
plt.colorbar(aspect=6)
plt.show()
'''

'''#DIRECTION ESTIMATION ACCURACY
#Difference calculations by method
difmean = abs(p_o - angle_mean)
difmedian = abs(p_o - angle_median)
difhisto = abs(p_o - angle_hist)

#Difference
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(p_o, difmean, s=10, label='Mean: ' + str(angle_mean) + 'º')
ax1.scatter(p_o, difmedian, s=10, label='Median: ' + str(angle_median) + 'º')
ax1.scatter(p_o, difhisto, s=10, label='Histogram: ' + str(angle_hist) + 'º')
plt.axhline(y=1, color='gray', linestyle='--')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Direction Estimation Accuracy')
plt.legend(loc='best')
plt.xlabel('Input angles (º)')
plt.ylabel('Estimated difference (º)')
plt.show()
#Time
plt.scatter( ['Mean', 'Median', 'Histogram'], [ta2-ta1, ta4-ta3, ta6-ta5])
plt.title('Direction Estimation Processing Time')
plt.ylabel('Time (s)')
plt.show()'''

t5 = tttt.time()  
print('Elapsed time = ' + str(t5-t1) + '\n ---------------')
print('Cue modification with headtracker...')
##########################################################################################################
#CUE MODIFICATION
HTphi = np.zeros((len(LAMBDA), len(MU)))
phi_dest = np.zeros((len(LAMBDA), len(MU)), dtype=int)

delta_phi[:,:] = round(angle_mean - HTangle, 2)
print('Input audio modified to ' + str(float(HTangle)) + 'º (dif ' + str(abs(round(angle_mean - HTangle, 2))) + ')')
phi_dest[:] = phi_orig[:] - delta_phi[:]
for x in range(len(LAMBDA)): #Threshold limiter for number over 359 or below 0          #1s aprox of time performance is lost here
    for y in range(len(MU)):
        if phi_dest[x, y] >= 360:
            phi_dest[x, y] = phi_dest[x, y]-360
        elif phi_dest[x, y] < 0:
            phi_dest[x, y] = phi_dest[x, y]+360         #to do usar módulo x % 360

deltaIPD = np.zeros((len(LAMBDA), len(MU)), dtype=complex)
G_IPDr = np.zeros((len(LAMBDA), len(MU)), dtype=complex)
G_IPDl = np.zeros((len(LAMBDA), len(MU)), dtype=complex)
G_ILDr = np.zeros((len(LAMBDA), len(MU)), dtype=complex)
G_ILDl = np.zeros((len(LAMBDA), len(MU)), dtype=complex)
for m in range(len(MU)): # Frequency index
    deltaIPD[:,m] = IPDcb[m,phi_dest[:,m]] - IPDcb[m,phi_orig[:,m]]

    G_IPDr[:,m] = np.exp(-1j*(deltaIPD[:,m]/2))    #eq13 from the paper
    G_IPDl[:,m] = np.exp(1j*(deltaIPD[:,m]/2))       #eq14 from the paper

    G_ILDr[:,m] = np.abs(codebook[phi_dest[:,m],1,m])/np.abs(codebook[phi_orig[:,m],1,m])             #eq15 from the paper
    G_ILDl[:,m] = np.abs(codebook[phi_dest[:,m],0,m])/np.abs(codebook[phi_orig[:,m],0,m])             #eq15 from the paper

Gr = G_ILDr * G_IPDr    #eq12 from the paper
Gl = G_ILDl * G_IPDl    #eq12 from the paper

G = np.asarray([np.transpose(Gl), np.transpose(Gr)]) #Transpose to mantain the original variable format
Ymod = Y * G

ibinaural_t, ibinaural = scipy.signal.istft(Ymod, sfreq, nperseg = numsamples) # Stft to Y domain convertion with the same window size used above
sf.write('C:/Users/pablo/Desktop/corrected_' + str(sourcePositions[inputidx,0]) + 'to' + str(float(HTangle)) + ' dif=' + str(delta_phi[1,1]) + '.wav', ibinaural[:,:len(binaural[0])].swapaxes(-1,0), samplerate) # Save into a WAV file


t999 = tttt.time()    
print('Total time: ' + str(t999-t1) + ' seconds')
print('This is the end')
##########################################################################################################