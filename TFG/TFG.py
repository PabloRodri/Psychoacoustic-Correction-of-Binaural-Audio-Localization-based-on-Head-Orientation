import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
import time as tttt
from scipy.stats import circmean
from pysofaconventions import *
from cmath import e, pi, sin, cos
#np.seterr(divide='ignore', invalid='ignore') #To avoid "divide by zero" or "divide by NaN" errors

t1 = tttt.time()  
print('Starting...\n ---------------')

#FILE
#Set audio files directory
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\KU100_44K_16bit_256tap_FIR_SOFA.sofa','r')
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic.sofa','r') 
sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic_1m.sofa','r')
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\NFHRIR_CIRC360_SOFA\HRIR_CIRC360_NF100.sofa','r')

if sofa.isValid(): #Check validity
    print("SOFA file is valid")
else:
    print("Sofa file is _NOT_ valid")
    exit()
sfreq = int(sofa.getSamplingRate()) 
#sofa.getSamplingRate()
#fs = sfreq[0]
sourcePositions = sofa.getVariableValue('SourcePosition') 
angles = sourcePositions[:,0]
#sP = sourcePositions.filled(sourcePositions.mean()) #To unmask the masked array and convert it into a ndarray
'''#Other SOFA commands
#sofa.printSOFAGlobalAttributes() to obtain info about how the SOFA file was "recorded"
#sofa.getPositionVariableInfo('SourcePosition') prints the info (units, coordinates): ('degree, degree, meter', 'spherical')
#samplingRateUnits = sofa.getSamplingRateUnits() if I want the units of the sampling frequency
#receiverPos = sofa.getReceiverPositionValues() to get the receiver ears position
'''

audios = sofa.getDataIR() # Read the data and create data array (8802, 2, 256)
numangles, numchannels, numsamples = audios.shape
PHI = numangles #Number of files



#CODEBOOK CREATION of HRTFs, ordered by audiofile
codebook = np.zeros((numangles, numchannels, numsamples//2+1),dtype=complex)
codebook[:] = np.fft.rfft(audios[:])
'''codebook = np.zeros((numfiles, numchannels, numsamples),dtype=complex)
codebook[:,0] = np.fft.fft(audios[:,0]) #run by "numfiles"
codebook[:,1] = np.fft.fft(audios[:,1])
#DFT sample frequencies, ordered by audiofile
#codebook_f[:] = np.fft.fftfreq(audios[:,0].shape[-1]) #L and R share the same values
'''


''' #RULE OUT ELEVATION
#Taking only samples with zero elevation
ind_azim = []
azimuth = []
for x in range(numfiles):
    #if (sP[:][x][1] == 0) == True:
        #ind_azim.append(x)
    if (sP[:][x][1] == 0) == True: #v2 - if there are more than one distance values
        if (sP[:][x][2] == 1) == True:
            ind_azim.append(x)
#azimuth = np.ones((len(ind_azim))) #Quadrant correction v2.5
#x = 0
for a in ind_azim:
    sPaz = float(sP[a][0])
    #if sPaz > 180: #Quadrant correction v1
        #sPaz = int(-(sPaz-180))
    if sPaz > 180: #Quadrant correction v2
        sPaz = int(sPaz-180)
    elif sPaz <= 180:
        sPaz = int(sPaz+180)
        if sPaz == 360:
            sPaz = 0
    #if x < 180: #Quadrant correction v2.5
        #azimuth[x+180] = np.round(sPaz)
    #else:
        #azimuth[x-180] = np.round(sPaz)
    #x+=1
    #azimuth.append(sPaz) #Create array with the ideal angles for the plot
    azimuth.append(np.round(sPaz)) #Create array with the ideal angles for the plot
'''



#RENDER WITH A FILE TO CREATE BINAURAL AUDIO
#data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\audiocheck.net_whitenoisegaussian.wav') #Open a mono wav gaussian noise.
data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\LNG_VocalLaugh_25.wav') #Open a mono wav file. I got this one from freesound https://freesound.org/people/Ryntjie/sounds/365061/ 
binaural_L = scipy.signal.fftconvolve(data,audios[225,0,:]) #Convolve it with the hrtf of xxº azimuth and 0º elevation 
binaural_R = scipy.signal.fftconvolve(data,audios[225,1,:]) #Convolve it with the hrtf of xxº azimuth and 0º elevation
binaural = np.asarray([binaural_L, binaural_R]) #.swapaxes(-1,0) to put the L/R channel first
'''#"audios" index HOTKEYS:        #QU_KEMAR1m: 270 for 90º,  225 for 45º,  135 for -45º, 315 for 135º
#Original:                                      2211 for 90º, 1123 for 45º, 5523 for -45º, 3103 for 135º
#QU_KEMARfull:                                  990 for 90º,  945 for 45º,  875 for -45º
#NFHRIR:                                        90 for 90º
'''
#sf.write('C:/Users/pablo/Desktop/resultadobinaural.wav', binaural.swapaxes(-1,0), samplerate) #Save into a WAV file

#DFT WINDOWING
''' #Old DFT Windowing:
BINAURAL_L = scipy.signal.stft(binaural_L, samplerate/2) #STFT of the binaural input to convert it to the time-freq domain    #nperseg=256 by default       #Samplingrate/2 to decrease processing time
BINAURAL_R = scipy.signal.stft(binaural_R, samplerate/2) #STFT of the binaural input to convert it to the time-freq domain                                  #Samplingrate/2 to decrease processing time
BINAURAL_LR = np.asarray([binaural_L, binaural_R])
MU, LAMBDA, Y = scipy.signal.spectrogram(BINAURAL_LR, samplerate, mode='complex')
BINAURAL = np.swapaxes(Y, 1, 2)'''
MU, LAMBDA, Y = scipy.signal.stft(binaural, sfreq, nperseg = numsamples) #Y to stft domain convertion with the same window size used above
#sf.write('C:/Users/pablo/Desktop/resultadobinauralpostwindowing.wav', BINAURAL.swapaxes(-1,0), samplerate) #Save into a WAV file



t2 = tttt.time()  
print('Elapsed time = ' + str(t2-t1) + '\n ---------------')
print('Codebook ILDs and IPDs calculation...')

#Codebook ILDs and IPDs
'''#Old ILDh & IPDh Loop
ILDh = np.zeros((len(MU), len(PHI)))
IPDh = np.zeros((len(MU), len(PHI)))
for lamb in range(len(LAMBDA)): #Time index                       #len(LAMBDA)
    for mu in range(len(MU)): #Frequency index
        #Codebook differences (eqs.6-7)
            ILDh[mu, :] = np.abs(codebook[ind_azim[:],1,mu])/np.abs(codebook[ind_azim[:],0,mu])
            IPDh[mu, :] = np.angle(codebook[ind_azim[:],1,mu]) - np.angle(codebook[ind_azim[:],0,mu]) #We use the second part of the eq as it's in rad and not in grad as the other one
'''
ILDcb = np.transpose(np.abs(codebook[:,1,:])/np.abs(codebook[:,0,:])) # ILDcb has shape len(MU), numangles
IPDcb = np.transpose(np.angle(codebook[:,1,:]) - np.angle(codebook[:,0,:])) # IPDcb has shape len(MU), numangles
IPDcb = IPDcb % (2*np.pi) #Normalization between (0,2π)


t3 = tttt.time()  
print('Elapsed time = ' + str(t3-t1) + '\n ---------------')
print('Input file ILDs and IPDs + Phi_origin calculation...')

#Input file ILDs and IPDs + direction estimation
'''#Old ILD & IPD Loop
ILD = np.zeros((len(LAMBDA), len(MU)))
IPD = np.zeros((len(LAMBDA), len(MU)))
F = np.zeros((len(PHI)))
phi_orig = np.zeros((len(LAMBDA), len(MU)))
for lamb in range(len(LAMBDA)): #Time index                       #len(LAMBDA)
                        #MU Frequency index
    #Input signal differences (eqs.4-5)
    ILD[lamb, :] = np.abs(BINAURAL[1,lamb,:]/BINAURAL[0,lamb,:])
    IPD[lamb, :] = np.angle(BINAURAL[1,lamb,:]) - np.angle(BINAURAL[0,lamb,:]) #float to prevent floor-division -> floor(int)/0   #We use the second part of the eq as it's in rad and not in grad as the other one
    for mu in range(len(MU)): #Frequency index
        F[:] = np.angle(ILDh[mu,:]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu,:]) - 2*np.cos(IPD[lamb, mu] - IPDh[mu,:]) #Azimuth estimation for every TF bin (output needs to be a complex number)
        phi_orig[lamb, mu] = np.argmin(F.tolist()) #phi index which gives the minimum F
'''
F = np.zeros((PHI))
phi_orig = np.zeros((len(LAMBDA), len(MU)))
ILDy = np.abs(Y[1, :,:])/np.abs(Y[0, :, :])
IPDy = np.angle(Y[1, :,:]) - np.angle(Y[0, :, :])
for lamb in range(len(LAMBDA)): #Time index                 #len(LAMBDA)
    for mu in range(len(MU)): #Frequency index
        F[:] = ILDcb[mu, :] / ILDy[mu, lamb] + ILDy[mu, lamb] / ILDcb[mu, :] - 2*np.cos( IPDy[mu, lamb] - IPDcb[mu, :])
        phi_orig[lamb, mu] = np.argmin(F) #phi index which gives the minimum F

#avg_angle = np.rad2deg( circmean(np.deg2rad(phi_orig[:])) )
avg_angle = np.rad2deg(  circmean(np.deg2rad(sourcePositions[ int(np.median(phi_orig[:])) ,0] ))  )
print('Estimated direction: ' + str(avg_angle) + 'º')

t4 = tttt.time()    
print('Total time (plots not included): ' + str(t4-t1) + ' seconds')


#PLOTS
plt.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig, 0, 1)) #Inverted axis to have the plot we want
plt.title('pcolormesh KU100 indexes')
plt.xlabel('Time (λ)')
plt.ylabel('Frequency (μ)')
plt.colorbar(aspect=6)
plt.show()

print('This is the end')


#fig, (ax1, ax2) = plt.subplots(2,1)
#ax1.plot(np.swapaxes(phi_orig, 0, 1))
#plt.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig, 0, 1), cmap=plt.cm.get_cmap('hsv')) #Inverted axis to have the plot we want
#ax1.set(ylabel='Frequency (μ)', title='KU100')
#ax2.set(xlabel='Time (λ)', ylabel='Frequency (μ)')
#axxx1 = fig.colorbar(axx1, ax=ax1, aspect=6)
#axxx2 = fig.colorbar(axx2, ax=ax2, aspect=6) #, ticks=[-np.pi/2, 0, -np.pi/2])
#axxx1.set_label('\u03C6' + 'orig')
#axxx2.set_label('\u03C6' + 'orig 2')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) #To simplify yaxis format
#plt.setp((ax1, ax2), xticks=[-90, -45, 0, 45, 90]) #To control the lower xaxis bins
#plt.subplots_adjust(hspace = 0.25) #Adjust height between subplots
