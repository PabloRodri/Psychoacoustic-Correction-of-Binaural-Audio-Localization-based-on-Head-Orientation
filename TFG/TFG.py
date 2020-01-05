import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
import time as tttt
from scipy.stats import circmean
from pysofaconventions import *
from cmath import e, pi, sin, cos
np.seterr(divide='ignore', invalid='ignore') #To avoid "divide by zero" or "divide by NaN" errors

t1 = tttt.time()  
print('Starting...')

#FILE
#Set audio files directory

#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\KU100_44K_16bit_256tap_FIR_SOFA.sofa','r') #sofa.printSOFAGlobalAttributes() to obtain info about how the SOFA file was "recorded"
sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\QU_KEMAR_anechoic.sofa','r') #sofa.printSOFAGlobalAttributes() to obtain info about how the SOFA file was "recorded"
#sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\NFHRIR_CIRC360_SOFA\HRIR_CIRC360_NF100.sofa','r') #sofa.printSOFAGlobalAttributes() to obtain info about how the SOFA file was "recorded"

if sofa.isValid(): #Check validity
    print("SOFA file is valid")
else:
    print("Sofa file is _NOT_ valid")
    exit()
sourcePositions = sofa.getVariableValue('SourcePosition') #sofa.getPositionVariableInfo('SourcePosition') prints the info (units, coordinates): ('degree, degree, meter', 'spherical')
sP = sourcePositions.filled(sourcePositions.mean()) #To unmask the masked array and convert it into a ndarray
audios = sofa.getDataIR() # Read the data and create data array (8802, 2, 256)
#receiverPos = sofa.getReceiverPositionValues() to get the receiver ears position
sfreq = int(sofa.getSamplingRate()) #samplingRateUnits = sofa.getSamplingRateUnits() if I want the units of the sampling frequency
numfiles = MU_cb = len(audios)
numchannels = len(audios[0])
numsamples = len(audios[0][0])



#CODEBOOK CREATION of HRTFs, ordered by audiofile
codebook = np.zeros((numfiles, numchannels, numsamples),dtype=complex)
codebook[:,0] = np.fft.fft(audios[:,0]) #run by "numfiles"
codebook[:,1] = np.fft.fft(audios[:,1])
#DFT sample frequencies, ordered by audiofile
#codebook_f[:] = np.fft.fftfreq(audios[:,0].shape[-1]) #L and R share the same values



#RULE OUT ELEVATION
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



#RENDER WITH A FILE TO CREATE BINAURAL AUDIO
#data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\audiocheck.net_whitenoisegaussian.wav') #Open a mono wav gaussian noise.
data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\LNG_VocalLaugh_25.wav') #Open a mono wav file. I got this one from freesound https://freesound.org/people/Ryntjie/sounds/365061/ 
binaural_L = scipy.signal.fftconvolve(data,audios[990,0,:]) #Convolve it with the hrtf of 90º azimuth and 0º elevation #1st 990(90)/945(45)/875(-45)   2nd 90
binaural_R = scipy.signal.fftconvolve(data,audios[990,1,:]) #Convolve it with the hrtf of 90º azimuth and 0º elevation
#2211 for 90º, 1123 for 45º, 3103 for 135º, 5523 for -45º
binaural = np.asarray([binaural_L, binaural_R]) #.swapaxes(-1,0) to put the L/R channel first
#sf.write('C:/Users/pablo/Desktop/resultadobinaural.wav', binaural.swapaxes(-1,0), samplerate) #Save into a WAV file

#DFT WINDOWING
BINAURAL_L = scipy.signal.stft(binaural_L, samplerate/2) #STFT of the binaural input to convert it to the time-freq domain    #nperseg=256 by default       #Samplingrate/2 to decrease processing time
BINAURAL_R = scipy.signal.stft(binaural_R, samplerate/2) #STFT of the binaural input to convert it to the time-freq domain                                  #Samplingrate/2 to decrease processing time
BINAURAL_LR = np.asarray([binaural_L, binaural_R])
MU, LAMBDA, Sxx = scipy.signal.spectrogram(BINAURAL_LR, samplerate, mode='complex')
PHI = np.arange(360) #360 gradians
BINAURAL = np.swapaxes(Sxx, 1, 2)
#sf.write('C:/Users/pablo/Desktop/resultadobinauralpostwindowing.wav', BINAURAL.swapaxes(-1,0), samplerate) #Save into a WAV file

t2 = tttt.time()  
print('Codebook ILDs and IPDs calculation...   Elapsed time = ' + str(t2-t1))
#Codebook ILDs and IPDs
ILDh = np.zeros((len(MU), len(PHI)))
IPDh = np.zeros((len(MU), len(PHI)))
'''for lamb in range(5): #Time index                       #len(LAMBDA)
    for mu in range(len(MU)): #Frequency index
        #Codebook differences (eqs.6-7)
        for x in range(len(ind_azim)): #ind_azim = [12, 35, 57, 79, 101, 123, 145, 167, 189, 211, 233, 255, 277, 310, ...]     azimuth = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, ...]
            ILDh[mu, int(azimuth[x])] = np.abs(codebook[ind_azim[x],1,mu])/np.abs(codebook[ind_azim[x],0,mu])
            IPDh[mu, int(azimuth[x])] = np.angle(codebook[ind_azim[x],1,mu]) - np.angle(codebook[ind_azim[x],0,mu]) #We use the second part of the eq as it's in rad and not in grad as the other one
'''
for lamb in range(5): #Time index                       #len(LAMBDA)
    for mu in range(len(MU)): #Frequency index
        #Codebook differences (eqs.6-7)
        #ind_azim = [12, 35, 57, 79, 101, 123, 145, 167, 189, 211, 233, 255, 277, 310, ...]     azimuth = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, ...]
            ILDh[mu, :] = np.abs(codebook[ind_azim[:],1,mu])/np.abs(codebook[ind_azim[:],0,mu])
            IPDh[mu, :] = np.angle(codebook[ind_azim[:],1,mu]) - np.angle(codebook[ind_azim[:],0,mu]) #We use the second part of the eq as it's in rad and not in grad as the other one



t3 = tttt.time()  
print('Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = ' + str(t3-t1))
#Input file ILDs and IPDs + Phi_origin
ILD = np.zeros((len(LAMBDA), len(MU)))
IPD = np.zeros((len(LAMBDA), len(MU)))
F = np.zeros((len(PHI)))
phi_orig = np.zeros((len(LAMBDA), len(MU)))
'''for lamb in range(5): #Time index                       #len(LAMBDA)
    for mu in range(len(MU)): #Frequency index
        #Input signal differences (eqs.4-5)
        ILD[lamb, mu] = np.abs(BINAURAL[1,lamb,mu]/BINAURAL[0,lamb,mu])
        IPD[lamb, mu] = np.angle(BINAURAL[1,lamb,mu]) - np.angle(BINAURAL[0,lamb,mu]) #float to prevent floor-division -> floor(int)/0   #We use the second part of the eq as it's in rad and not in grad as the other one
        F = [np.angle(ILDh[mu,phi]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu, phi]) - 2*np.cos(IPD[lamb, mu] - IPDh[mu, phi]) for phi in PHI] #Azimuth estimation for every TF bin (output needs to be a complex number
        phi_orig[lamb, mu] = np.argmin(F) #phi index which gives the minimum F
                            #phi_orig[lamb, mu] = np.unravel_index(np.argmin(F, axis=None), F.shape)
                            #F[phi_orig[lamb, mu]]
        #print('lambda:' + str(lamb) + ' mu:' + str(mu))
'''
for lamb in range(5): #Time index                       #len(LAMBDA)
                        #MU Frequency index
    #Input signal differences (eqs.4-5)
    ILD[lamb, :] = np.abs(BINAURAL[1,lamb,:]/BINAURAL[0,lamb,:])
    IPD[lamb, :] = np.angle(BINAURAL[1,lamb,:]) - np.angle(BINAURAL[0,lamb,:]) #float to prevent floor-division -> floor(int)/0   #We use the second part of the eq as it's in rad and not in grad as the other one
    for mu in range(len(MU)): #Frequency index
        F = [np.angle(ILDh[mu,phi]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu, phi]) - 2*np.cos(IPD[lamb, mu] - IPDh[mu, phi]) for phi in PHI] #Azimuth estimation for every TF bin (output needs to be a complex number
        phi_orig[lamb, mu] = np.argmin(F) #phi index which gives the minimum F
                            #phi_orig[lamb, mu] = np.unravel_index(np.argmin(F, axis=None), F.shape)
                            #F[phi_orig[lamb, mu]]
        #print('lambda:' + str(lamb) + ' mu:' + str(mu))

avg_angle = np.rad2deg( circmean(np.deg2rad(phi_orig[:5])) )                    #phi_orig[:5]!!!
print('Origin phi is: ' + str(avg_angle))

t4 = tttt.time()    
print('Total time (plots not included): ' + str(t4-t1) + ' seconds')

#Full 1 - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 1673.2812156677246                          27.888020261128744 minutes
#Full 1 - Total time (plots not included): 2368.8777115345 seconds                                                          39.48129519224167 minutes
#Full 2 - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 897.7831192016602                           14.963051986694336666666666666667 minutes
#Full 2 - Total time (plots not included): 1302.7927815914154 seconds                                                       21.71321302652359‬ minute (laptop just started and without using it)
#Full 3(STFT samplerate/2) - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 868,56902384758          14,476150397459666666666666666667‬ minutes !!
#Full 3(STFT samplerate/2) - Total time (plots not included): 1298,391851902008 seconds                                     21,6398641983668‬ minutes !!

#Partial 1 - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 18.076984167099
#Partial 1 - Total time (plots not included): 24.700349807739258 seconds
#Partial 2(STFT samplerate/2) - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 16.646246671676636
#Partial 2(STFT samplerate/2) - Total time (plots not included): 23.171047687530518 seconds
#Partial 3(STFT samplerate/4) - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 21.892667293548584
#Partial 3(STFT samplerate/4) - Total time (plots not included): 29.640634775161743 seconds
#Partial 4(STFT samplerate/2) - Input file ILDs and IPDs + Phi_origin calculation...   Elapsed time = 11.035381317138672!!
#Partial 4(STFT samplerate/2) - Total time (plots not included): 15.686950922012329 seconds!!

#PLOTS
plt.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig, 0, 1)) #Inverted axis to have the plot we want
plt.title('pcolormesh KU100')
plt.xlabel('Time (λ)')
plt.ylabel('Frequency (μ)')
plt.colorbar(aspect=6)
plt.show()

print('This is the end')
'''
plt.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig, 0, 1), cmap=plt.cm.get_cmap('hsv')) #Inverted axis to have the plot we want
plt.title('plot KU100')
plt.xlabel('Time? (λ)')
plt.ylabel('Frequency? (μ)')
plt.show()
'''

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
