import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf
from pysofaconventions import *
from cmath import e, pi, sin, cos
np.seterr(divide='ignore', invalid='ignore') #To avoid "divide by zero" or "divide by NaN" errors


#FILE
#Set audio files directory
sofa = SOFAFile(r'C:\Users\pablo\source\repos\TFG\Muestras\KU100_44K_16bit_256tap_FIR_SOFA.sofa','r') #sofa.printSOFAGlobalAttributes() to obtain info about how the SOFA file was "recorded"
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
numsamples = PHI = len(audios[0][0])



#PEAKS ARRAYS and CODEBOOK CREATION
time = np.zeros((numfiles, 256))
peaks = np.zeros((numfiles, 2))
tpeaks = np.zeros((numfiles, 2))
codebook_f = np.zeros((numfiles, numsamples))
codebook = np.zeros((numfiles, numchannels, numsamples),dtype=complex)
for x in range(numfiles): 
    #3D array for audios + time array
    time[x] = np.arange(0,int(len(audios[x,0]))) / sfreq #Time array, ordered by audiofile
    #Previous ILD & ITD work (from input audio)
    peaks[x] = np.amax(audios[x], axis=1) #Peak array, ordered by audiofile
    extractionL, = np.where(audios[x,0]==peaks[x,0])
    extractionR, = np.where(audios[x,1]==peaks[x,1])
    tpeaks[x] = [max(extractionL),max(extractionR)] #Time position array, ordered by audiofile

    #DFT sample frequencies, ordered by audiofile
    codebook_f[x] = np.fft.fftfreq(audios[x,0].shape[-1]) #L and R share the same values
    #Codebook of HRTFs, ordered by audiofile
    codebook[x,0] = np.fft.fft(audios[x,0])
    codebook[x,1] = np.fft.fft(audios[x,1])


#ILD and IPD CALCULATION
#Taking only samples with zero elevation
ind_azim = []
ILD = []
ITD = []
IPD = []
azimuth = []
idealazimuth = []
for x in range(numfiles):
    if (sP[:][x][1] == 0) == True:
        ind_azim.append(x)
ILDc = np.zeros((360, int(numsamples/2))) #Only of the first half of samples as it's symmetrical
IPDc = np.zeros((360, int(numsamples/2))) #Only of the first half of samples as it's symmetrical
x=0
for a in ind_azim:
    azimuth.append(float(sP[a][0])) #Azimuth array [0,360)
    sPaz = float(sP[a][0])
    if sPaz > 180: #Quadrant correction
        sPaz = int(-(sPaz-180))
    idealazimuth.append(sPaz) #Azimuth array [-180,180)
    
    if (azimuth[x]).is_integer() == True:
        ILD.append(-20 * mt.log10(abs(peaks[a,0]/peaks[a,1]))) #Peak level difference (in dB). Negative to make the negative side of the grids the right ear as in https://www.york.ac.uk/sadie-project/database.html
        ITD.append(10**(-4)*(tpeaks[a,0]-tpeaks[a,1])) #Time difference between peaks (*10^-4). 
        IPD.append(-20 * mt.log10(abs(peaks[a,0]/peaks[a,1]))*2*np.pi*sfreq) #IPD = ILD*2πf

    for b in range(int(numsamples/2)):
        ILDc[int(idealazimuth[x]), b] = -10 * mt.log10(np.abs(codebook[a,1,b])/np.abs(codebook[a,0,b])) #ILD (following eq 6 from the paper) (in dB)
        IPDc[int(idealazimuth[x]), b] = np.angle((codebook[a,1,b]/codebook[a,0,b]), deg=False) #IPD in radians (following eq 7 from the paper)
    x+=1



#RENDER WITH A FILE TO CREATE BINAURAL AUDIO
#data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\Otras\audiocheck.net_whitenoisegaussian.wav') #Open a mono wav gaussian noise.
data, samplerate = sf.read(r'C:\Users\pablo\source\repos\TFG\Muestras\LNG_VocalLaugh_25.wav') #Open a mono wav file. I got this one from freesound https://freesound.org/people/Ryntjie/sounds/365061/ 
binaural_L = scipy.signal.fftconvolve(data,audios[2211,0,:]) #Convolve it with the hrtf of 90º azimuth and 0º elevation
binaural_R = scipy.signal.fftconvolve(data,audios[2211,1,:]) #Convolve it with the hrtf of 90º azimuth and 0º elevation
#2211 for 90º, 1123 for 45º, 3103 for 135º, 5523 for -45º
binaural = np.asarray([binaural_L, binaural_R]) #.swapaxes(-1,0) to put the L/R channel first
#sf.write('C:/Users/pablo/Desktop/resultadobinaural.wav', binaural.swapaxes(-1,0), samplerate) #Save into a WAV file

#DFT WINDOWING

BINAURAL_L = scipy.signal.stft(binaural_L, samplerate) #STFT of the binaural input to convert it to the time-freq domain    #nperseg=256 by default
BINAURAL_R = scipy.signal.stft(binaural_R, samplerate) #STFT of the binaural input to convert it to the time-freq domain 
BINAURAL = np.asarray([binaural_L, binaural_R])

MU, LAMBDA, Sxx = scipy.signal.spectrogram(BINAURAL, samplerate)
BINAURAL = np.swapaxes(Sxx, 1, 2)
_, _, Sxx_phase = scipy.signal.spectrogram(binaural, samplerate, mode='angle')
BINAURAL_phase = np.swapaxes(Sxx_phase, 1, 2)
#sf.write('C:/Users/pablo/Desktop/resultadobinauralpostwindowing.wav', BINAURAL.swapaxes(-1,0), samplerate) #Save into a WAV file
#print(np.allclose(binaural, BINAURAL))

#plt.pcolormesh(LAMBDA, MU, 10*np.log10(Sxx[0]))
#plt.xlabel('Time (λ)')
#plt.ylabel('Frequency (μ)')
#plt.show()


#print('Hi World')
'''
#DIRECTION ESTIMATION (eq.11 from the paper)
ILDi = -20 * mt.log10(abs(np.amax(BINAURAL[0])/np.amax(BINAURAL[1]))) 
IPDi = -20 * mt.log10(abs(np.amax(BINAURAL[0])/np.amax(BINAURAL[1])))*2*np.pi*samplerate
dir = []
for b in range(360):
    dir.append(  )  #Azimuth estimation for every TF bin
'''

#PHI[:] = ind_azim[:]
#MU = range(int(numsamples/2))

''''
for lambda in LAMBDA: #Time index
    for mu in MU: #Frequency index
        #Input signal differences (eqs.4-5)
        ILD = abs(np.amax(BINAURAL[1])/np.amax(BINAURAL[0]))
        IPD = ILD*2*np.pi*sfreq  #IPD(lambda,mu) = ILD(lambda,mu)*2πf
        for phi in PHI:  #ind_azim a.k.a. file number
            #Codebook differences (eqs.6-7)
            ILDh[mu, phi] = np.abs(codebook[phi,1,mu])/np.abs(codebook[phi,0,mu])
            IPDh[mu, phi] = np.angle((codebook[phi,1,mu]/codebook[phi,0,mu]), deg=False)
            F.append(  np.angle(ILDh[mu,phi]/ILD) + (ILD/ILDh[mu,phi]) - 2*np.cos(IPD - IPDh[mu,phi])  ) #Azimuth estimation for every TF bin (output needs to be a complex number)
        phi_orig[lambda, mu] = np.min(F) #phi which gives the minimum F
'''
ILD = np.zeros((len(LAMBDA), len(MU)))
IPD = np.zeros((len(LAMBDA), len(MU)))
IPD2 = np.zeros((len(LAMBDA), len(MU)))
ILDh = np.zeros((len(MU), PHI))
IPDh = np.zeros((len(MU), PHI))
F = np.zeros((PHI))
F2 = np.zeros((PHI))
phi_orig = np.zeros((len(LAMBDA), len(MU)))
phi_orig2 = np.zeros((len(LAMBDA), len(MU)))
for lamb in range(len(LAMBDA)): #Time index
    for mu in range(len(MU)): #Frequency index
        #Input signal differences (eqs.4-5)
        ILD[lamb, mu] = abs(BINAURAL[1,lamb,mu]/BINAURAL[0,lamb,mu])
        IPD[lamb, mu] = float(BINAURAL_phase[1,lamb,mu])/BINAURAL_phase[0,lamb,mu] #float to prevent floor-division -> floor(int)/0
        IPD2[lamb, mu] = BINAURAL_phase[1,lamb,mu]-BINAURAL_phase[0,lamb,mu]
        '''
            for phi in range(PHI):  #ind_azim a.k.a. file number
            #Codebook differences (eqs.6-7)
            ILDh[mu, phi] = np.abs(codebook[phi,1,mu])/np.abs(codebook[phi,0,mu])
            IPDh[mu, phi] = np.angle((codebook[phi,1,mu]/codebook[phi,0,mu]), deg=False)
            F[phi] = np.angle(ILDh[mu,phi]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu,phi]) - 2*np.cos(IPD[lamb, mu] - IPDh[mu,phi]) #Azimuth estimation for every TF bin (output needs to be a complex number)
            F2[phi] = np.angle(ILDh[mu,phi]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu,phi]) - 2*np.cos(IPD2[lamb, mu] - IPDh[mu,phi]) #Azimuth estimation for every TF bin (output needs to be a complex number)
        '''
        phi = 0
        ILDh[mu] = [np.abs(codebook[phi,1,mu])/np.abs(codebook[phi,0,mu]) for phi in range(PHI)]
        IPDh[mu] = [np.angle((codebook[phi,1,mu]/codebook[phi,0,mu]), deg=False) for phi in range(PHI)]
        F = [np.angle(ILDh[mu,phi]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu,phi]) - 2*np.cos(IPD[lamb, mu] - IPDh[mu,phi]) for phi in range(PHI)] #Azimuth estimation for every TF bin (output needs to be a complex number)
        F2 = [np.angle(ILDh[mu,phi]/ILD[lamb, mu]) + (ILD[lamb, mu]/ILDh[mu,phi]) - 2*np.cos(IPD2[lamb, mu] - IPDh[mu,phi]) for phi in range(PHI)] #Azimuth estimation for every TF bin (output needs to be a complex number)
        
        phi_orig[lamb, mu] = np.min(F) #phi which gives the minimum F
        phi_orig2[lamb, mu] = np.min(F2) #phi which gives the minimum F
       
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(np.swapaxes(phi_orig, 0, 1))
ax2.plot(np.swapaxes(phi_orig2, 0, 1))
axx1 = ax1.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig, 0, 1), cmap=plt.cm.get_cmap('hsv')) #Inverted axis to have the plot we want
axx2 = ax2.pcolormesh(LAMBDA, MU, np.swapaxes(phi_orig2, 0, 1), cmap=plt.cm.get_cmap('hsv')) #Inverted axis to have the plot we want
ax1.set(ylabel='Frequency (μ)', title='KU100')
ax2.set(xlabel='Time (λ)', ylabel='Frequency (μ)')

axxx1 = fig.colorbar(axx1, ax=ax1, aspect=6)
axxx2 = fig.colorbar(axx2, ax=ax2, aspect=6) #, ticks=[-np.pi/2, 0, -np.pi/2])
axxx1.set_label('\u03C6' + 'orig')
axxx2.set_label('\u03C6' + 'orig 2')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) #To simplify yaxis format
#plt.setp((ax1, ax2), xticks=[-90, -45, 0, 45, 90]) #To control the lower xaxis bins
plt.subplots_adjust(hspace = 0.25) #Adjust height between subplots
plt.show()



#PLOTS
fig, (ax1, ax2) = plt.subplots(2,1)
axx1 = ax1.pcolormesh(ILDc.swapaxes(-1,0)) #Inverted axis to have the plot we want
axx2 = ax2.pcolormesh(IPDc.swapaxes(-1,0), cmap=plt.cm.get_cmap('hsv')) #Inverted axis to have the plot we want
ax1.set(ylabel='Frequency (kHz)', title='KU100')
ax2.set(xlabel='\u03C6 (º)', ylabel='Frequency (kHz)')

axxx1 = fig.colorbar(axx1, ax=ax1, aspect=6)
axxx2 = fig.colorbar(axx2, ax=ax2, aspect=6) #, ticks=[-np.pi/2, 0, -np.pi/2])
axxx1.set_label('ILD Codebook (dB)')
axxx2.set_label('IPD Codebook (rad)')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) #To simplify yaxis format
#plt.setp((ax1, ax2), xticks=[-90, -45, 0, 45, 90]) #To control the lower xaxis bins
plt.subplots_adjust(hspace = 0.25) #Adjust height between subplots
plt.show()
print("End")