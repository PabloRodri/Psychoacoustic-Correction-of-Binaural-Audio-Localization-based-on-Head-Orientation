import numpy as np
import math as mt
import matplotlib.pyplot as plt
from pysofaconventions import *



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
numfiles = len(audios)
numchannels = len(audios[0])
numsamples = len(audios[0][0])

""" #Plot of the 1st audio file by splitting left and right
plt.plot(audios[0][0], label="left", linewidth=0.5,  marker='o', markersize=1)
plt.plot(audios[0][1], label="right", linewidth=0.5,  marker='o', markersize=1)
plt.grid()
plt.legend()
plt.show() # It's pretty clear, based on the ITD and ILD, that the source is located at the left, which on the other hand confirms the sourcePositions[0] information
"""
""" #RENDER WITH A FILE AND SAVING IT
data, samplerate = sf.read('C:/Users/pablo/Desktop/365061__ryntjie__pouring-cat-food-into-a-plastic-bowl.wav') #Open a mono wav file. I got this one from freesound https://freesound.org/people/Ryntjie/sounds/365061/ 
binaural_left = scipy.signal.fftconvolve(data,hrtf[0]) #Convolve it with the hrtf
binaural_right = scipy.signal.fftconvolve(data,hrtf[1])
binaural = np.asarray([binaural_left, binaural_right]).swapaxes(-1,0)
sf.write('C:/Users/pablo/Desktop/resultadobinaural.wav', binaural, samplerate) # Write to a file
"""



#ARRAY CREATION
#Variable initialization
"""
time = np.zeros((numfiles, 256))
peaks = np.zeros((numfiles, 2))
tpeaks = np.zeros((numfiles, 2))
"""
codebook_f = np.zeros((numfiles, numsamples))
codebook = np.zeros((numfiles, numchannels, numsamples),dtype=complex)
for x in range(numfiles): 
    #DFT sample frequencies, ordered by audiofile
    codebook_f[x] = np.fft.fftfreq(audios[x,0].shape[-1]) #L and R share the same values
    #Codebook of HRTFs, ordered by audiofile
    codebook[x,0] = np.fft.fft(audios[x,0])
    codebook[x,1] = np.fft.fft(audios[x,1])

"""
#3D array for audios + time array
time[x] = np.arange(0,int(len(audios[x,0]))) / sfreq #Time array, ordered by audiofile
#Previous ILD & ITD work
peaks[x] = np.amax(audios[x], axis=1) #Peak array, ordered by audiofile
extractionL, = np.where(audios[x,0]==peaks[x,0])
extractionR, = np.where(audios[x,1]==peaks[x,1])
tpeaks[x] = [max(extractionL),max(extractionR)] #Time position array, ordered by audiofile
x+=1
"""



#RULE OUT ELEVATION
#Taking only samples with zero elevation
ind_azim = []
azimuth = []
for x in range(numfiles):
    if (sP[:][x][1] == 0) == True:
        ind_azim.append(x)
ILD = np.zeros((int(numsamples/2), 360)) #Only of the first half of samples as it's symmetrical
IPD = np.zeros((int(numsamples/2), 360)) #Only of the first half of samples as it's symmetrical
x=0
for a in ind_azim:
    sPaz = float(sP[a][0])
    if sPaz > 180: #Quadrant correction
        sPaz = int(-(sPaz-180))
    azimuth.append(sPaz) #Create array with the ideal angles for the plot

    for b in range(int(numsamples/2)):
        ILD[b,int(azimuth[x])] = -10 * mt.log10(np.abs(codebook[a,1,b])/np.abs(codebook[a,0,b])) #ILD (following eq 6 from the paper) (in dB)
        IPD[b,int(azimuth[x])] = np.angle((codebook[a,1,b]/codebook[a,0,b]), deg=False) #IPD in radians (following eq 7 from the paper)
        #print('x' + str(x) + 'a' + str(a) + 'b' + str(b))
        #ILD.append(a1[a])
        #IPD.append(a2[a])
    x+=1



#PLOTS
fig, (ax1, ax2) = plt.subplots(2,1)
axx1 = ax1.pcolormesh(ILD)
axx2 = ax2.pcolormesh(IPD, cmap=plt.cm.get_cmap('hsv'))
ax1.set(ylabel='Frequency (kHz)', title='KU100')
ax2.set(xlabel='\u03C6 (º)', ylabel='Frequency (kHz)')

axxx1 = fig.colorbar(axx1, ax=ax1, aspect=6)
axxx2 = fig.colorbar(axx2, ax=ax2, aspect=6) #, ticks=[-np.pi/2, 0, -np.pi/2])
axxx1.set_label('ILD (dB)')
axxx2.set_label('IPD (rad)')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True) #To simplify yaxis format
#plt.setp((ax1, ax2), xticks=[-90, -45, 0, 45, 90]) #To control the lower xaxis bins
plt.subplots_adjust(hspace = 0.25) #Adjust height between subplots
plt.show()
print("End")