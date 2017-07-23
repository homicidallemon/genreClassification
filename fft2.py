import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os

class FourierTransform:
    def __init__(self,num):
        self.ultimateVector = []
        self.target = []
        targetDict = {}
        classCount = 0
        directory = "/media/homicidallemon/LUCAS REIS/ASS/"
        previousRoot = None
        #for i in xrange(1,12):

        """if i < 10:
            name = "/media/homicidallemon/LUCAS REIS/ASS/Recess [FLAC]/Skrillex/track0"+str(i)+".wav"
        else:
            name = "/media/homicidallemon/LUCAS REIS/ASS/Recess [FLAC]/Skrillex/track"+str(i)+".wav"""

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".wav"):
                    name = root+'/'+file
                    if root not in targetDict:
                        targetDict[root] = classCount
                        classCount += 1
                    print name
                else:
                    continue

                self.target.append(targetDict[root])

                fs, data = wavfile.read(name)
                s1 = data[:,0]
                n = len(s1)
                #s1 = s1[int(n/2-0.2*n):int(n/2+0.2*n)]
                s1 = s1[int(n/2-2000):int(n/2+2000)]
                s1 = [(ele/2**16.)*2-1 for ele in s1]

                out = fft(s1)
                #print np.asarray(out)
                #out = out/float(n)
                n = len(out)/2
                k = np.arange(len(out))
                T = float(len(out))/fs
                frqlabel = k/T
                vec = abs(out[1:n-1])
                #plt.plot(frqlabel[1:n-1],vec)
                #f = open(name + ".txt",'w')

                vec[:] = [(i-min(vec))/(max(vec)-min(vec)) for i in vec]

                plt.plot(frqlabel[1:n-1],vec)
                #print vec
                vec2 = []
                for i in xrange(num):
                    vec2.append(sum(vec[i*n/num:(i+1)*n/num]))
                self.ultimateVector.append(vec2)
                #f.write(str(vec2))
                #f.write("\n")
                #f.close()
                #plt.show()
                """if i < 10:
                    plt.savefig("Skrillex-track0"+str(i))
                else:
                plt.savefig(name+".png")"""
                plt.close()
