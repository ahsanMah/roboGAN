import numpy as np
import matplotlib.pyplot as plt


def R(alpha, l ):
    return np.array([[np.cos(alpha),-np.sin(alpha), np.cos(alpha)*l],[np.sin(alpha), np.cos(alpha), np.sin(alpha)*l],[0, 0, 1]])

def T(x,y):
    return np.array([[1, 0, x],[0, 1, y],[0,0,1]])

def length(x,y):
    return np.linalg.norm([x,y])

def computeX(alphas):
    nrSamples = alphas.shape[0]
    nrLinks = alphas.shape[1]
    x = np.zeros([nrSamples, nrLinks * 2])

    for i in range(nrLinks):
        #print(x.shape)

        x[:,2*i]=np.sin(alphas[:,i])
        x[:,(2*i)+1]=np.cos(alphas[:,i])
    return x

#y = pos for each joint, including endeffector
def computeY(alphas, lengths):
    nrSamples = alphas.shape[0]
    nrLinks = alphas.shape[1]
    y = np.zeros((nrSamples,nrLinks * 3))
    for i in range(nrSamples): #nrSamples
        pose = np.identity(3)#forwardR(alphas[i,nrLinks - 1],l[nrLinks - 1])
        for j in range(nrLinks):
            pose = np.matmul(pose, R(alphas[i,j],lengths[j]))
            #print(pose)
            #print(pose)
            y[i,2*j:2*(j+1)]=pose[0:2,2] #position
            #if j>0:
            y[i,(nrLinks*2) + j] = length(pose[0,2],pose[1,2]) #dist from origin
    return y

        #pose1 =
        #pose2 = forwardRR(alpha1[i,0],l1[i,0],alpha2[i,0],l2[i,0])
        #pose3 = forwardRRR(alpha1[i,0],l1[i,0],alpha2[i,0],l2[i,0],alpha3[i,0],l3[i,0])
        #pose = np.matmul(R(alpha1[i,0],l1[i,0]),R(alpha2[i,0],l2[i,0]))
        #print(pose)
        #y[i,0:2] = pose1[0:2,2] #joint1
       # y[i,2:4] = pose2[0:2,2] #joint2
        #y[i,4:6] = pose3[0:2,2] #joint2
        #y[i,6] = length(pose2[0,2],pose2[1,2])
        #y[i,7] = length(pose3[0,2],pose3[1,2])
    return y

def sampleAngles(minAngle, maxAngle, nrSamples):
    alpha = np.expand_dims(np.asarray(np.random.uniform(minAngle, maxAngle, nrSamples)),axis=0).T
    #print(alpha.shape)
    return alpha

def computeRobotData(angles, nrLinks, lengths, nrSamples):
    alphas = np.zeros([nrSamples, nrLinks])

    for i in range(nrLinks):
        alphas[:,i:i+1] = sampleAngles(angles[0], angles[1], nrSamples)
    print(alphas.shape)
    #alpha1 = sampleAngles(minAngle, maxAngle)
    #l1 = np.expand_dims(np.asarray(np.random.uniform(0,10, nrSamples)),axis=0).T
    #alpha2 = sampleAngles(minAngle, maxAngle)
    #alpha3 = sampleAngles(minAngle, maxAngle)
#l2 = np.expand_dims(np.asarray(np.random.uniform(0,10, nrSamples)),axis=0).T
    x = computeX(alphas)
    y = computeY(alphas, lengths)
    return x,y

#call with positions (first 2*nrLinks elements of y)
def drawRobotArm(data):
    nrLinks = round(len(data)/2)
    print(nrLinks)
    
    for i in range(nrLinks):
        if i==0:
            x0=0
            y0=0
           
        else:
            x0=data[2*(i-1)]
            y0=data[2*(i-1)+1]
        x1=data[2*(i)]
        y1=data[2*(i)+1]
        plt.plot([x0,x1], [y0,y1])
    plt.plot(x1,y1,'ro')
    plt.xlim([-4,4])
    plt.ylim([-4,4])




#x90,y90 = computeRobotData(0, np.pi/2,3,[2,1,0.5], 1000)
