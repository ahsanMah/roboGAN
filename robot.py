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
def drawRobotArm(data, showLinks=True):
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=2)
        
    print(data.shape)
    nrLinks = round(data.shape[1]/2)
#     colors = [['blue', 'orange', 'green', 'darkviolet'], ['dodgerblue', 'goldenrod', 'chartreuse', 'deeppink']]
    colors = [['blue', 'orange', 'green', 'darkviolet'], ['chartreuse', 'deeppink', 'dodgerblue', 'goldenrod']]
    
    print(nrLinks)
    for k in range(data.shape[2]):
        for j in range(data.shape[0]):
            for i in range(nrLinks):
                if i==0:
                    x0=0
                    y0=0

                else:
                    x0=data[j,2*(i-1),k]
                    y0=data[j,2*(i-1)+1,k]
                x1=data[j,2*(i),k]
                y1=data[j,2*(i)+1,k]
                if(showLinks):
                    if(j==0):
                        if(i==0):
                            label_ = 'Original'
                        else:
                            label_ = '_'
                    else:
                        label_ = '_nolegend_'
                    plt.plot([x0,x1], [y0,y1], color = colors[k][i], label=label_)
                plt.plot(x1,y1,'o',markersize=2, color = colors[k][i],label='_nolegend_')
    
    lst = [" "] * 2*nrLinks
    lst[0] = "Original"
    lst[nrLinks] = "Generated"
    print(lst)
    plt.legend(lst)
    plt.plot(0,0,'o',color='black',linewidth=5, label='_nolegend_')
    plt.xlim([-5,5])
    plt.ylim([-5,5]) 
    
def plotRobotDistribution(data, colorId=0):
    colors = [['blue', 'orange', 'green'], ['violet', 'seagreen', 'pink']]

    nrLinks = round(data.shape[1]/2)
    for i in range(nrLinks-1,-1,-1):
        print(i)
        plt.scatter(data[:,(i)*2], data[:,(i)*2+1], alpha= 0.5, color = colors[colorId][i])
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    
    
def getAvgAngle(s,c):
    a1,a2 = getOriginalAngles(s,c)
    return (a1+a2)/2


def getOriginalAngles(s,c):
    if c >= 0:
        ang1 = np.arcsin(s)
        if s >= 0:
            ang2 = np.arccos(c)
        else:
            ang2 = -np.arccos(c)
    else:
        if s >= 0:
            ang1 = np.pi - np.arcsin(s)
            ang2 = np.arccos(c)
        else:
            ang1 = -np.pi - np.arcsin(s)
            ang2 = -np.arccos(c)
   
    return ang1,ang2
    
       
def endEffectorDist(set1, set2, nrLinks1, nrLinks2):
    
    endeff1 = set1[:,nrLinks1*4-2:nrLinks1*4]
    endeff2 = set2[:,nrLinks2*4-2:nrLinks2*4]
    norms = np.linalg.norm(endeff1-endeff2, axis=1)
    return np.mean(norms)

def positionsFromAngles(data, nrLinks, lengths):
    angles=np.zeros([data.shape[0], nrLinks])
   
    for i in range(data.shape[0]):
        for j in range(nrLinks):
            #print(np.arcsin(data[i,2*j]))
            #print(np.arccos(data[i,2*j+1]))
            #angles[i,j] = (np.arcsin(data[i,2*j]) + np.arccos(data[i,2*j+1]))/2
            angles[i,j] = getAvgAngle(data[i,2*j], data[i,2*j+1])
    
    return computeY(angles, lengths)
    #print(Y.shape)
    #return np.concatenate([data[:,:2*nrLinks], Y], axis = 1)
    
def compareInternalPositions(data, nrLinks, lengths): #pos from angles vs real pos
    anglePos = positionsFromAngles(data, nrLinks, lengths)[:,:nrLinks*2] #only positions, not distances
    pos = data[:,nrLinks * 2 : nrLinks * 4]
    norms = np.linalg.norm(anglePos - pos, axis=1)
    return np.mean(norms)

def replaceAnglePos(data, nrLinks, lengths):
    Y=positionsFromAngles(data, nrLinks, lengths)
    data[:,2*nrLinks:] = Y
    return data
    






#x90,y90 = computeRobotData(0, np.pi/2,3,[2,1,0.5], 1000)
