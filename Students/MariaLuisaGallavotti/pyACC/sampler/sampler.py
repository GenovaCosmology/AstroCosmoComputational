import numpy as np

#n_samples: number of points that approximate the target distribution
#x_min, x_max: they define the range of the variable x
#X_acc, Y_acc: arrays with accepted values
#X_rej, Y_rej: arrays with rejected values
def Rejection_Sampling(n_samples,target_distribution,proposal_distribution,x_min,x_max,X_acc,Y_acc,X_rej,Y_rej):
    samples=[]
    accepted=0
    while accepted < n_samples:
        x=np.random.uniform(x_min,x_max)
        u=np.random.uniform(0,proposal_distribution(x))
        if u<target_distribution(x):
            samples.append(x)
            accepted+=1
            X_acc.append(x)
            Y_acc.append(u)
        else:
            X_rej.append(x)
            Y_rej.append(u)
    return samples