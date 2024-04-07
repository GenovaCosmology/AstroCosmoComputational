import numpy as np

# REJECTION_SAMPLING

'''
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
'''

def Rejection_Sampling(n_samples,target_distribution,proposal_distribution,dim,X_acc,Y_acc,X_rej,Y_rej):
    samples=[]
    accepted=0
    total_proposed=0
    while accepted < n_samples:
        x=np.random.uniform(-10,10,dim)
        total_proposed += 1
        u=np.random.uniform(0,proposal_distribution(x))
        if np.all(u < target_distribution(x)):
            samples.append(x)
            accepted += 1
            X_acc.append(x)
            Y_acc.append(u)
        else:
            X_rej.append(x)
            Y_rej.append(u)
    if total_proposed!=0:
        efficiency = accepted / total_proposed
    else:
        efficiency=0
    return samples,efficiency

# METROPOLIS_HASTINGS

def Metropolis_Hastings(target_distribution,proposal_distribution,acceptance_probability,x_0,num_samples):
    #x_0: initial sample
    #x_t: current sample
    #x_star: candidate sample
    
    samples=[x_0]
    x_t=x_0
    n_acc=0 #to count the number of accepted samples to then compute the efficiency

    for _ in range(num_samples):
        #generation of a candidate sample (x^* -> x_star) from proposal distribution
        if isinstance(x_0,(float,int)):
            x_star=x_t+np.random.uniform(-1,1)
        else:
            x_star=x_t+np.random.multivariate_normal(np.zeros_like(x_0),np.eye(len(x_0)))
        #acceptance probability (alpha)
        alpha=acceptance_probability(x_star,x_t)
        #accept or reject the candidate sample
        if np.random.rand()<alpha:
            x_t=x_star
            n_acc+=1
        samples.append(x_t)

    efficiency=n_acc/num_samples

    return samples,efficiency