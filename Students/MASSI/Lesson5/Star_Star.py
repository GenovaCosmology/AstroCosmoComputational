import numpy as np
import matplotlib.pyplot as plt

G=6.674e-11  # m^3/kg*s^2
M1=2e28  # kg
M2=2e28  # kg

# Massa ridotta
mu = (M1*M2)/(M1 + M2)



# Posizione centro di massa
# R_CM = (M1 * Pos1 + M2 * Pos2) / (M1 + M2)  # = 0 Considerando le coordinate x, y


#acceleration function
def force(pos1, pos2):
    r=np.linalg.norm(pos1-pos2)
    return (G*M1*M2/r**3)*(pos2-pos1)


# Eulero integration method
def eulero(pos1, vel1, pos2, vel2, dt):
    new_vel1=vel1+(force(pos1,pos2)/M1)*dt
    new_pos1=pos1+new_vel1*dt
    new_vel2=vel2+(-force(pos1,pos2)/M2)*dt
    new_pos2=pos2+new_vel2*dt
    return new_pos1,new_vel1, new_pos2,new_vel2



#time step
dt=86400  #seconds in one day
#number of years
N=10
#notal time for simulation in seconds
T=N*365.25*dt  #seconds in N years
time=np.arange(0,T,dt)


Pos1 = np.array([-1e11,0])
Pos2 = np.array([1e11,5e10])

# initial velocity
V1=np.array([0,1000])   # m/s
V2=np.array([0,-1000])   # m/s

# lists to store position, velocity, and acceleration for plotting
pos1=[]
pos2=[]
vel1=[]
vel2=[]
acc1=[]
acc2=[]
posx1=[]
posy1=[]
posx2=[]
posy2=[]

#iteration to update Earth's position
for t in time:
    pos1.append(Pos1)
    vel1.append(V1)
    pos2.append(Pos2)
    vel2.append(V2)
    acc1.append(force(Pos1, Pos2)/M1)
    acc2.append(force(Pos1, Pos2)/M2)
    Pos1,V1,Pos2,V2=eulero(Pos1,V1,Pos2,V2,dt)

    posx1.append(Pos1[0])
    posy1.append(Pos1[1])
    posx2.append(Pos2[0])
    posy2.append(Pos2[1])

# convert lists to arrays
pos1=np.array(pos1)
vel1=np.array(vel1)
acc1=np.array(acc1)

pos2=np.array(pos2)
vel2=np.array(vel2)
acc2=np.array(acc2)


# plotting position
plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(time,np.linalg.norm(pos2,axis=1))
#plt.plot(time,np.linalg.norm(pos2,axis=1))
plt.xlabel('Time (seconds)')
plt.ylabel('Position of Star2 (meters)')
plt.title('Position of Star2 over {} Years'.format(N))
plt.grid(True)

# plotting velocity
plt.subplot(2,1,2)
plt.plot(time,np.linalg.norm(vel2,axis=1),color='orange')
plt.plot(time,np.linalg.norm(vel2,axis=1),color='orange')
plt.xlabel('Time (seconds)')
plt.ylabel('Velocity of Star2 (m/s)')
plt.title('Velocity of Star2 over {} Years'.format(N))
plt.grid(True)

plt.tight_layout()
plt.show()


#plotting trajectory
plt.figure(figsize=(8,8))


# PER VEDERE LA POSIZIONE IN TEMPO "REALE":
for i in range (0,len(posx1)):
    if i % 20 == 0:
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(posx1[i],posy1[i],s=5,c='gold')    # s=size punti
        plt.scatter(posx2[i],posy2[i],s=5,c='orange')    # s=size punti
        plt.pause(0.05)   # per avere i punti in "tempo reale"
plt.show()


plt.plot(posx1, posy1, color="blue")
plt.plot(posx2, posy2, color="red")
plt.plot(-1e11,0 ,marker="o", markersize=4, color="yellow")
plt.plot(1e11,0 ,marker="o", markersize=4, color="orange")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()