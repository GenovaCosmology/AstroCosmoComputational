import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import ArtistAnimation



G=6.674e-11  # m^3/kg*s^2
M1=3e28  # kg
M2=2e28  # kg

#number of years
N=11

# Posizione iniziale
Pos1 = np.array([-10e10,5e10,0])
Pos2 = np.array([5e10,0,-5e10])


# initial velocity
V1=np.array([2000,0,100])    # m/s
V2=np.array([0,-2500,100])   # m/s

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
#total time for simulation in seconds
T=N*365.25*dt  #seconds in N years
time=np.arange(0,T,dt)




# lists to store position, velocity, and acceleration for plotting
pos1=[]
pos2=[]

vel1=[]
vel2=[]

acc1=[]
acc2=[]

posx1=[]
posy1=[]
posz1=[]

posx2=[]
posy2=[]
posz2=[]

posx_cm=[]
posy_cm=[]
posz_cm=[]

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
    posz1.append(Pos1[2])

    posx2.append(Pos2[0])
    posy2.append(Pos2[1])
    posz2.append(Pos2[2])

    center_of_mass = (M1 * Pos1 + M2 * Pos2) / (M1 + M2)

    posx_cm.append(center_of_mass[0])
    posy_cm.append(center_of_mass[1])
    posz_cm.append(center_of_mass[2])


# convert lists to arrays
pos1=np.array(pos1)
vel1=np.array(vel1)
acc1=np.array(acc1)

pos2=np.array(pos2)
vel2=np.array(vel2)
acc2=np.array(acc2)


# Plot 3D trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(posx1, posy1, posz1, c='red', label='Star 1')
ax.plot(posx2, posy2, posz2, c='orange', label='Star 2')
ax.plot(posx_cm, posy_cm, posz_cm, c='black', label='Center of mass')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory')
ax.legend()

plt.show()


# Scatter 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

### Serve solo per la Legend:
ax.scatter(posx1[0], posy1[0], posz1[0], c='red',s=3, label='Star 1')
ax.scatter(posx2[0], posy2[0], posz2[0], c='orange',s=3, label='Star2')
ax.scatter(posx_cm[0], posy_cm[0], posz_cm[0], c='black',s=3, label='Center of mass')
ax.legend()
###

for i in range (len(posx1)):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot (Frame {})'.format(i))

    if (i % 30 == 0):
        ax.scatter(posx1[i], posy1[i], posz1[i], c='red',s=5, label='Star 1')
        ax.scatter(posx2[i], posy2[i], posz2[i], c='orange',s=5, label='Star2')
        ax.scatter(posx_cm[i], posy_cm[i], posz_cm[i], c='black',s=2, label='Center of mass')
        plt.pause(0.01)  # Pause for 0.1 seconds

    plt.draw()  # Draw the updated plot
plt.show()


# plotting projection xy
plt.figure(figsize=(8,8))
plt.title('proiezione sul piano XY')
for i in range (0,len(posx1)):
    plt.xlabel("X")
    plt.ylabel("Y")
    if i % 25 == 0:
        plt.scatter(posx1[i],posy1[i],s=5,c='red')    # s=size punti
        plt.scatter(posx2[i],posy2[i],s=5,c='orange')    # s=size punti
        plt.pause(0.01)   # per avere i punti in "tempo reale"
plt.show()


# plotting projection xz
plt.figure(figsize=(8,8))
plt.title('proiezione sul piano ZX')
for i in range (0,len(posx1)):
    plt.xlabel("Z")
    plt.ylabel("X")
    if i % 25 == 0:
        plt.scatter(posz1[i],posx1[i],s=5,c='red')    # s=size punti
        plt.scatter(posz2[i],posx2[i],s=5,c='orange')    # s=size punti
        plt.pause(0.01)   # per avere i punti in "tempo reale"
plt.show()

