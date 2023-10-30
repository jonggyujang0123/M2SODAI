import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from data_tools.Image_HSI_to_JPG import HSItensor2imgs, HSItensor2imgs_chan
from tqdm import tqdm

parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('dir',
                   help='directory for mean and var calculation')
args = parser.parse_args()

data = sio.loadmat(args.dir)['data']

# from mpl_toolkits.mplot3d import Axes3D

JPG = HSItensor2imgs(data)

data = data-np.min(data,(0,1))
data = data/np.max(data,(0,1))

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.linspace(0, 1, 224)
X, Y = np.meshgrid(x, x)
Z = np.zeros_like(X)

levels = np.linspace(-1, 1, 40)

num_channel = 127
for i in tqdm(range(num_channel)):
    if i == num_channel-1:
        ax.plot_surface(X, Y, 0.1*i+Z, rstride=2, cstride=2,
                facecolors=JPG/255)
    else:
        ax.plot_surface(X, Y, 0.1*i+Z, rstride=4, cstride=4,
                facecolors=cm.coolwarm(data[:,:,i]))
        # ax.contourf(X, Y, 0.1*i+data[:,:,i], zdir='z', levels=0.1*i + .1*levels)
# ax.contourf(X, Y, 3+data[:,:,20], zdir='z', levels=3+.1*levels)
# ax.contourf(X, Y, 7+data[:,:,90], zdir='z', levels=7+.1*levels)

# ax.legend()
ax.set_xlim3d(0, 1)
ax.set_ylim3d(0, 1)
ax.set_zlim3d(0, 13)

plt.show()