import matplotlib.pyplot as plt
import numpy as np

x_train = np.loadtxt('../dataset_for_training/Gaussian_2d_mixture.dat')

x = np.loadtxt('../epoch_6000_sample.dat')
y = np.loadtxt('../epoch_6000_prior.dat')


m,n = x.shape

n1 = 0
n2 = 1

yL = np.zeros((m,))
for i in range(m):
    yL[i] = np.sqrt(y[i,n1]**2+y[i,n2]**2) 

permutation = np.argsort(yL)

xS = x[permutation]
yS = y[permutation]

i1 = 5000
i2 = 9000
stride = 1
ss= 1 

fig,axs=plt.subplots(1,3, figsize=(12.3,4), sharey=True)

axs[0].scatter(x_train[:m:ss,n1],x_train[:m:ss,n2], s=8, color='blue',alpha=0.1)
axs[0].axis([-8, 8, -8, 8])
#axs[0].set(xlabel='$y_1$',ylabel='$y_2$')
axs[0].grid(True)
axs[0].set_title('Samples from true PDF')
axs[0].set_rasterized(True)

color1='blue'
color2='blue'#'red'
color3='blue'#'green'

axs[1].scatter(xS[:i1:stride,n1], xS[:i1:stride,n2], s=8, color=color1, alpha=0.1)
axs[1].scatter(xS[i1:i2:stride,n1], xS[i1:i2:stride,n2], s=8, color=color2,alpha=0.1)
axs[1].scatter(xS[i2::stride,n1], xS[i2::stride,n2], s=8, color=color3,alpha=0.1)
axs[1].axis([-8, 8, -8, 8])
#axs[1].set(xlabel='$y_1$')
axs[1].grid(True)
axs[1].set_title('Samples from model: $Y=f^{-1}(Z)$')
axs[1].set_rasterized(True)

axs[2].scatter(yS[:i1:stride,n1], yS[:i1:stride,n2], s=8, color=color1,alpha=0.1)
axs[2].scatter(yS[i1:i2:stride,n1], yS[i1:i2:stride,n2], s=8, color=color2,alpha=0.1)
axs[2].scatter(yS[i2::stride,n1], yS[i2::stride,n2], s=8, color=color3,alpha=0.1)
axs[2].axis([-8, 8, -8, 8])
#axs[2].set(xlabel='$y_1$')
axs[2].grid(True)
axs[2].set_title('$Z$: normal rv')
axs[2].set_rasterized(True)


#fig.savefig('Figure.eps', rasterized=True, dpi=300)

plt.show()

