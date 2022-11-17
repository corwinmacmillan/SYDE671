import numpy as np
import matplotlib.pyplot as plt

img = np.random.randint(0, 255, size=(256, 256))
fig1 = plt.figure()
plt.subplot(121)
plt.imshow(img, cmap='gray')

mean = np.mean(img)

poisson = np.random.poisson(mean, size=(256,256))
plt.subplot(122)
plt.imshow(poisson, cmap='gray')
plt.show()