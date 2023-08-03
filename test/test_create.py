import create
import matplotlib.pyplot as plt

shape = create.Image(1000, 2000)

shape.square(1, [200, 700], [900, 1300])
shape.square(3, [500, 700], [100, 300])
shape.circle(1, (1500, 500), 300)

plt.imshow(shape.img)
plt.show()

blur = create.Blur(shape.img)

img_b, ker = blur.gaussian(0, 100, return_kernel=True)

plt.plot(ker[1000])
plt.plot(ker[:, 500])
plt.show()

plt.imshow(ker)
plt.show()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(shape.img)
ax[1].imshow(img_b)
plt.show()
