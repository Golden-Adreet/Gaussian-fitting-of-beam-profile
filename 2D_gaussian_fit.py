import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#loading and prepering the image

img = cv2.imread("D:/Code files pycharm/Masters/Green_laser_pointer_TEM00_profile.JPG", cv2.IMREAD_GRAYSCALE)
print(img)
img = img.astype('float')
img-= np.min(img)
img/=np.max(img)

# Showing the image 
plt.imshow(img, cmap='inferno')
plt.colorbar()
plt.title("Beam Profile")
plt.show()


y = np.arange(img.shape[0])
x = np.arange(img.shape[1])

x,y = np.meshgrid(x,y)

#define the gaussian function:
def gauss_2d(cords, A, x0,y0,sx,sy,c):
    x, y = cords
    return (A * np.exp(-((x - x0)**2 / (2 * sx**2)+ (y - y0)**2 / (2 * sy**2))) + c).ravel()

A0 = np.max(img)
x0 = img.shape[0]/2
y0 = img.shape[1]/2
sx0 = img.shape[0]/4
sy0 = img.shape[1]/4
c0 = np.min(img)
p0 = [A0, x0, y0, sx0, sy0, c0]

val,error = curve_fit(gauss_2d, (x,y),img.ravel(),p0 = p0)

A,x0,y0,sx,sy,c = val

fitted_img = gauss_2d((x,y), A, x0,y0,sx,sy,c).reshape(img.shape)

plt.imshow(fitted_img, cmap='inferno')
plt.colorbar()
plt.title("fitted Beam Profile")
plt.show()
print(f'A = {A}\nx_0 = {x0}\ny_0 = {y0}\nsx = {sx}\nsy = {sy}')
plt.imshow(abs(fitted_img-img), cmap='inferno')
plt.colorbar()
plt.title('resedue')
plt.show()