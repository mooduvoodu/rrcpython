import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, Polygon   #C:\Users\KyleClubb\AppData\Local\Programs\Python\Python38-32\Lib\site-packages



# Data for plotting matplotlib
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()

#numpy
a = np.arange(6)                         # 1d array
print(a)
b = np.arange(12).reshape(4,3)           # 2d array
print(b)

#shapely
point1 = Point(2.2, 4.2)
point2 = Point(7.2, -25.1)
point3 = Point(9.26, -2.456)
point3D = Point(9.26, -2.456, 0.57)
print(type(point1))

