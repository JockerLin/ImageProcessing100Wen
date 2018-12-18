import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("imori_dark.jpg").astype(np.float)
H, W, C = img.shape

# Trans [0, 255]
m0 = 128
s0 = 52

m = np.mean(img)
s = np.std(img)

print(s)

out = img.copy()
out = s0 / s * (out - m) + m0
out = out.astype(np.uint8)

# Display histogram
plt.hist(out.ravel(), rwidth=0.8, range=(0, 255))
plt.show()

# Save result
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("out.jpg", out)