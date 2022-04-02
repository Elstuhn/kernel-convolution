import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
RD1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
RD2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
boxBlur_kernel = np.multiply(1/9, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
gaussianBlur3 = np.multiply(1/16, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]))
"""
def applyKernel(image : np.ndarray, kernel : np.ndarray, stride : tuple = (1, 1)) -> np.ndarray:
    heightK = kernel.shape[0]
    widthK = kernel.shape[1]
    height = image.shape[0]
    width = image.shape[1]
    
    if any([
        (width - widthK) % stride[1],
        (height - heightK) % stride[0]
    ]):
        raise Exception("Kernel shape/stride not suitable for image!")
    
    fMap = []
    for row in range(height):
        if (row > height - heightK):
            break
        for column in range(width):
            if (column > width - widthK):
                break
            endCol = column + widthK
            endHeight = row + heightK
            joint = np.vstack([image[row + i][column : endCol] for i in range(heightK)])
            convolved = np.matmul(kernel, joint)
            if not column:
                rowConv = convolved
                continue
            rowConv = np.hstack((rowConv, convolved))
        fMap.append(rowConv)
    fMap = np.vstack(fMap)
    return fMap
"""
def applyKernel(image : np.ndarray, kernel : np.ndarray, stride : tuple = (1, 1)):

    heightK = kernel.shape[0]
    widthK = kernel.shape[1]
    height = image.shape[0]
    width = image.shape[1]
    if any([
        (width - widthK) % stride[1],
        (height - heightK) % stride[0]
    ]):
        raise Exception("Kernel shape/stride not suitable for image!")
    fMap = []
    for row in range(height):
        if (row > height - heightK):
            break
        for column in range(width):
            if (column > width - widthK):
                break
            endCol = column + widthK
            joint = np.vstack([image[row + i][column : endCol] for i in range(heightK)])
            convolved = np.matmul(kernel, joint)
            pixel = np.sum(convolved)
            if not column:
                rowConv = np.array([pixel])
                continue
            rowConv = np.hstack((rowConv, pixel))
        fMap.append(rowConv)
    fMap = np.vstack(fMap)
    return fMap

def ConvLayer(image : np.ndarray, kernel : np.ndarray, stride : tuple = (1, 1)):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    fMap = np.dstack([applyKernel(channel, kernel, stride) for channel in [r, g, b]])
    return fMap

cat = Image.open("path")
cat = np.array(cat)
#image = np.random.rand(30, 30, 3)
#plt.plot(image)
#plt.imshow(cat)
#plt.show()
#fMap = ConvLayer(cat, identity_kernel)
#print(fMap.shape)
kernels = [identity_kernel, RD1, RD2, sharpen_kernel, boxBlur_kernel, gaussianBlur3]
kernelNames = ["Identity Kernel", "Ridge Detection 1", "Ridge Detection 2", "Sharpen Kernel", "Box Blur", "Gaussian Blur"]
fig = plt.figure(figsize = (35, 35))
fig.add_subplot(4, 2, 1)
plt.imshow(cat)
plt.title("Original")
for i in range(2, 8):
    fMap = ConvLayer(cat, kernels[i-2])
    print(fMap.shape)
    fig.add_subplot(4, 2, i)
    plt.imshow(fMap)
    plt.title(kernelNames[i-2])

fig.tight_layout(pad = 10.0)
plt.show()
