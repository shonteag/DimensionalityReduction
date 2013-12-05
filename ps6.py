import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

def visualize(scores, faces):
  """
  The function for visualization part, 
  which put the image at the coordinates given by their coefficients of 
  the first two principal components (with translation and scaling).

  scores: n x 2 array, where each row contains the first 2 principal component scores of each face
  faces: n x 4096 array
  """
  pc_min, pc_max = np.min(scores, 0), np.max(scores, 0)
  pc_scaled = (scores - pc_min) / (pc_max - pc_min)  
  fig, ax = plt.subplots()
  for i in range(len(faces)):
    imagebox = offsetbox.OffsetImage(faces[i, :].reshape(64,64).T, cmap=plt.cm.gray, zoom=0.5)
    box = offsetbox.AnnotationBbox(imagebox, pc_scaled[i, 0:2])
    ax.add_artist(box)
  plt.show()

# Example code starts from here
# Load the data set
faces = sp.genfromtxt('faces.csv', delimiter=',')

# Example for displaying the first face, which may help you how the data set presents
# plt.imshow(faces[0, :].reshape(64, 64).T, cmap=plt.cm.gray)
# plt.show()


# Your code starts from here ....
import time
from copy import deepcopy

# a. Randomly display a face
# STUDENT CODE TODO
import random
t0 = time.time()
ran = random.randint(0,len(faces))
print "(a) Random Face Index: " + str(ran)
plt.imshow(faces[ran, :].reshape(64,64).T, cmap=plt.cm.gray)
#plt.show()


# b. Compute and display the mean face
# STUDENT CODE TODO
t0 = time.time()

mean = []
facesTranspose = deepcopy(faces)

for feature in facesTranspose.T:
  mean.append(np.mean(feature))

plt.imshow(np.matrix(mean).reshape(64,64).T, cmap=plt.cm.gray)
#plt.show()

print "(b) Mean computation complete. " + str(time.time() - t0) + " seconds"


# c. Centralize the faces by substracting the mean
# STUDENT CODE TODO
t0 = time.time()
facesCentered = deepcopy(faces)

for index,face in enumerate(facesCentered):
  facesCentered[index] = [faceVal - meanVal for faceVal,meanVal in zip(face,mean)]

print "(c) Centralize complete. " + str(time.time() - t0) + " seconds"

# d. Perform SVD (you may find scipy.linalg.svd useful)
# STUDENT CODE TODO

U, s, Vh = la.svd(facesCentered)
W = U * s

# e. Show the first 10 priciple components
# STUDENT CODE TODO
for i in range(0,10):
  print "Entry " + str(i) + " in V."
  plt.imshow(Vh[i,:].reshape(64,64).T, cmap=plt.cm.gray)
  #plt.show()

print "(e) Complete. Yep. Those were some creepy images."


# f. Visualize the data by using first 2 principal components using the function "visualize"
# STUDENT CODE TODO
scores = [[]] * 30
visFaces = np.matrix([[0] * 4096] * 30)

for i in range(30):
  ran = random.randint(0,len(faces)-1)
  visFaces[i, :] = facesCentered[ran, :]
  score = np.dot(facesCentered[ran],Vh[0:2,:].T)
  # score0 = np.dot(facesCentered[ran],Vh[0,:])
  # score1 = np.dot(facesCentered[ran],Vh[1,:])

  scores[i] = score

visualize(scores,visFaces)
print "(f) Visualization complete."


# g. Plot the proportion of variance explained
# STUDENT CODE TODO

totalVariance = 0.0
componentVariance = [0.0] * 10

sAsArray = np.asarray(s)

totalVariance = sum(val for val in sAsArray)

for index in range(0,10):
  componentVariance[index] = sAsArray[index] / totalVariance

print componentVariance

plt.figure(figsize=(8,6), dpi=80)
plt.xlim(0,9)
plt.xticks(np.linspace(0,9,10,endpoint=True))
plt.plot(componentVariance, color="green")
plt.ylabel("Explained Variance")
plt.xlabel("Component Index")

plt.show()

# h. Face reconstruction using 5, 10, 25, 50, 100, 200, 300, 399 principal components
# STUDENT CODE TODO
numComponentsArray = [5,10,25,50,100,200,300,399]

ran = random.randint(0,len(faces)-1)

for numComponents in numComponentsArray:
  reconstruct = deepcopy(mean)
  for k in range(0,numComponents):
    reconstruct += Vh[k,:] * W[ran,k]

  plt.imshow(np.matrix(reconstruct).reshape(64,64).T, cmap=plt.cm.gray)
  plt.show()



# i. Plot the reconstruction error for k = 5, 10, 25, 50, 100, 200, 300, 399 principal components
#    and the sum of the squares of the last n-k (singular values)
#    [extra credit]
# STUDENT CODE TODO
