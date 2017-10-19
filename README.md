# Image Classification Task

This repository is part of a project by Giancarlo Antonucci and Thomas Babb, submitted as part of the CDT in Industrially Focused Mathematical Modelling at The University of Oxford.

This code uses the Yale face database (YaleB_32x32.mat) to test the effectiveness of a hybrid facial recognition algorithm. The function imag_class.m takes inputs from the user and outputs the percentage success rate of the algorithm for those inputs.

To use it, download both the .m and .mat files and run them in Matlab.

--------------------------

## Choosing the Training Set

There are 2414 photos, say images, of 38 people in the Yale database. The user chooses which people are in the test set in one of two ways:

- imag_class('NumOfPeople', N) picks N random people from the 38 available and builds the test set from these.
- imag_class('People', V) builds the test set from the people indexed in V. For example, if V = [1 3 6], the test set will contain images from person 1, person 3, and person 6.

In either case, the function selects 10 random images of each person. All the other images are put in the training set.

## Implementing the PCA Algorithm

PCA is a standard technique used to approximate the original data with lower dimensional feature vectors. We extract a new orthogonal basis, called the feature space, from the training samples and project both the test and the training images into this space. We then remove the dimensions that contain the lowest amount of information.

Additionally, to improve this algorithm we calculate the "mean face" [average of all images in training set] and it from every image, in both the training and test sets. We then do PCA as described above, and finally, we weight every basis vector in the feature space by relative importance.

## Implementing the LDA Algorithm

We know in advance the people that each image in the training space corresponds to. The images in the training set that are of the same person make up a "class". LDA aims to increase the separability of the classes in feature space. LDA searches for a projection which makes each class have the smallest within-class scatter (all the images in a given class are "close together") and the largest between-class scatter (each class is "far away" from every other class).

We compute the within-class and between-class scatter matrices. We build a projection into the LDA subspace based on these, and map each image into it.

## Implementing the K-Nearest Neighbours algorithm
In the LDA subspace, for each image in the test set, we find the nearest images in the training set to determine the person represented.

Finally, we estabilish the percentage success rate of the overall algorithm.
