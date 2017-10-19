# Image Classification Task

This repository is part of a project by Giancarlo Antonucci and Thomas Babb, submitted as part of the CDT in Industrially Focused Mathematical Modelling at The University of Oxford.

This code uses the Yale face database (YaleB_32x32.mat) to test the effectiveness of a hybrid facial recognition algorithm. The function imag_class.m takes inputs from the user and outputs the percentage success rate of the algorithm for those inputs.

To use it, download both the .m and .mat files and run them in Matlab.

--------------------------

## Choosing the training set

There are 2414 photos of 38 people in the Yale database. The user chooses which people are in the test set in one of two ways:

- imag_class('NumOfPeople', N) picks N random people from the 38 available and builds the test set from these.
- imag_class('People', V) builds the test set from the people indexed in V. For example, if V = [1 3 6], the test set will contain images from person 1, person 3, and person 6.

In either case, the function selects 10 random images of each person. All the other images are in the training set.

## Implementing the PCA algorithm.

% Find the "mean face" [average of all images in training set] and
% subtract it from all images in both the training and the test sets.

% Find the eigenvalues and eigenvectors of the corrected training set.

% Use the eigenvectors from above as a new basis. We transform our
% corrected training set into this bases then cut unimportant dimensions
% [diag(diag(PCAEigVals).^(-1/2)) weights each eigenvector by relative importance].

% Compute the within-class and between-class scatter matrices

% Find the projection that maximises between-class spacing and
% minimises within-class spacing

% For each member of the test set, find the nearest members of the training
% set to determine the person represented
