## CSC320 Winter 2018 
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    pcord = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()
    confi, confb = copyutils.getWindow(confidenceImage, pcord, w)
    filli, fillb = copyutils.getWindow(filledImage, pcord, w)
    C = np.sum(confi[filli > 0]) / np.sum(fillb)
    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    pcord = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()
    grey_img = cv.cvtColor(inpaintedImage, cv.COLOR_BGR2GRAY)
    iinfo, ib = copyutils.getWindow(grey_img, pcord, w)
    finfo, fb = copyutils.getWindow(filledImage, pcord, w)
    xsize = iinfo.shape[0]
    ysize = iinfo.shape[1]
    for i in range(xsize):
        for j in range(ysize):
            if finfo[i, j] == 0:
                iinfo[i, j] = 0
    sobelx = cv.Sobel(iinfo, cv.CV_64F, 1, 0, ksize=5)
    sobely = cv.Sobel(iinfo, cv.CV_64F, 0, 1, ksize=5)
    graph = sobelx ** 2 + sobely ** 2
    s = graph.shape
    x, y = np.unravel_index(np.argmax(graph), s)
    Dx = sobelx[x,y]
    Dy = sobely[x,y]


    #########################################
    
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    pcord = (psiHatP.row(), psiHatP.col())
    w = psiHatP.radius()
    fronti, frontb = copyutils.getWindow(fillFront, pcord, w)
    imagei, imageb = copyutils.getWindow(filledImage, pcord, w)
    xsize = imagei.shape[0]
    ysize = imagei.shape[1]
    for i in range(xsize):
        for j in range(ysize):
            if fronti[i, j] == 0:
                imagei[i, j] = 0
    sobelx = cv.Sobel(imagei, cv.CV_64F, 1, 0, ksize = 5)
    sobely = cv.Sobel(imagei, cv.CV_64F, 0, 1, ksize = 5)
    i = np.sqrt(sobelx ** 2 + sobely ** 2)
    Nx = sobelx / i
    Ny = sobely / i
    #########################################

    return Ny, Nx