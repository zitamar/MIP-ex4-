import imageio as imageio
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from numpy import dot
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from skimage import transform
from skimage.transform import AffineTransform
from skimage.feature import canny



import numpy as np

import utils


def read_image(filename, presentation):
    """
    reads image with wanted representation
    :param filename: path to the image
    :param presentation: 1 for grayscale , 2 for rgb
    :return: the image in wanted
    """
    # im = imread(filename)
    im = imageio.imread(filename)
    one_pixel = im[1][1]

    if one_pixel.dtype == 'uint8':
        im_float = im.astype(np.float64)
        im_float /= 255

    # if one_pixel.dtype == 'unit8' :

    if presentation == 1:
        new_im = rgb2gray(im_float)
        return new_im
        # return plt.imshow(new_im, cmap=plt.cm.gray)
        # plt.show()
    elif presentation == 2:
        new_im = im_float
    return new_im

def display_inliers_on_pics(im1, im2, FP1,FP2, inliers=[], outliers=[]):
    im1_points_numbers = range(len(FP1))
    print("FP1", FP1[outliers])
    figure = plt.figure()
    plt1 = figure.add_subplot(121)
    plt2 = figure.add_subplot(122)
    plt1.imshow(im1, cmap=plt.cm.gray)
    # plt1.scatter(FP1[:,0],FP1[:,1])

    if (len(inliers)!=0):
        FP1_O = FP1[np.array(outliers)]
        FP2_O = FP2[np.array(outliers)]
        FP1=  FP1[np.array(inliers)]
        FP2 =  FP2[np.array(inliers)]

        plt1.scatter(FP1_O[:, 0], FP1_O[:, 1],marker='^')
        plt2.scatter(FP2_O[:, 0], FP2_O[:, 1],marker='^')



    for i, point in enumerate(im1_points_numbers):
        if (i==5):
            break
        plt1.annotate(point,(FP1[i,0],FP1[i,1]))
        plt1.scatter(FP1[i, 0], FP1[i, 1])

    im2_points_numbers = list(range(len(FP2)))

    plt2.imshow(im2, cmap=plt.cm.gray)
    for i, point in enumerate(im2_points_numbers):
        plt2.annotate(point,(FP2[i,0],FP2[i,1]))
        plt2.scatter(FP2[i, 0], FP2[i, 1])
    plt.show()



def centroid(arr):
    """
    find centroid of array of points
    :param arr:
    :return:
    """

    length = arr.shape[0]
    sum_x = np.sum(arr[:,0])
    sum_y = np.sum(arr[:,1])
    centro = np.array([sum_x/length , sum_y/length])
    return centro




def calcPointBasedReg(BLPoints,FUPoints):
    """
    follows algorithm to compute best fitting rigis transformation that aligns
    two sets of corresponding points
    :param BLPoints:
    :param FUPoints:
    :return:
    """

    # 1  compute the weighted centroids of both point sets

    BLCentroid = centroid(BLPoints)
    FUCentroid = centroid(FUPoints)
    #2 COMPUTE THE centered vectors
    BLcv = BLPoints - BLCentroid
    FUcv = FUPoints - FUCentroid

    #3 compute dxd covariance matrix
    W = np.identity(len(BLCentroid))
    X = BLcv.T
    # cov_mat = np.dot(np.dot(X,W),FUcv)
    cov_mat = np.dot(X,FUcv)
    # cov_mat=np.dot(X,np.dot(W,FUcv.T))

    #4 compute SVD

    U,S,V = np.linalg.svd(cov_mat)


    mat_in_middle_to_mult = np.identity(V.shape[1])
    mat_in_middle_to_mult[-1,-1] = np.linalg.det(dot(V,U.T))
    R = dot(V,dot(mat_in_middle_to_mult,U.T))

    t = FUCentroid - dot(R,BLCentroid)
    # t=t.reshape(2,1)

    rigidReg = np.zeros([3,3])
    rigidReg[0:2,0:2] = R
    rigidReg[0:2,2] = t
    rigidReg[-1,-1]=1
    return rigidReg




def calcDist(BLPoints, FUPoints, rigidReg):
    """
    calculates distance of every points been translation
    :param BLPoints:
    :param rigidReg:
    :return:
    """


    mat_size = FUPoints.shape[0]
    mat = np.ones([mat_size,3])

    mat[0:mat_size,0:2] = BLPoints
    transformation = dot(rigidReg,mat.T)#
    transformation = transformation.T
    transformation = transformation[:,:-1]
    # print ("transfor", transformation)
    # print ("FUPoints", FUPoints)
    # print ("BU", BLPoints)
    dist = euclidean_distances(FUPoints,transformation)
    dist = np.diagonal(dist)
    rmse = np.sqrt(mean_squared_error( FUPoints,transformation))
    print ("rmse " , rmse)


    return dist







    # return euclidean_distances(BLPoints, rigidReg), np.sqrt(mean_squared_error(BLPoint,rigidReg))


def display_one_and_edge(im1,im2):

    """
    displays im2 edges on top of im1
    :param im1:
    :param im2:
    :return:
    """


    plt.figure()
    im2_edge = canny(im2,sigma=2.1).astype(int)*255
    im1 = im1*255
    im1 = im1.astype(int)

    image = im1 + im2_edge

    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def calcRobustPointBasedReg(FUPoints, BLPoints):
    """
    using ransac to find inliers
    :param FUPoints:
    :param BLPoints:
    :return:
    """
    f, inlierIdx= utils.ransac(BLPoints,FUPoints,calcPointBasedReg,calcDist,4,100,20,0.1)
    return f, inlierIdx

def build_and_disp(im1,im2,points1,points2, toend=0):
    """
    build warped image and shows. if toend - take ransac
    :param im1:
    :param im2:
    :param points1:
    :param points2:
    :param toend: after ransac or not
    :return:
    """

    if (toend ==1):
        f, inliers = calcRobustPointBasedReg(points2,points1)
        inliers_1=points1[inliers]
        inliers_2=points2[inliers]
        print("inli", inliers_1,inliers_2)
        rigidReg = calcPointBasedReg(inliers_1,inliers_2)
        print ("final rmse", calcDist(inliers_1,inliers_2,rigidReg))
        tform = AffineTransform(np.linalg.inv(rigidReg))
        warped_img = transform.warp(im1, tform, preserve_range=True)
        outliers = [i for i in range(len(points2)) if i not in inliers]
        outliers = np.array(outliers)
        print ("inliers" ," outliers" , inliers, outliers)
        display_one_and_edge(im2, warped_img)

        display_inliers_on_pics(im2,im1,points2,points1,inliers, outliers)

        return
    rigidReg = calcPointBasedReg(points1,points2)
    tform = AffineTransform(np.linalg.inv(rigidReg))
    warped_img = transform.warp(im1,tform, preserve_range=True)
    display_one_and_edge(im2,warped_img)








    











if __name__ == "__main__":
    BL01 = read_image('BL01.tif',1)
    FU01 = read_image('FU01.tif',1)

    [BLPoint , FUPoints ] = utils.getPoints('with_outliers')

    build_and_disp(BL01, FU01 , BLPoint,FUPoints,1)
    # display_inliers_on_pics(BL01, FU01, BLPoint,FUPoints)
    # #
    # rigid = calcPointBasedReg(BLPoint,FUPoints)
    # calcDist(BLPoint,FUPoints, rigid)











