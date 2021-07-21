from pathlib import Path
import pathlib
import cv2
import numpy as np
import csv
from numpy.core.shape_base import hstack
from scipy.spatial.transform import Rotation as R
import os

current_dir = pathlib.Path.cwd()

IMGFOLDER = current_dir.joinpath('images/')
POSFILE = current_dir.joinpath('pose_test.csv')
OUTFOLDER = current_dir.joinpath('imagesOut/')

MODE = 'FIRST' #other options: 'FIRST', 'MEAN'

STEP = 0.001 #mm

K = np.array([[3707.172386, 0.0, 1270.210342,0.0], [0.0, 3700.81825, 1058.7798870000001, 0.0], [0.0, 0.0, 1.0, 0.0]])
D = np.array([-0.188644, 0.145316, -0.001098, -0.000444, 0.0])

def main():
    images = []
    poses = []

    #read images
    for img in IMGFOLDER.glob('*.png'):
        images.append(img)
    images.sort()

    #read poses from file
    with open(str(POSFILE)) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rowFloat = [np.float(i) for i in row]
            rowArray = np.array(rowFloat[1:])
            rowArray = np.transpose(np.reshape(rowArray, (4,4)))
            poses.append(rowArray)

    initPose = getInitialPose(poses)
    rectPoses = getRectifingPoses(initPose, poses)
    rectifyImages(images, rectPoses)
    
def getInitialPose(poses):
    convPoses = []

    for pose in poses:
        #convert to euler
        r = R.from_matrix(pose[0:3,0:3])
        rot = r.as_euler('zyx', degrees=False)
        p = [rot[0], rot[1], rot[2], pose[0,3], pose[1,3], pose[2,3]]
        convPoses.append(p)
    
    if MODE == 'FIRST':
        print("MODE: FIRST")
        convPoses = np.array(convPoses)
        convInitPose = convPoses[0]
    elif MODE == 'MEAN':
        print("MODE MEAN")
        convPoses = np.array(convPoses)
        convInitPose = np.mean(convPoses, axis=0)
        #set x to first pose
        convInitPose[3] = convPoses[0][3]
    else:
        print("select MODE option!")
        exit()

    #convert back to matrix
    rot = R.from_euler('zyx',convInitPose[0:3], degrees=False)
    rot = rot.as_matrix()
    t = np.array(convInitPose[3:6])
    t = np.reshape(t, (3,1))

    initPose = np.vstack((np.hstack((rot,t)),[0,0,0,1]))

    print("INIT POSE: \n", initPose)
    print()

    return initPose

def rectifyImages(images, relPoses):
    ''' Rectify the images with the provided relativePoses'''

    # K inverse
    Kinv = np.zeros((4,3))
    Kinv[:3,:3] = np.linalg.inv(K[:3,:3])
    Kinv[-1,:] = [0, 0, 1]

    #for all images
    i = 0
    for imgPath in images:

        #load and undistort
        img = cv2.imread(str(imgPath))
        #img = cv2.undistort(img, K, D, None)

        #get euclidian homography matrix from relative pose
        relR = np.array(relPoses[i][0:3,0:3])
        relT = np.array(relPoses[i][0:3,3])
        relTDiv = [x/0.26 for x in relT]
        relT = relTDiv  

        R = np.hstack((relR, np.array([[0],[0],[0]])))
        R = np.vstack((R, np.array([0,0,0,1])))
        
        T = np.identity(4)
        T[0:3,3] = relT

        H = np.linalg.multi_dot([K, R, T, Kinv])
        print("H: \n", H)

        #transfrom image and save
        img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        cv2.imwrite(str(OUTFOLDER.joinpath(imgPath.name)), img)
        i=i+1


def getRectifingPoses(initPose, poses):
    ''' Return all relative poses in a list '''
    i = 0
    relativePose = []
    for pose in poses:
        tempInitPose = np.array(initPose)
        #for each image increase x-dir via step size
        #tempInitPose[0,3] = initPose[0,3]+(i*STEP)
        rPose = computeRelativePose(tempInitPose, pose)
        relativePose.append(rPose)
        i=i+1
    
    return relativePose

def computeRelativePose(initPose, pose):
    ''' Compute the relative pose '''
    initR = initPose[0:3,0:3]
    initT = initPose[0:3,3]

    poseR = pose[0:3,0:3]
    poseT = pose[0:3,3]

    R_relToInit = initR @ poseR.T
    T_relToInit = initR @ (-poseR.T @ poseT) + initT

    relPose = np.hstack((R_relToInit, T_relToInit.reshape((3,1))))
    relPose = np.vstack((relPose, np.array([0,0,0,1])))
    return relPose

if __name__ == "__main__":
    main()
