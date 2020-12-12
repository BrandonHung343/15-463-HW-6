import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
import cp_hw6
import cv2  
import os
from skimage import io, feature, color
from skimage.transform import rescale
from scipy import interpolate, signal


def disp(im):
    plt.imshow(im)
    plt.show()

def saveIm(name, im):
    io.imsave(name + '.jpeg', im, quality=95)

def saveData(name, data, multiple=False):
    # if multiply, saves data as a list
    with open(name+'.npy', 'wb') as f:
        if multiple:
            for item in data: 
                np.save(f, item)
        else:
            np.save(f, data)
    print('Saved to file ' + name + '.npy')

def loadData(name):
    data = None
    with open(name+'.npy', 'rb') as f:
        data = np.load(f)
    return data

def loadIm(name):
    return io.imread(name)

def findAreas(name):
    im = loadIm(name)
    disp(im)

def in_bounds(r1, r2, c1, c2, im):
   return im[r1:r2, c1:c2]

def camToPlane(R, p, t):
    return np.matmul(np.transpose(R), np.transpose(p) - t)

def spatial_temporal_shadow_edge(path, pxSpeed, small=False):
    display = False
    test = False
    threshold = 0

    cols = 1024
    rows = 768

    vrow1 = 130*2
    vrow2 = 150*2
    vcol1 = 175*2
    vcol2 = 370*2

    hrow1 = 680
    hrow2 = 760
    hcol1 = 240
    hcol2 = 780

    hSplit = 146*2

    if small:
        cols = 512
        rows = 384

        vrow1 = 130
        vrow2 = 150
        vcol1 = 175
        vcol2 = 370

        hrow1 = 400
        hrow2 = 440
        hcol1 = 120
        hcol2 = 400

        hSplit = 146

    imageList = os.listdir(path)
    frames = len(imageList)
    # print(frames)


    hlines = np.zeros((2, frames))
    vlines = np.zeros((2, frames))

    vid = np.zeros((rows, cols, frames))
    count = 0

    for image in os.listdir(path):
        im = loadIm(path + image)
        vid[:, :, count] = color.rgb2gray(im)
        count += 1

    imax = np.amax(vid, axis=2)
    imin = np.amin(vid, axis=2)
    # print(imin.shape)
    ishadow = imax / 2 + imin / 2

    tLines = imax - imin
    contrastBound = 0.2
    lowContrast = np.where(tLines < contrastBound)
    tStart = 64

    for d in range(frames-2):
        idiff = cv2.GaussianBlur(vid[:, :, d], (5, 5), 0.75) - ishadow
        idiff[lowContrast] = 0
        signs = np.sign(idiff)
        print(d)
        
        signGrad = np.diff(signs, axis=1)
        # signGrad[signGrad < threshold] = 0



        vSigns = in_bounds(vrow1, vrow2, vcol1, vcol2, signGrad)
        hSigns = in_bounds(hrow1, hrow2, hcol1, hcol2, signGrad)

        rightPoints = np.where(signGrad > 0)
        leftPoints = np.where(signGrad < 0)
        # shadedCols = np.arange(np.min(leftCol), np.min
        tLines[rightPoints] = d+tStart
        tLines[leftPoints] = d+tStart

        vRows, vCols = np.where(vSigns > threshold)
        vCols += vcol1
        vRows += vrow1
        hRows, hCols = np.where(hSigns > threshold)
        hCols += hcol1
        hRows += hrow1


        vx = np.transpose(np.vstack((vCols, np.ones((vCols.shape[0])))))
        # print(vx)
        hx = np.transpose(np.vstack((hCols, np.ones((hCols.shape[0])))))

        vy = vRows
        hy = hRows

        # if (d > 20 or d == 59):
        #     plt.imshow(vSigns)
        #     plt.show()
        #     plt.imshow(hSigns)
        #     plt.show()

        # print(hx, hy)


        vCoeff = np.linalg.lstsq(vx, vy, rcond=None)[0]
        hCoeff = np.linalg.lstsq(hx, hy, rcond=None)[0]

        vlines[:, d] = vCoeff
        hlines[:, d] = hCoeff
        # print(hCoeff)

        vxInt = int((hSplit - vCoeff[1]) / vCoeff[0])
        vyInt = int(-vCoeff[1] / vCoeff[0])
        # print(vyInt)
        vRange = np.arange(vxInt-1, vyInt)
        # print(vRange)
        # print(rows)

        # print(hCols, hRows)

        hxInt = int((rows - hCoeff[1]) / hCoeff[0])

        hyInt = int((hSplit - hCoeff[1]) / hCoeff[0])
        # print(hxInt)
        # print(hyInt)
        hRange = np.arange(np.minimum(hxInt, hyInt)+1, np.maximum(hxInt, hyInt)-2)

        currIm = vid[:, :, d]
        
        if (d == 21):
            plt.imshow(np.dstack((currIm, currIm, currIm)))
            plt.plot(vRange, vCoeff[0] * vRange + vCoeff[1])
            plt.plot(hRange, hCoeff[0] * hRange + hCoeff[1])
            print(hCoeff)
            
            # # hRange = np.arange(hcol1, hcol1)
            plt.show()


    tLines[lowContrast] = 0

    # interpolate the tLines
    # left = 0
    # right = 0
    # row = 0
    # col = 0
    # for row in range(rows):
    #     left = 0 
    #     col = 0
    #     while (col < cols):
    #         if tLines[row, col] > 64:
    #             left = tLines[row, col]
    #             colList = []
    #             scout = col + 1
    #             # print(col)
    #             while (scout < cols and tLines[row, scout] < 64):
    #                 colList.append(scout)
    #                 scout += 1
    #                 # print(scout)
    #                 # print(tLines[row, scout])
    #                 # input('sda')
    #             if scout < cols:
    #                 for j in colList:
    #                     tLines[row, j] = left
    #                     # print(left)
    #                     # input('hi')
    #         col += 1



    plt.imshow(tLines, cmap='jet')
    plt.show()

    if display:
        for d in range(frames):
            vm = vlines[0, d]
            vb = vlines[1, d]
            vxs = rowRange[:hSplit]
            vPoints = np.multiply(vm, vxs) + vb

            hm = hlines[0, d]
            hb = hlines[1, d]
            hxs = rowRange[hSplit:]
            hPoints = np.multiply(hm, hxs) + hb

            plt.imshow(vid[:, :, d])
            plt.plot(vxs, vPoints, linewidth=2)
            plt.plot(hxs, hPoints, linewidth=2)
            plt.show()



    if not test: 
        saveData('vid', vid)
        saveData('vlines', vlines)
        saveData('hlines', hlines)
        saveData('tLines', tLines)


def load_calibration(path):
    file = path + 'extrinsic_calib.npz'
    fi = open(file, 'rb')
    extOut = np.load(fi)
    tvecH = extOut['tvec_h']
    print('tH', tvecH)
    rmatH = extOut['rmat_h']
    print('rH', rmatH)
    tvecV = extOut['tvec_v']
    print('tV', tvecV)
    rmatV = extOut['rmat_v']
    print('rH', rmatV)
    fi.close()
    fi = open('../data/calib/intrinsic_calib.npz', 'rb')
    intOut = np.load(fi)
    K = intOut['mtx']
    dist = intOut['dist']
    print('K', K)
    print('dist', dist)
    fi.close()
    return tvecH, rmatH, tvecV, rmatV, K, dist

def shadow_line_calib(dPath, extPath, hSplit, rows):
    vid = loadData(dPath + 'vid')
    hlines = loadData(dPath + 'hlines')
    vlines = loadData(dPath + 'vlines')
    tvecH, rmatH, tvecV, rmatV, K, dist = load_calibration(extPath)

    row, col, frames = vid.shape


    hPoints3D = np.zeros((3, frames*2))
    vPoints3D = np.zeros((3, frames*2))
    count = 0

    for frame in range(frames-2):
        hCoeff = hlines[:, frame]
        hx1 = int((rows - hCoeff[1]) / hCoeff[0])
        hx2 = int((hSplit - hCoeff[1]) / hCoeff[0])
        print(hCoeff)
        mH = hCoeff[0]
        bH = hCoeff[1]
        hy1 = mH * hx1 + bH
        hy2 = mH * hx2 + bH

        hPointMat = np.array([[hy1, hx1],
                              [hy2, hx2]])
        hRays = cp_hw6.pixel2ray(hPointMat, K, dist)
        hPlaneRays = np.matmul(np.transpose(rmatH), np.transpose(np.squeeze(hRays)))
        hPlanePoints = camToPlane(rmatH, [[hy1, hx1, 0],
                                          [hy2, hx2, 0]], tvecH)
        print('Point', hPlanePoints[:, 0])
        print('Ray', hPlaneRays[:, 0])
        ht1 = -hPlanePoints[2, 0] / hPlaneRays[2, 0]
        ht2 = -hPlanePoints[2, 1] / hPlaneRays[2, 1]
        hP1 = hPlanePoints[:, 0] + ht1 * hPlaneRays[:, 0]
        hP2 = hPlanePoints[:, 1] + ht2 * hPlaneRays[:, 1]
        hP = np.transpose(np.vstack((hP1, hP2)))
        hp = np.matmul(rmatH, hP) + tvecH
        hPoints3D[:, count:count+2] = hp

        vCoeff = vlines[:, frame]
        vx1 = int((hSplit - vCoeff[1]) / vCoeff[0])
        vx2 = int(-vCoeff[1] / vCoeff[0])
        mV = vCoeff[0]
        bV = vCoeff[1]
        vy1 = mV * vx1 + bV
        vy2 = mV * vx2 + bV
        
        vPointMat = np.array([[vy1, vx1],
                              [vy2, vx2]])
        vRays = cp_hw6.pixel2ray(vPointMat, K, dist)
        print('Vray', np.squeeze(vRays))
        vPlaneRays = np.matmul(np.transpose(rmatV), np.transpose(np.squeeze(vRays)))
        print('Trans', np.matmul(rmatV, vPlaneRays))
        vPlanePoints = camToPlane(rmatV, [[vy1, vx1, 0],
                                          [vy2, vx2, 0]], tvecV)
        print('vPlanePoint', vPlanePoints)
        print('vCamPoints', np.matmul(rmatV, vPlanePoints) + tvecV)
         #print('vRay', vPlaneRays)
        vt1 = -vPlanePoints[2, 0] / vPlaneRays[2, 0]
        vt2 = -vPlanePoints[2, 1] / vPlaneRays[2, 1]
        vP1 = vPlanePoints[:, 0] + vt1 * vPlaneRays[:, 0]
        vP2 = vPlanePoints[:, 1] + vt2 * vPlaneRays[:, 1]
        vP = np.transpose(np.vstack((vP1, vP2)))
        print(vP.shape)
        vp = np.matmul(rmatV, vP) + tvecV
        vPoints3D[:, count:count+2] = vp
        # print(vPoints3D

        # print('P3:', vx1, vy1)
        # print('P4:', vx2, vy2)
        # print('P1:', hx1, hy1)
        # print('P2:', hx2, hy2)


        # plt.plot([hx1, hx2], [hy1, hy2], c='blue')
        # plt.plot([vx1, vx2], [vy1, vy2], c='orange')
        # plt.show()



        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # xs = vp[0, :]
        # ys = vp[1, :]
        # zs = vp[2, :]
        # ax.plot(xs, -ys, zs, c='orange')
        # xs = hp[0, :]
        # ys = hp[1, :]
        # zs = hp[2, :]
        # ax.plot(xs, -ys, zs, c='blue')
        # plt.show()


        count += 2

    saveData('vPoints3D', vPoints3D)
    saveData('hPoints3D', hPoints3D) 


def shadow_plane_calib(path):
    vPoints3D = loadData(path + 'vPoints3D')
    hPoints3D = loadData(path + 'hPoints3D')
    dim, numPoints = vPoints3D.shape
    numPoints = numPoints // 2
    count = 0

    shadowPlanes = np.zeros((3, numPoints, 2))

    for i in range(numPoints-2):
        P1 = hPoints3D[:, count]
        P2 = hPoints3D[:, count+1]
        P3 = vPoints3D[:, count]
        P4 = vPoints3D[:, count+1]

        n = np.cross(P2 - P1, P4 - P3)
        n = n / np.linalg.norm(n)
        print(n)

        shadowPlanes[:, i, 0] = np.transpose(P1)
        shadowPlanes[:, i, 1] = np.transpose(n)

        d = -np.dot(P1, n)

        xx, yy = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        z = (-n[0] * xx - n[1] * yy - d) / n[2]

        # plt3d = plt.figure().gca(projection='3d')
        # plt3d.plot_surface(xx, yy, z, alpha=0.2)
        # plt.show()


        count += 2

    saveData('shadowPlanes', shadowPlanes)

    # print(vPoints3D.shape)


def reconstruction(dPath, cropX, cropY, extPath, tStart, refImage):
    tLines = loadData(dPath + 'tLines')
    shadowPlanes = loadData(dPath + 'shadowPlanes')
    tvecH, rmatH, tvecV, rmatV, K, dist = load_calibration(extPath)
    # vlines = loadData(dPath + 'vlines')
    startCol = 285
    startRow = 300

    outlierThreshold = 500

    xx, yy = np.meshgrid(cropY, cropX)
    gridSize = xx.shape
    print(gridSize)
    numPoints = gridSize[0] * gridSize[1]
    xVec = np.reshape(xx, numPoints, 1)
    yVec = np.reshape(yy, numPoints, 1)
    xyPoints = np.transpose(np.vstack((xVec, yVec)))
    xyPoints = np.expand_dims(xyPoints, 1)
    print(xyPoints.shape)
    xyPoints = xyPoints.astype(np.double)

    rays = np.squeeze(cp_hw6.pixel2ray(xyPoints, K, dist))

    depthMap = np.zeros((gridSize[0], gridSize[1], 3))
    tLines = np.array(tLines)
    threshold = 64
    print(xyPoints[-1, :])

    for i in range(numPoints):
        point = xyPoints[i, :]
        point = point[0]
        # print(point[0])
        row = int(point[0])
        col = int(point[1])

        ray = rays[i, :]
        val = tLines[row, col]
        if val < threshold:
            continue
        val = int(val)
        P1 = shadowPlanes[:, val-tStart, 0]
        n = shadowPlanes[:, val-tStart, 1]
        p = np.array([row, col, 0])

        t = np.dot(p - P1, n) / np.dot(ray, n)

        P = p + ray * t
        # print(col - startCol)
        # print(row)

        if np.abs(P[0]) < outlierThreshold:
            continue

        depthMap[row - startRow, col - startCol :] = P

    dXs = depthMap[:, :, 0]
    dYs = depthMap[:, :, 1]
    dZs = depthMap[:, :, 2]

    crop = refImage[300:630, 285:750]
    crop = np.reshape(crop, (numPoints,3))/ 255
    # crop = color.rgb2gray(crop) 



    print(crop.shape)
    print(dXs.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect(aspect='equal')

    ax.scatter(dXs, dYs, dZs, c=crop)
    plt.show()



def main():

    # Section 1.1: Temporal and spatial edge detections
    spatial_temporal_shadow_edge('../data/frog/v1shad/', 0, False)


    # Section 1.2: Calibration
    # Intrinsic and extrinsic calibrations
    path = '../data/frog/v1/'
    load_calibration(path)

    # Shadow line calibration
    shadow_line_calib('',  '../data/frog/v1/', 292, 768)

    # Shadow plane calibration
    shadow_plane_calib('')


    # Section 1.3: Reconstruction
    refImage = loadIm('../data/frog/v1/000001.jpg')
    reconstruction('', np.arange(285, 750), np.arange(300, 630), '../data/frog/v1/', 64, refImage)





if __name__ == '__main__':
    main()
