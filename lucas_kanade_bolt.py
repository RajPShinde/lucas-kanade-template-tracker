# https://www.youtube.com/watch?v=7soIa95QNDk
# https://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
# https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html
# https://github.com/waynezv/Computer-Vision/tree/master/HW4/code
# https://www.mathworks.com/matlabcentral/fileexchange/24677-lucas-kanade-affine-template-tracking?focused=5139141&tab=function
# https://cs.brown.edu/courses/csci1950-g/asgn/proj6/resources/ImageWarping.pdf
# https://docs.opencv.org/master/da/d85/tutorial_js_gradients.html

# https://github.com/topics/lucas-kanade
# https://github.com/AVINASH793/Lucas-Kanade-Tracker/blob/master/Lucas_Kanade_Own.py
# https://github.com/ziliHarvey/Lucas-Kanade-tracking-and-Correlation-Filters/blob/master/src/LucasKanadeAffine.py
# https://github.com/t-martyniuk/Lucas-Kanade-Tracking/blob/master/task3.py

# https://www.life2coding.com/crop-image-using-mouse-click-movement-python/
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
import sys, os
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2 as cv
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

path_bolt = '/home/prasheel/Workspace/ENPM673/Project4/Bolt2/img'

def read_bolt():    
    bolt_data = 294
    all_imgs = defaultdict(list)
    for i in range(1, bolt_data):
        if i < 10:
            string_path = path_bolt +"/000"+str(i)+".jpg"
        elif i >= 10 and i < 100:
            string_path = path_bolt +"/00"+str(i)+".jpg"
        else:
            string_path = path_bolt +"/0"+str(i)+".jpg"
        # print(i)
        img = cv.imread(string_path)
        all_imgs[i] = img
    return all_imgs

# Mouse callback
def select_point(event, x, y, flags, params):
    global point, template_image
    if event == cv.EVENT_LBUTTONDOWN:
        point=[(x, y)]

    elif event == cv.EVENT_LBUTTONUP:
        point.append((x, y))
        cv.rectangle(original_image, point[0], point[1], (0, 255, 0), 1)
        if len(point) == 2:
            region_of_interest = original_image[point[0][1]:point[1][1], point[0][0]:point[1][0]]
            template_image = region_of_interest
            cv.imshow("Cropped", region_of_interest)

def read_template(img):
    cv.namedWindow("FirstFrame")
    cv.setMouseCallback("FirstFrame", select_point)
    while(1):
        cv.imshow('FirstFrame', all_images[1])
        key = cv.waitKey(1)
        if key == 27:
            break

def get_region_of_interest(image, roi_points):
    return image[roi_points[1]:roi_points[3], roi_points[0]:roi_points[2]]

def jacobian(x,y):
    return np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]], dtype=np.float32)

def affine_matrix(p):    
    W = np.array([[1, 0, 0], [0, 1, 0]]) + p.reshape((2,3), order = 'F')
    return W

def lucas_kanade_tracker(ref_points, template_image, next_image):
    threshold = 0.1
    # dp = 10
    # p = np.zeros(6)
    p = np.array([0,0,0,0,0,0], dtype=np.float32)
    dp = np.array([100,100,100,100,100,100], dtype=np.float32)

    template_image = cv.equalizeHist(template_image)
    ref_points = [ref_points[0][0], ref_points[0][1], ref_points[1][0], ref_points[1][1]]
    count = 0
    W = affine_matrix(p)
    while np.linalg.norm(dp) > threshold: 
        count +=1
        # Warped image with template region of interest.
        warp_image = get_region_of_interest(cv.warpAffine(next_image, W, (next_image.shape[1], next_image.shape[0]), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        # warp_image = get_region_of_interest(cv.warpAffine(next_image, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)

        # Compute error image
        err = template_image - warp_image
        err_image = err.reshape(-1, 1)

        Ix = cv.Sobel(next_image, cv.CV_64F, dx=1, dy=0, ksize=5) # ksize -1 for scharr
        Iy = cv.Sobel(next_image, cv.CV_64F, dx=0, dy=1, ksize=5)
        # abs_ix = np.absolute(Ix)
        # abs_iy = np.absolute(Iy)

        # Ix = np.uint8(abs_ix)
        # Iy = np.uint8(abs_iy)
        warp_ix = get_region_of_interest(cv.warpAffine(Ix, W, (Ix.shape[1], Ix.shape[0]), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        warp_iy = get_region_of_interest(cv.warpAffine(Iy, W, (Iy.shape[1], Iy.shape[0]), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        # warp_ix = get_region_of_interest(cv.warpAffine(Ix, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        # warp_iy = get_region_of_interest(cv.warpAffine(Iy, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)

        # cv.imshow('next', next_image)
        cv.imshow('next_image', next_image)
        cv.imshow('warp_image', warp_image)
        #calculating jacobian and steepest descent image - step 4 and 5
        # grid_x = np.asarray(list(range(warp_ix.shape[1])))
        # grid_y = np.asarray(list(range(warp_iy.shape[0])))        
        # grid_x , grid_y = np.meshgrid(grid_x, grid_y)         
        # steepestImg = np.array([np.multiply(warp_ix.flatten(),grid_x.flatten()),np.multiply(warp_iy.flatten(),grid_x.flatten()),np.multiply(warp_ix.flatten(),grid_y.flatten()),
        #      np.multiply(warp_iy.flatten(),grid_y.flatten()),warp_ix.flatten(),warp_iy.flatten()]).T
    
        # #hessian matrix - step 6
        # H = np.dot(steepestImg.T,steepestImg)
        # cv.imshow("steepest", steepestImg)
        
        # #update parameters - step 7
        # p_change = np.dot(np.linalg.pinv(H), np.dot(steepestImg.T, error)) 
        
        # #compositional image alignment trial -> not much change in the results
        # p0,p1,p2,p3,p4,p5 = p_change[0], p_change[1], p_change[2], p_change[3], p_change[4], p_change[5]
        
        # p_prev[0] = p_prev[0] + p0 + p_prev[0]*p0 + p_prev[2]*p1
        # p_prev[1] = p_prev[1] + p1 + p_prev[1]*p0 + p_prev[3]*p1
        # p_prev[2] = p_prev[2] + p2 + p_prev[0]*p2 + p_prev[2]*p3
        # p_prev[3] = p_prev[3] + p3 + p_prev[1]*p2 + p_prev[3]*p3
        # p_prev[4] = p_prev[4] + p4 + p_prev[0]*p4 + p_prev[2]*p5
        # p_prev[5] = p_prev[5] + p5 + p_prev[1]*p4 + p_prev[3]*p5
        
        I = np.vstack((warp_ix.ravel(), warp_iy.ravel())).T

        deltaI = np.zeros((template_image.shape[0] * template_image.shape[1], 6))

        # Compute steepest gradient
        for i in range(template_image.shape[0]):
            for j in range(template_image.shape[1]):
                I_individual = np.array([I[i * template_image.shape[1] + j]]).reshape(1, 2)

                jacobian_individual = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])

                deltaI[i * template_image.shape[1] + j] = I_individual @ jacobian_individual

        # plt.imshow(deltaI, cmap='gray')
        # plt.show()
        H = deltaI.T @ deltaI
        # if np.count_nonzero(H) == 0 :
        #     break

        dp = np.linalg.inv(H) @ (deltaI.T) @ err_image

        print(dp)
        # # p += dp
        # p[0] = p[0] + dp[0, 0] + p[0]*dp[0, 0] + p[2]*dp[1, 0]
        # p[1] = p[1] + dp[1, 0] + p[1]*dp[0, 0] + p[3]*dp[1, 0]
        # p[2] = p[2] + dp[2, 0] + p[0]*dp[2, 0] + p[2]*dp[3, 0]
        # p[3] = p[3] + dp[3, 0] + p[1]*dp[2, 0] + p[3]*dp[3, 0]
        # p[4] = p[4] + dp[4, 0] + p[0]*dp[4, 0] + p[2]*dp[5, 0]
        # p[5] = p[5] + dp[5, 0] + p[1]*dp[4, 0] + p[3]*dp[5, 0]

        p[0] += dp[0, 0]*10
        p[1] += dp[1, 0]*10
        p[2] += dp[2, 0]*10
        p[3] += dp[3, 0]*10
        p[4] += dp[4, 0]*10
        p[5] += dp[5, 0]*10
        if(count > 1000):
            break
        # steepest_images = [np.zeros(shape=template_image.shape) for _ in range(6)]
        # # print(len(steepest_images))
        # for y in range(template_image.shape[0]):
        #     for x in range(template_image.shape[1]):
        #         curr_values = np.array([warp_ix[y,x], warp_iy[y,x]]).dot(jacobian(x + ref_points[0], y + ref_points[1]))
        #         for i in range(6):
        #             steepest_images[i][y,x] = curr_values[i]
        # # plt.imshow(steepest_images[3], cmap='gray')
        # # plt.show()
        # steep_im_transp = np.array([steepest_images[i].flatten() for i in range(6)])
        # hessian = steep_im_transp.dot(steep_im_transp.transpose())
        # sd_params = steep_im_transp.dot(err.flatten())
        # # print("count: ", count)
        # # print(hessian)
        # # if np.count_nonzero(hessian) == 0 :
        # #     break
        # dp = np.linalg.inv(hessian).dot(sd_params)
        # p += dp
        # W = np.array([[1 + p[0], p[2], p[4]],
        #               [p[1], 1 + p[3], p[5]]], dtype=np.float32)
        W = affine_matrix(p)

    affine_transform_matrix = affine_matrix(p) #np.array([[1.0 + p[0], p[1], p[2]], [p[3], 1.0 + p[4], p[5]]])
    return affine_transform_matrix

all_images = read_bolt()
# original_image = all_images[1]
# read_template(all_images[1])

# print(point)
# print(template_image.shape)

template_image = np.array([])
for i in range(2, 294):
    print(i);
    if i < 51:
        # point = np.array([])
        point = np.array([(269, 75), (303, 139)])
        region_of_interest = all_images[1][point[0][1]:point[1][1], point[0][0]:point[1][0]]
        template_image = region_of_interest
        

        lab = cv.cvtColor(template_image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        template_image = clahe.apply(l)
        rect1 = np.array([[point[0][0], point[0][1]], [point[1][0], point[0][1]], [point[1][0], point[1][1]], [point[0][0], point[1][1]]])

        # next_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2GRAY)
        lab_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l1, a1, b1 = cv.split(lab_image)
        # Applying CLAHE to L-channel---
        clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        next_image = clahe1.apply(l1)

        new_affine_matrix = lucas_kanade_tracker(point, template_image, next_image)
        rectTemp = np.dot(new_affine_matrix, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i > 50 and i < 101:
        # point = np.array([])
        point = np.array([(238, 78), (272, 142)])
        region_of_interest = all_images[1][point[0][1]:point[1][1], point[0][0]:point[1][0]]
        template_image = region_of_interest
        

        lab = cv.cvtColor(template_image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        template_image = clahe.apply(l)
        rect1 = np.array([[point[0][0], point[0][1]], [point[1][0], point[0][1]], [point[1][0], point[1][1]], [point[0][0], point[1][1]]])

        # next_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2GRAY)
        lab_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l1, a1, b1 = cv.split(lab_image)
        # Applying CLAHE to L-channel---
        clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        next_image = clahe1.apply(l1)

        new_affine_matrix = lucas_kanade_tracker(point, template_image, next_image)
        rectTemp = np.dot(new_affine_matrix, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i > 100 and i < 151:
        # point = np.array([])
        point = np.array([(253, 76), (287, 140)])
        region_of_interest = all_images[1][point[0][1]:point[1][1], point[0][0]:point[1][0]]
        template_image = region_of_interest
        

        lab = cv.cvtColor(template_image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        template_image = clahe.apply(l)
        rect1 = np.array([[point[0][0], point[0][1]], [point[1][0], point[0][1]], [point[1][0], point[1][1]], [point[0][0], point[1][1]]])

        # next_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2GRAY)
        lab_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l1, a1, b1 = cv.split(lab_image)
        # Applying CLAHE to L-channel---
        clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        next_image = clahe1.apply(l1)

        new_affine_matrix = lucas_kanade_tracker(point, template_image, next_image)
        rectTemp = np.dot(new_affine_matrix, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i > 150 and i < 201:
        # point = np.array([])
        point = np.array([(276, 75), (310, 139)])
        region_of_interest = all_images[1][point[0][1]:point[1][1], point[0][0]:point[1][0]]
        template_image = region_of_interest
        

        lab = cv.cvtColor(template_image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        template_image = clahe.apply(l)
        rect1 = np.array([[point[0][0], point[0][1]], [point[1][0], point[0][1]], [point[1][0], point[1][1]], [point[0][0], point[1][1]]])

        # next_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2GRAY)
        lab_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l1, a1, b1 = cv.split(lab_image)
        # Applying CLAHE to L-channel---
        clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        next_image = clahe1.apply(l1)

        new_affine_matrix = lucas_kanade_tracker(point, template_image, next_image)
        rectTemp = np.dot(new_affine_matrix, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i > 200 and i < 251:
        # point = np.array([])
        point = np.array([(352, 77), (386, 141)])
        region_of_interest = all_images[1][point[0][1]:point[1][1], point[0][0]:point[1][0]]
        template_image = region_of_interest
        

        lab = cv.cvtColor(template_image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        template_image = clahe.apply(l)
        rect1 = np.array([[point[0][0], point[0][1]], [point[1][0], point[0][1]], [point[1][0], point[1][1]], [point[0][0], point[1][1]]])

        # next_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2GRAY)
        lab_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l1, a1, b1 = cv.split(lab_image)
        # Applying CLAHE to L-channel---
        clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        next_image = clahe1.apply(l1)

        new_affine_matrix = lucas_kanade_tracker(point, template_image, next_image)
        rectTemp = np.dot(new_affine_matrix, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i > 250 and i < 294:
        # point = np.array([])
        point = np.array([(365, 106), (399, 170)])
        region_of_interest = all_images[1][point[0][1]:point[1][1], point[0][0]:point[1][0]]
        template_image = region_of_interest
        

        lab = cv.cvtColor(template_image, cv.COLOR_BGR2LAB)
        l, a, b = cv.split(lab)
        clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        template_image = clahe.apply(l)
        rect1 = np.array([[point[0][0], point[0][1]], [point[1][0], point[0][1]], [point[1][0], point[1][1]], [point[0][0], point[1][1]]])

        # next_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2GRAY)
        lab_image = cv.cvtColor(all_images[i], cv.COLOR_BGR2LAB)
        # Splitting the LAB image to different channels
        l1, a1, b1 = cv.split(lab_image)
        # Applying CLAHE to L-channel---
        clahe1 = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        next_image = clahe1.apply(l1)

        new_affine_matrix = lucas_kanade_tracker(point, template_image, next_image)
        rectTemp = np.dot(new_affine_matrix, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)


    k = cv.waitKey(20) & 0xFF
    if k == 27:
        break 
        
cv.destroyAllWindows()
