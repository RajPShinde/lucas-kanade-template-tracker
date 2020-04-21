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
out = cv.VideoWriter('lkt_on_bolt_code_new.avi', cv.VideoWriter_fourcc('M','J','P','G'), 15, (480, 270))

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

def lucas_kanade_tracker(ref_points, template_image, next_image, p_prev):
    threshold = 0.03
    max_iterations = 500
    # dp = 10
    # p = np.zeros(6)
    # p = p_prev
    # dp = np.array([120,120,120,120,120,120], dtype=np.float32)

    # template_image = cv.equalizeHist(template_image)
    ref_points = [ref_points[0][0], ref_points[0][1], ref_points[1][0], ref_points[1][1]]
    count = 0
    W = affine_matrix(p_prev)
    Ix = cv.Sobel(next_image, cv.CV_16S, dx=1, dy=0, ksize=-1) # ksize -1 for scharr
    Iy = cv.Sobel(next_image, cv.CV_16S, dx=0, dy=1, ksize=-1)
    # while(1):
    #     cv.imshow('Ix', Ix)
    #     cv.imshow('Iy', Iy)
    #     key = cv.waitKey(1)
    #     if key == 27:
    #         break
    # print(W)
    for it in range(max_iterations):
        count +=1
        # Warped image with template region of interest.
        warp_image = get_region_of_interest(cv.warpAffine(next_image, W, (next_image.shape[1], next_image.shape[0]), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        # warp_image = get_regihmmmon_of_interest(cv.warpAffine(next_image, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        cv.imshow("warped", warp_image)
        # step (2)
        error = template_image - warp_image


        gx_img_w = get_region_of_interest(cv.warpAffine(Ix, W, (Ix.shape[1], Ix.shape[0]), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        gy_img_w = get_region_of_interest(cv.warpAffine(Iy, W, (Iy.shape[1], Iy.shape[0]), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP), ref_points)
        I = np.vstack((gx_img_w.ravel(), gy_img_w.ravel())).T

        deltaI = np.zeros((template_image.shape[0] * template_image.shape[1], 6))

        # Compute steepest gradient
        for i in range(template_image.shape[0]):
            for j in range(template_image.shape[1]):
                I_individual = np.array([I[i * template_image.shape[1] + j]]).reshape(1, 2)

                jacobian_individual = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])

                deltaI[i * template_image.shape[1] + j] = I_individual @ jacobian_individual

        err = template_image - warp_image
        err_image = err.reshape(-1, 1).astype(np.float32)

        H = deltaI.T @ deltaI
        dp = np.linalg.inv(H) @ (deltaI.T) @ err_image
        # print(dp)
        p_prev[0] += dp[0, 0]
        p_prev[1] += dp[1, 0]
        p_prev[2] += dp[2, 0]
        p_prev[3] += dp[3, 0]
        p_prev[4] += dp[4, 0]
        p_prev[5] += dp[5, 0]

        if np.linalg.norm(dp) < threshold:
            break;
        W = affine_matrix(p_prev)

    print("iterations: ", count)
      return p_prev

# Main Program
all_images = read_bolt()
p_init1 = np.zeros(6)
p_init2 = np.zeros(6)
p_init3 = np.zeros(6)
p_init4 = np.zeros(6)
p_init5 = np.zeros(6)
p_init6 = np.zeros(6)
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

        p_init1 = lucas_kanade_tracker(point, template_image, next_image, p_init1)
        w = affine_matrix(p_init1)
        rectTemp = np.dot(w, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
        out.write(output)
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

        p_init2 = lucas_kanade_tracker(point, template_image, next_image, p_init2)
        w = affine_matrix(p_init2)
        rectTemp = np.dot(w, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
        out.write(output)
    elif i > 100 and i < 151:
        # point = np.array([])
        point = np.array([(253, 76), (287, 170)])
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

        p_init3 = lucas_kanade_tracker(point, template_image, next_image, p_init3)
        w = affine_matrix(p_init3)
        rectTemp = np.dot(w, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
        out.write(output)
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

        p_init4 = lucas_kanade_tracker(point, template_image, next_image, p_init4)
        w = affine_matrix(p_init4)
        rectTemp = np.dot(w, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
        out.write(output)
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

        p_init5 = lucas_kanade_tracker(point, template_image, next_image, p_init5)
        w = affine_matrix(p_init5)
        rectTemp = np.dot(w, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
        out.write(output)
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

        p_init6 = lucas_kanade_tracker(point, template_image, next_image, p_init6)
        w = affine_matrix(p_init6)
        rectTemp = np.dot(w, np.vstack((rect1.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(all_images[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
        out.write(output)

    k = cv.waitKey(20) & 0xFF
    if k == 27:
        break 
out.release()   
cv.destroyAllWindows()
