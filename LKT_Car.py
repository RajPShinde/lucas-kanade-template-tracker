import cv2 as cv
import glob
import numpy as np
import copy

images = []
original_frame = []
for file in glob.glob("car/*.jpg"):
    frame = cv.imread(file)
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    images.append(grey)
    original_frame.append(frame)

#not being used currently
def adjust_brightness(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv.LUT(image, table)

def affine_matrix(p):    
    W = np.array([[1, 0, 0], [0, 1, 0]]) + p.reshape((2,3), order = 'F')
    return W

def lucas_kanade_tracker(img, tmp, rect, p_prev):
    iterations = 0
    cost_function = 1
    threshold = 0.006 #check different threshold
    W = affine_matrix(p_prev)

    
    tmp = cv.equalizeHist(tmp)
#    tmp_2 = copy.deepcopy(tmp)

    while cost_function > threshold:
        iterations = iterations + 1
        
        #form the warped image  - Step 1
        Iw = cv.warpAffine(img, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP)#combination trial      
        Iw = Iw[rect[0,1]:rect[2,1] , rect[0,0]:rect[2,0]]
        cv.imshow("warped", Iw)
#        Iw_2 = copy.deepcopy(Iw)   
        
        #Todo
        #Check if brightness has decreased for a frame and then increase it
        
        #calculating error - step 2
        error = tmp.flatten().astype(np.int)-Iw.flatten().astype(np.int)
#        cv2.imshow("error", cv2.subtract(tmp_2, Iw_2))
   
        #calculate the gradient of main image along x and y - step 3
        grad_x = cv.Sobel(np.float32(img), cv.CV_64F, 1, 0, ksize =3) #changing kernel size improved results at start
        grad_y = cv.Sobel(np.float32(img), cv.CV_64F, 0, 1, ksize =3)
#        cv.imshow("y", grad_y)
        
        #calculating the warp of the gradient
        grad_x = cv.warpAffine(grad_x, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP)
        grad_y = cv.warpAffine(grad_y, W, (0, 0), flags=cv.INTER_AREA + cv.WARP_INVERSE_MAP)
        
        #selecting only the template area
        grad_x = grad_x[rect[0,1]:rect[2,1] , rect[0,0]:rect[2,0]]
        grad_y = grad_y[rect[0,1]:rect[2,1] , rect[0,0]:rect[2,0]]        
        
        #calculating jacobian and steepest descent image - step 4 and 5
        grid_x = np.asarray(list(range(grad_x.shape[1])))
        grid_y = np.asarray(list(range(grad_y.shape[0])))        
        grid_x , grid_y = np.meshgrid(grid_x, grid_y)         
        steepestImg = np.array([np.multiply(grad_x.flatten(),grid_x.flatten()),np.multiply(grad_y.flatten(),grid_x.flatten()),np.multiply(grad_x.flatten(),grid_y.flatten()),
             np.multiply(grad_y.flatten(),grid_y.flatten()),grad_x.flatten(),grad_y.flatten()]).T
    
        #hessian matrix - step 6
        H = np.dot(steepestImg.T,steepestImg)
        cv.imshow("steepest", steepestImg)
        
        #update parameters - step 7
        p_change = np.dot(np.linalg.pinv(H), np.dot(steepestImg.T, error)) 
        
        #compositional image alignment trial -> not much change in the results
        p0,p1,p2,p3,p4,p5 = p_change[0], p_change[1], p_change[2], p_change[3], p_change[4], p_change[5]
        
        p_prev[0] = p_prev[0] + p0 + p_prev[0]*p0 + p_prev[2]*p1
        p_prev[1] = p_prev[1] + p1 + p_prev[1]*p0 + p_prev[3]*p1
        p_prev[2] = p_prev[2] + p2 + p_prev[0]*p2 + p_prev[2]*p3
        p_prev[3] = p_prev[3] + p3 + p_prev[1]*p2 + p_prev[3]*p3
        p_prev[4] = p_prev[4] + p4 + p_prev[0]*p4 + p_prev[2]*p5
        p_prev[5] = p_prev[5] + p5 + p_prev[1]*p4 + p_prev[3]*p5
        
        #calculate updated cost function
        cost_function = np.linalg.norm(p_change)
        #Normal LKT parameter update formula
#        p_prev = p_prev + p_change
        W = affine_matrix(p_prev)

        if(iterations > 2000):
            break
    print("iterations : " + str(iterations))
    return p_prev



rect1 = np.array([[70,51], [177,51], [177,138], [70,138]])
#rect2 = np.array([[132,63], [205,63], [205,119], [132,119]])
rect2 = np.array([[136,61], [209,61], [209,110], [136,110]])
#rect3 = np.array([[194,63], [270,63], [270,116], [194,116]])
rect3 = np.array([[182,65], [248,65], [248,115], [182,115]])

param_1 = np.zeros(6) 
param_2 = np.zeros(6)
param_3 = np.zeros(6)

template1 = images[0][rect1[0,1]:rect1[2,1] , rect1[0,0]:rect1[2,0]]
template2 = images[200][rect2[0,1]:rect2[2,1] , rect2[0,0]:rect2[2,0]]
#template3 = images[300][rect3[0,1]:rect3[2,1] , rect3[0,0]:rect3[2,0]]
template3 = images[264][rect3[0,1]:rect3[2,1] , rect3[0,0]:rect3[2,0]]

rect_updated_1 = copy.deepcopy(rect1)
rect_updated_2 = copy.deepcopy(rect2)
rect_updated_3 = copy.deepcopy(rect3)

for i in range(1,len(images)):
    print("Frame no : "+str(i))    
    if i < 200:
        param_1 = lucas_kanade_tracker(images[i], template1, rect1, param_1)        
        w = affine_matrix(param_1)
        #draw rectange as per updated parameters
        rectTemp = np.dot(w,np.vstack((rect1.T, np.ones((1,4))))).T
        #extract corner elements
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(original_frame[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i >= 200 and i < 264:
        param_2 = lucas_kanade_tracker(images[i], template2, rect2, param_2)
        w = affine_matrix(param_2)
        rectTemp = np.dot(w,np.vstack((rect2.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        rect_updated_2=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(original_frame[i],[rect_updated_2],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    elif i >= 264:
        param_3 = lucas_kanade_tracker(images[i], template3, rect3, param_3)
        w = affine_matrix(param_3)
        rectTemp = np.dot(w,np.vstack((rect3.T, np.ones((1,4))))).T
        [xmax, ymax] = list(np.max(rectTemp, axis = 0).astype(np.int))
        [xmin, ymin] = list(np.min(rectTemp, axis = 0).astype(np.int))
        rect_updated_3=np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]])        
        output = cv.polylines(original_frame[i],[rect_updated_3],True,(0,0,255),thickness = 2)
        cv.imshow("tracker",output)
    k = cv.waitKey(20) & 0xFF
    if k == 27:
        break 

cv.destroyAllWindows()