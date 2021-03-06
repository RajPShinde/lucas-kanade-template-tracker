import cv2 as cv
import glob
import numpy as np
import copy

images = []
original_frame = []
for file in glob.glob("baby/*.jpg"):
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

def gaussian_pyramid(img, tmp, levels = 3):
    img_copy_down, tmp_down = img.copy(), tmp.copy()
    img_copy_up, tmp_up = img.copy(), tmp.copy()
    gaussian_pyr_img_down, gaussian_pyr_temp_down = [img_copy_down], [tmp_down]
    gaussian_pyr_img_up, gaussian_pyr_temp_up = [img_copy_up], [tmp_up]
    for i in range(levels):
        img_copy_down = cv.pyrDown(img_copy_down)
        img_copy_up = cv.pyrUp(img_copy_up)
        gaussian_pyr_img_down.append(img_copy_down)
        gaussian_pyr_img_up.append(img_copy_up)
        
        tmp_down = cv.pyrDown(tmp_down)
        tmp_up = cv.pyrUp(tmp_up)
        gaussian_pyr_temp_down.append(tmp_down)
        gaussian_pyr_temp_up.append(tmp_up)
    return gaussian_pyr_img_down, gaussian_pyr_img_up, gaussian_pyr_temp_down, gaussian_pyr_temp_up
    

def lucas_kanade_tracker(img, tmp, rect, p_prev):
    iterations = 0
    cost_function = 1
    threshold = 0.001 #check different threshold
    W = affine_matrix(p_prev) 

    while cost_function > threshold:
        iterations = iterations + 1
        
        #form the warped image  - Step 1
        Iw = cv.warpAffine(img, W, (0, 0), flags=cv.INTER_CUBIC + cv.WARP_INVERSE_MAP)#combination trial      
        Iw = Iw[rect[0,1]:rect[2,1] , rect[0,0]:rect[2,0]]
        
        #calculating error - step 2
        error = tmp.flatten().astype(np.int)-Iw.flatten().astype(np.int)
#        error = error**2
#        cv2.imshow("error", cv2.subtract(tmp_2, Iw_2))
   
        #calculate the gradient of main image along x and y - step 3
        grad_x = cv.Sobel(np.float32(img), cv.CV_64F, 1, 0, ksize =9) #changing kernel size improved results at start
        grad_y = cv.Sobel(np.float32(img), cv.CV_64F, 0, 1, ksize =9)
        cv.imshow("y", grad_y)
        
        #calculating the warp of the gradient
        grad_x = cv.warpAffine(grad_x, W, (0, 0), flags=cv.INTER_CUBIC + cv.WARP_INVERSE_MAP)
        grad_y = cv.warpAffine(grad_y, W, (0, 0), flags=cv.INTER_CUBIC + cv.WARP_INVERSE_MAP)
        
        #selecting only the template area
        grad_x = grad_x[rect[0,1]:rect[2,1] , rect[0,0]:rect[2,0]]
        grad_y = grad_y[rect[0,1]:rect[2,1] , rect[0,0]:rect[2,0]]        
        
        #calculating jacobian and steepest descent image - step 4 and 5
        grid_x = np.asarray(list(range(grad_x.shape[1])))
        grid_y = np.asarray(list(range(grad_y.shape[0])))        
        grid_x , grid_y = np.meshgrid(grid_x, grid_y)         
        steepest_image = np.array([np.multiply(grad_x.flatten(),grid_x.flatten()),np.multiply(grad_y.flatten(),grid_x.flatten()),np.multiply(grad_x.flatten(),grid_y.flatten()),
             np.multiply(grad_y.flatten(),grid_y.flatten()),grad_x.flatten(),grad_y.flatten()]).T
    
        #hessian matrix - step 6
        H = np.dot(steepest_image.T,steepest_image)
#        cv.imshow("steepest", steepest_image)
        
        #update parameters - step 7
        p_change = np.dot(np.linalg.pinv(H), np.dot(steepest_image.T, error)) 
        
        #compositional image alignment trial -> not much change in the results
        p0,p1,p2,p3,p4,p5 = p_change[0], p_change[1], p_change[2], p_change[3], p_change[4], p_change[5]
        
#        p_prev[0] = p_prev[0] + p0 + p_prev[0]*p0 + p_prev[2]*p1
#        p_prev[1] = p_prev[1] + p1 + p_prev[1]*p0 + p_prev[3]*p1
#        p_prev[2] = p_prev[2] + p2 + p_prev[0]*p2 + p_prev[2]*p3
#        p_prev[3] = p_prev[3] + p3 + p_prev[1]*p2 + p_prev[3]*p3
#        p_prev[4] = p_prev[4] + p4 + p_prev[0]*p4 + p_prev[2]*p5
#        p_prev[5] = p_prev[5] + p5 + p_prev[1]*p4 + p_prev[3]*p5
        
        #calculate updated cost function
        cost_function = np.linalg.norm(p_change)
        #Normal LKT parameter update formula
        p_prev = p_prev + p_change*100
#        p_prev = p_prev - p_change*50
        W = affine_matrix(p_prev)

        if(iterations > 1000):
            break
    print("iterations : " + str(iterations))
    return p_prev

vid_output = cv.VideoWriter('lkt_baby_pyramid.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (images[0].shape[1], images[0].shape[0]))




rect1 = np.array([[160,83], [216,83], [216,148], [160,148]])

rect2 = np.array([[152,118], [227,118], [227,236], [152,236]])

rect3 = np.array([[182,65], [248,65], [248,115], [182,115]])

rect1_pyramid_layer1 = np.array([[84,39], [107,39], [107,73], [84,73]])
rect1_pyramid_layer2 = np.array([[42,21], [54,21], [54,34], [42,34]])
rect1_pyramid_layer3 = np.array([[21,10], [27,10], [27,15], [21,15]])

rect1_pyramid_layer1 = np.array([[84,39], [107,39], [107,73], [84,73]])
rect1_pyramid_layer2 = np.array([[42,21], [54,21], [54,34], [42,34]])
rect1_pyramid_layer3 = np.array([[21,10], [27,10], [27,15], [21,15]])

param_1 = np.zeros(6) 
param_2 = np.zeros(6)
param_3 = np.zeros(6)

p1_level = np.zeros(6)
p2_level = np.zeros(6)
p3_level = np.zeros(6)

template1 = images[0][rect1[0,1]:rect1[2,1] , rect1[0,0]:rect1[2,0]]
template2 = images[85][rect2[0,1]:rect2[2,1] , rect2[0,0]:rect2[2,0]]

rect_updated_1 = copy.deepcopy(rect1)
rect_updated_2 = copy.deepcopy(rect2)

for i in range(1,len(images)-1):
    print("Frame no : "+str(i))    
    if i < 120:
        img_1 = images[0]
        img_2 = cv.pyrDown(img_1)
        img_3 = cv.pyrDown(img_2)
        
        img_1_next = images[i]
        img_2_next = cv.pyrDown(img_1_next)
        img_3_next = cv.pyrDown(img_2_next)
        
        #3rd layer
        init_temp_0_0 = img_3
        init_temp_1_0 = img_3_next
        p1_level = lucas_kanade_tracker(init_temp_1_0, init_temp_0_0[rect1_pyramid_layer2[0,1]:rect1_pyramid_layer2[2,1] , 
                                                                 rect1_pyramid_layer2[0,0]:rect1_pyramid_layer2[2,0]], 
                                                                    rect1_pyramid_layer2, p1_level)
        
        
        #2nd layer
        init_temp_0_1 = img_2
        init_temp_1_1 = img_2_next
        p2_level = lucas_kanade_tracker(init_temp_1_1, init_temp_0_1[rect1_pyramid_layer1[0,1]:rect1_pyramid_layer1[2,1] , 
                                                                 rect1_pyramid_layer1[0,0]:rect1_pyramid_layer1[2,0]],rect1_pyramid_layer1, p1_level*2)
        
        
        #1st layer
        init_temp_0_2 = img_1
        init_temp_1_2 = img_1_next
        p_temp = (p1_level*4 + p2_level*2)
        cv.imshow("1", init_temp_0_2[rect1[0,1]:rect1[2,1], rect1[0,0]:rect1[2,0]])
        p3_level = lucas_kanade_tracker(init_temp_1_2, init_temp_0_2[rect1[0,1]:rect1[2,1], rect1[0,0]:rect1[2,0]], rect1, p_temp)
        
        print(p3_level)
        
        rect_trial_updated = np.array([[160+int(p3_level[4]),83+int(p3_level[5])], [216+int(p3_level[4]),83+int(p3_level[5])], 
                                       [216+int(p3_level[4]),148+int(p3_level[5])], [160+int(p3_level[4]),148+int(p3_level[5])]])
       
        w1 = affine_matrix(p3_level)
        #draw rectange as per updated parameters
        temp_rect_1 = np.dot(w1,np.vstack((rect1.T, np.ones((1,4))))).T

        #extract corner elements
        [xmax1, ymax1] = list(np.max(temp_rect_1, axis = 0).astype(np.int))
        [xmin1, ymin1] = list(np.min(temp_rect_1, axis = 0).astype(np.int))
        #create new rectangle/bounding box
        rect_updated_1=np.array([[xmin1,ymin1],[xmax1,ymin1],[xmax1,ymax1],[xmin1,ymax1]]) 

        output = cv.polylines(original_frame[i],[rect_updated_1],True,(0,0,255),thickness = 2)
        vid_output.write(output)
        cv.imshow("tracker",output)

    k = cv.waitKey(20) & 0xFF
    if k == 27:
        break 
vid_output.release()
cv.destroyAllWindows()

