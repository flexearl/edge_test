import cv2
import numpy as np
import math

def apply_sobel_filter(img, kernel_size=5):
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=kernel_size) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=kernel_size) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=kernel_size)
    filtered_image_xy = cv2.convertScaleAbs(sobelxy)
    return filtered_image_xy

def apply_laplacian_filter(img, kernel_size=5):
    laplacian = cv2.Laplacian(img, ksize=kernel_size, ddepth=cv2.CV_64F)
    filtered_img = cv2.convertScaleAbs(laplacian)
    return filtered_img

def calculate_contour_distance(contour1, contour2): 
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    c_x1 = x1 + w1/2
    c_y1 = y1 + h1/2

    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c_x2 = x2 + w2/2
    c_y2 = y2 + h2/2

    return math.sqrt((c_x2-c_x1)**2 + (c_y2-c_y1)**2)

def merge_contours(contour1, contour2):
    return np.concatenate((contour1, contour2), axis=0)

def merge_matches(contours,current_cnt,matches, keys):
    for i in range(len(keys)):
        
        if keys[i] in matches and len(matches[keys[i]]) == 0:
            print("Key:", keys[i])
            current_cnt = merge_contours(current_cnt, contours[keys[i]])
            del matches[keys[i]]
        elif keys[i] in matches:
            print("Key:", keys[i])
            current_cnt = merge_matches(contours, current_cnt, matches, matches[keys[i]])
            
            del matches[keys[i]]
    return current_cnt


def group_contours(contours, threshold_distance):
    visited_points = {}
    grouped_contours = []
    matches = {}

    for i in range(len(contours)):

        for j in range(i+1,len(contours)):
            if i != j:
                distance = calculate_contour_distance(contours[i], contours[j])
                if distance< threshold_distance:
                    if i not in matches:
                        matches[i] = [j]
                    else:
                        matches[i].append(j)

        if i not in matches:
            print("Adding after")
            matches[i] = []

    merged_contours = []
    print(matches)
    
    for key in matches.copy():
       
        if key in matches:
            print("Key:", key)
            arr = matches[key]
            print(arr)
            if len(arr) != 0:
                merged = merge_matches(contours, contours[key], matches, arr)
                merged_contours.append(merged)
                
            else:
                merged_contours.append(contours[key])
       
    return tuple(merged_contours)

    for i in range(len(contours)-1):
        cnt_pos_a = get_contour_pos(contours[i])
        if cnt_pos_a not in visited_points:
            for j in range(i+1, len(contours)):
                cnt_pos_b = get_contour_pos(contours[j])
                if cnt_pos_b not in visited_points:
                    cnt_distance = (calculate_contour_distance(contours[i], contours[j]))
                    if (cnt_distance)<threshold_distance:
                        print("Found:",cnt_distance)
                        merged = merge_contours(contours[i], contours[j])
                        visited_points[cnt_pos_a] = True
                        visited_points[cnt_pos_b] = True
                        grouped_contours.append(merged)
        if cnt_pos_a not in visited_points:
            grouped_contours.append(contours[i])
    return tuple(grouped_contours)

def get_contour_pos(cnt):
    M = cv2.moments(cnt)
    cx=-1
    cy=-1
    if M['m00'] !=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) 
    return cx, cy

def apply_canny_filter(img, lower_threshold=100, higher_threshold=200):
    edges = cv2.Canny(image=img.copy(), threshold1=lower_threshold, threshold2=higher_threshold)
    return edges

def blur(img, strength=3):
    img_blur = cv2.GaussianBlur(img, (strength,strength), 0)
    return img_blur

def get_contours(img):
    contours, _ = cv2.findContours(img,  
    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours 

def close_img(img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing

def dilate_img(img, kernel_size):
    dilated = cv2.dilate(img.copy(), None, iterations=kernel_size)
    return dilated


def show_seperate_cnt(img, contours):
    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE )
   
    for i, cnt in enumerate(contours):
        black = np.zeros(img.shape, dtype = np.uint8)
        pos = get_contour_pos(cnt)
        print("Pos:", pos)
        cv2.drawContours(black, [cnt], -1, (255, 0, 0), 1) 
        cv2.imwrite(f"seperate_cnt/{i}.png",black)
       



def remove_same_contours(contours):
    contours = list(contours)
    pos_map = {}
    valid_contours = []
    for i, cnt in enumerate(contours):
        pos = get_contour_pos(cnt)
        if pos not in pos_map:
            pos_map[pos] = True
            valid_contours.append(cnt)

            
        print(i)    
    contours = tuple(valid_contours)
    return contours

def output_cnt_area(contours):
    output = ""
    for i,cnt in enumerate(contours):
        output += str(cv2.contourArea(cnt))+", "
    print(output)

class Shape:
    def __init__(self, x, y, w, h):
        self.x = x 
        self.y = y 
        self.w = w 
        self.h = h
    def is_valid_shape(self, img_shape):
        return self.x>0 and self.x<img_shape[1] and self.y>0 and self.y<img_shape[0] 

def get_bounding_box_of_contour(cnt,offset=0):
  # Only draw the the largest number of boxes

    # Use OpenCV boundingRect function to get the details of the contour
    x,y,w,h = cv2.boundingRect(cnt)
    shape = Shape(x-offset,y-offset, w+offset, h+offset)
    return True, shape

def crop_img_in_shape(img, shape, offset=0):
    
    cropped_img=img[shape.y-offset:shape.y+(shape.h+offset), shape.x-offset:shape.x+(shape.w+offset)]
    
    return cropped_img


img = cv2.imread("Output.png")

print(img.shape)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (0,0), fx=3.0, fy=3.0) 
EDGE_METHOD = 'canny'
FONT = cv2.FONT_HERSHEY_SIMPLEX  
ORG = (50, 50) 
FONTSCALE = 0.5
COLOR = (255, 0, 0)  
THICKNESS = 1
base_img = img.copy()


img = blur(img, 3)
img = apply_canny_filter(img,lower_threshold=0, higher_threshold=95)
cv2.imshow("img",img)
cv2.imwrite("Output_edges.png", img)
cv2.waitKey(0)


contours = get_contours(img)

print("Length of contours before:", len(contours))
contours = remove_same_contours(contours)
print("Length of contours after removal:", len(contours))
#contours = sorted(contours, key=cv2.contourArea, reverse=False)
merged_contours = group_contours(contours, threshold_distance=50)
print("Length of contours after cluster:", len(merged_contours))

hull = []

for i, cnt in enumerate(merged_contours):
    hull.append(cv2.convexHull(cnt, False))



for i, cnt in enumerate(hull):
    _,hull_shape = get_bounding_box_of_contour(cnt)
    cropped_img = crop_img_in_shape(img.copy(), hull_shape, offset=5)
    cv2.imwrite(f"cropped/{i}.png", cropped_img)


show_seperate_cnt(img, contours=hull)

black = np.zeros(img.shape, dtype = np.uint8)

cv2.drawContours(black, hull, -1, (255, 0, 0), 1) 
cv2.imwrite(f"final_contours.png",black)

'''
for i in range(1, 10,2):
    
    img = base_img.copy()

    blurred_img = blur(img, strength=i)
    
        
    if EDGE_METHOD == 'canny':
        low_threshold = 0
        high_threshold = 255
        
        THRESHOLD_STEP = 5
        while low_threshold <= high_threshold:
            high_thresh_temp = high_threshold
            while low_threshold<=high_thresh_temp:
                img = blurred_img.copy()
                img = apply_canny_filter(img, lower_threshold=low_threshold, higher_threshold=high_thresh_temp)
                contours = get_contours(img)
                black = np.zeros(img.shape, dtype = np.uint8)
                black = cv2.putText(black, str(len(contours)), ORG, FONT,FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
                cv2.drawContours(black, contours, -1, (255, 255, 0), 1) 
                print(f"{EDGE_METHOD}/{len(contours)}_{low_threshold}_{high_thresh_temp}_{i}.png")
                cv2.imwrite(f"{EDGE_METHOD}/{len(contours)}_{low_threshold}_{high_thresh_temp}_{i}.png", black)
                cv2.waitKey(0)
                high_thresh_temp-=THRESHOLD_STEP
            low_threshold+=THRESHOLD_STEP

    else :
        for j in range(1,10,2):
            img = blurred_img.copy()

            if EDGE_METHOD == 'sobel':
                img = apply_sobel_filter(img, kernel_size=j)
                img = np.array(img, np.uint8)
                print("Shape:",img.shape)
                print("Info", np.info(img))
                contours = get_contours(img)
            
            elif EDGE_METHOD == 'laplacian':
                img = apply_laplacian_filter(blurred_img, kernel_size=j)
                contours = get_contours(img)

            
            cv2.imshow("Img", img)
            cv2.waitKey(0)
            img = cv2.putText(img, str(len(contours)), ORG, FONT,  
                            FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA) 
            cv2.drawContours(img, contours, -1, (255, 255, 0), 1) 
            cv2.imwrite(f"{EDGE_METHOD}/kernel_{j}_blur_{i}.png", img)
    
'''


