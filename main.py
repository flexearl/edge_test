import cv2
import numpy as np
import math
import utils
import hierarchy

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
            current_cnt = merge_contours(current_cnt, contours[keys[i]])
            del matches[keys[i]]
        elif keys[i] in matches:
            
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
            matches[i] = []

    merged_contours = []
    
    
    for key in matches.copy():
       
        if key in matches:
        
            arr = matches[key]
            
            if len(arr) != 0:
                merged = merge_matches(contours, contours[key], matches, arr)
                merged_contours.append(merged)
                
            else:
                merged_contours.append(contours[key])
       
    return tuple(merged_contours)

def show_seperate_cnt(img, contours):
    cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE )
   
    for i, cnt in enumerate(contours):
        black = np.zeros(img.shape, dtype = np.uint8)
       
       
        cv2.drawContours(black, [cnt], -1, (255, 0, 0), 1) 
        cv2.imwrite(f"seperate_cnt/{i}.png",black)
       



def remove_same_contours(contours):
    contours = list(contours)
    pos_map = {}
    valid_contours = []
    for i, cnt in enumerate(contours):
        pos = utils.get_contour_pos_centre(cnt)
        if pos not in pos_map:
            pos_map[pos] = True
            valid_contours.append(cnt)

            
          
    contours = tuple(valid_contours)
    return contours

def output_cnt_area(contours):
    output = ""
    for i,cnt in enumerate(contours):
        output += str(cv2.contourArea(cnt))+", "
    print(output)



def remove_lower_area_cnt(sorted_contours, lowest_area):
    sorted_contours = list(sorted_contours)
    for i, cnt in enumerate(sorted_contours):
        area = cv2.contourArea(cnt)
        if area>lowest_area:
            return tuple(sorted_contours[i:])


img = cv2.imread("Output.png")
img = cv2.resize(img, (0,0), fx=3.0, fy=3.0) 
original_img = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

EDGE_METHOD = 'canny'
FONT = cv2.FONT_HERSHEY_SIMPLEX  
ORG = (50, 50) 
FONTSCALE = 0.5
COLOR = (255, 0, 0)  
THICKNESS = 1
base_img = img.copy()




cv2.imshow("img",img)
cv2.imwrite("Output_edges.png", img)
cv2.waitKey(0)


contours = utils.get_contours(img)

print("Length of contours before:", len(contours))
contours = remove_same_contours(contours)
print("Length of contours after removal:", len(contours))

#contours = sorted(contours, key=cv2.contourArea, reverse=False)
merged_contours = group_contours(contours, threshold_distance=30)
print("Length of contours after cluster:", len(merged_contours))
merged_contours = sorted(merged_contours, key=cv2.contourArea, reverse=False)
merged_contours = remove_lower_area_cnt(merged_contours, 5)
output_cnt_area(merged_contours)

hull = []

for i, cnt in enumerate(merged_contours):
    hull.append(cv2.convexHull(cnt, False))


black = np.zeros(img.shape, dtype = np.uint8)


sorted_x_contours = sorted(hull, key =lambda x: utils.get_contour_pos_left(x)[0])
hierarchy_map = hierarchy.create_hierarchy_map(sorted_x_contours, img=original_img)


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


