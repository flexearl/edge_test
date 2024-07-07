import cv2
import numpy as np

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
        print("Cnt:", pos)
        print("Area:", cv2.contourArea(cnt))
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
        else:
            
            valid_contours.append(cnt)
        print(i)    
    contours = tuple(valid_contours)
    return contours

def output_cnt_area(contours):
    output = ""
    for i,cnt in enumerate(contours):
        output += str(cv2.contourArea(cnt))+", "
    print(output)


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
contours = get_contours(img)

print("Length of contours before:", len(contours))
contours = remove_same_contours(contours)
#contours = sorted(contours, key=cv2.contourArea, reverse=False)
print("Length of contours after:", len(contours))
output_cnt_area(contours)
show_seperate_cnt(img, contours=contours)



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


