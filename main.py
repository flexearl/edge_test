import cv2

def apply_sobel_filter(img, kernel_size=5):
    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    return sobelxy

def apply_laplacian_filter(img, kernel_size=5):
    img = cv2.Laplacian(img, ksize=kernel_size, ddepth=cv2.CV_16S)
    img = cv2.convertScaleAbs(img)
    return img

def apply_canny_filter(img, lower_threshold=100, higher_threshold=200):
    edges = cv2.Canny(image=img.copy(), threshold1=lower_threshold, threshold2=higher_threshold)
    return edges

def blur(img, strength=3):
    img_blur = cv2.GaussianBlur(img, (strength,strength), 0)
    return img_blur

def get_contours(img):
    contours, _ = cv2.findContours(img,  
    cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours 

img = cv2.imread("images/phone_icon.png")
img = cv2.resize(img, (0,0), fx=4.0, fy=4.0) 

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

EDGE_METHOD = 'canny'
FONT = cv2.FONT_HERSHEY_SIMPLEX  
ORG = (50, 50) 
FONTSCALE = 0.5
COLOR = (255, 0, 0)  
THICKNESS = 1
base_img = img_gray.copy()


for i in range(1, 10,2):
    
    img = base_img.copy()

    blurred_img = blur(img, strength=i)
    if EDGE_METHOD == 'canny':
        low_threshold = 0
        high_threshold = 255
        while low_threshold <= high_threshold:
            img = blurred_img.copy()
            img = apply_canny_filter(img, lower_threshold=low_threshold, higher_threshold=high_threshold)
            contours = get_contours(img)
            img = cv2.putText(img, str(len(contours)), ORG, FONT,FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA)
            cv2.drawContours(img, contours, -1, (255, 255, 0), 1) 
            low_threshold+=1
            
            print(f"{EDGE_METHOD}/low:_{low_threshold}_high:_{high_threshold}_blur:_{i}_cnt:{len(contours)}")
            cv2.imwrite(f"{EDGE_METHOD}/low_{low_threshold}_high_{high_threshold}_blur_{i}.png", img)
            cv2.waitKey(0)

    else :
        for j in range(1,10,2):
            if EDGE_METHOD == 'sobel':
                img = apply_sobel_filter(img_gray,kernel_size=j)
                contours = get_contours(img)
            elif EDGE_METHOD == 'laplacian':
                img = apply_laplacian_filter(img_gray, kernel_size=j)
                contours = get_contours(img)

            
            
            img = cv2.putText(img, str(len(contours)), ORG, FONT,  
                            FONTSCALE, COLOR, THICKNESS, cv2.LINE_AA) 
            cv2.drawContours(img, contours, -1, (0, 255, 0), 3) 
            cv2.imwrite(f"{EDGE_METHOD}/kernel:_{j}_blur:_{i}.png", img)
        



cv2.imshow("First img",img)
cv2.waitKey(0)