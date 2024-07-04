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

def apply_canny_filter(img, kernel_size=5, lower_threshold=100, higher_threshold=200):
    edges = cv2.Canny(image=img.copy(), threshold1=lower_threshold, threshold2=higher_threshold)
    return img

def blur(img, strength=3):
    img_blur = cv2.GaussianBlur(img, (strength,strength), 0)
    return img_blur

img = cv2.imread("images/phone_icon.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

EDGE_METHOD = 'canny'

for i in range(1, 10):
    blurred_img = blur(img, strength=i)
    if EDGE_METHOD == 'canny':
        low_threshold = 0
        high_threshold = 255
    else :
        for j in range(1,10):
            if EDGE_METHOD == 'sobel':
                
            elif EDGE_METHOD == 'laplacian':
        



cv2.imshow("First img",img)
cv2.waitKey(0)