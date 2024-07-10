import cv2 
import numpy as np

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

def get_contour_pos_centre(cnt):
    M = cv2.moments(cnt)
    cx=-1
    cy=-1
    if M['m00'] !=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) 
    return cx, cy

def get_contour_pos_left(cnt):
    _,bounds =  get_bounding_box_of_contour(cnt)
    M = cv2.moments(cnt)
    cx=-1
    cy=-1
    if M['m00'] !=0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00']) 
    return cx-(bounds.w//2), cy-(bounds.h//2)


def output_cnt_pos(contours):
    output =""
    for i, cnt in enumerate(contours):
        output += str(get_contour_pos_centre(cnt)) +", "
    print(output)