import cv2
import numpy as np 
import utils

def find_cnt_idx_in_arr(contours, x,y, threshold = 2):
    l = 0
    r = len(contours)-1
    while l<=r:
        m = (l+r)//2
        cnt_pos = utils.get_contour_pos_left(contours[m])
        if cnt_pos[0] < x -threshold:
            l = m +1
        elif cnt_pos[0]>x+threshold:
            r = m-1
        else:

            return m

    return -1





def create_hierarchy_map(sorted_x_contours, img):
    print("Creating map")
    hierarchy_map = {}
    utils.output_cnt_pos(sorted_x_contours)
    
    for i, cnt in enumerate(sorted_x_contours):
        print("i:", i)
        pos = utils.get_contour_pos_left(cnt)
        og_pos = pos
        print("Looking for:", pos)
        idx = find_cnt_idx_in_arr(sorted_x_contours, pos[0], pos[1],threshold=0)
        if idx != 0:
            _,bounds = utils.get_bounding_box_of_contour(cnt)
            pos  = utils.get_contour_pos_left(sorted_x_contours[idx])
            print("Received:", pos)
            while pos[0]<bounds.x + bounds.w and idx<len(sorted_x_contours):
                if bounds.y<pos[1] and pos[1] < bounds.y+bounds.h:
                    print(f"Og pos: {og_pos} for {i}" )
                    print(f"Point {pos} inbetween {bounds.x}-{bounds.x+bounds.w} and inbetween {bounds.y}-{bounds.y+bounds.h}: {idx}")
                    if i not in hierarchy_map and i !=idx:
                        img_copy = img.copy()
                        cv2.drawContours(img_copy, (cnt, sorted_x_contours[idx]), -1, (0,0,255), 2)
                        cv2.imwrite(f"inside/{i}.png", img_copy) 
                    
                        print("Adding")
                        hierarchy_map[i] = [idx]
                    elif i in hierarchy_map and idx not in hierarchy_map[i] and i!=idx:
                        print("Adding")
                        hierarchy_map[i].append(idx)
                idx+=1
                if idx != len(sorted_x_contours):
                    pos = utils.get_contour_pos_left(sorted_x_contours[idx])
                


    return hierarchy_map
        
        