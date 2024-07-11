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





def create_hierarchy_map(sorted_x_contours):
    hierarchy_map = {}
    for i, cnt in enumerate(sorted_x_contours):
        pos = utils.get_contour_pos_left(cnt)
        idx = find_cnt_idx_in_arr(sorted_x_contours, pos[0], pos[1],threshold=0)
        if idx != 0:
            _,bounds = utils.get_bounding_box_of_contour(cnt)
            pos  = utils.get_contour_pos_left(sorted_x_contours[idx])
            while pos[0]<bounds.x + bounds.w and idx<len(sorted_x_contours):
                if bounds.y<pos[1] and pos[1] < bounds.y+bounds.h:
                
                    if i not in hierarchy_map and i !=idx:

                        hierarchy_map[i] = [idx]
                    elif i in hierarchy_map and idx not in hierarchy_map[i] and i!=idx:

                        hierarchy_map[i].append(idx)
                idx+=1
                if idx != len(sorted_x_contours):
                    pos = utils.get_contour_pos_left(sorted_x_contours[idx])

    return hierarchy_map
        
        