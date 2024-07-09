import cv2
import easyocr as eo
import time



start_time = time.time()
img = cv2.imread('images/hubspot_top.png')

reader = eo.Reader(['en'], gpu=True)

text = reader.readtext(img)

for t in text:
    print(t)
    bbox, text, score = t
    l_bbox = bbox[0][0]
    l_bbox1 = bbox[0][1]
    r_bbox = bbox[2][0]-2
    r_bbox1 = bbox[2][1]-2

    cv2.rectangle(img, (int(l_bbox), int(l_bbox1)), (int(r_bbox), int(r_bbox1)), (0, 255, 0),-1)
    #cv2.putText(img, text, (int(l_bbox), int(l_bbox1)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 ,0), 2)

print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow("Output", img)
cv2.imwrite("Output.png", img)
cv2.waitKey(0)