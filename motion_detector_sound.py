import cv2
import pygame

files = "motion_detected_v2.mp3"
first_frame = None
second_frame = None
status_list = [None, None]
playing = False

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(files)


video = cv2.VideoCapture(0)
check, frame = video.read()

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
        
    if second_frame is None:
        if first_frame is None:
            first_frame = gray
            continue
        second_frame = gray
        continue

    delta_frame = cv2.absdiff(second_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue
        status = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    #cv2.imshow("Gray frame", gray)
    #cv2.imshow("Delta frame", delta_frame)
    cv2.imshow("Threshold frame", thresh_frame)
    cv2.imshow("Color frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    
    if status == 1:
        if playing == False:
            pygame.mixer.music.play(loops = -1)
            playing = True       
    elif status == 0:
        if playing == True:
            pygame.mixer.music.stop()
            playing = False

video.release()
cv2.destroyAllWindows