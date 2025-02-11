#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##########################################
## Laxfarm Playground - Facerecognition ##
##########################################
import cv2
import numpy as np
import argparse
import time

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    beta = brightness - 50
    alpha = contrast / 50.0
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def detect_shapes(frame, iterations, bw_mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if bw_mode:
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    
    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if cv2.isContourConvex(approx):
            detected_shapes.append(approx)
    
    return frame, detected_shapes

def update_filled_shapes(detected_shapes, filled_shapes, current_time, fill_duration=5.0):
    for shape in detected_shapes:
        found = False
        for fs in filled_shapes:
            match_val = cv2.matchShapes(shape, fs["contour"], cv2.CONTOURS_MATCH_I1, 0.0)
            if match_val < 0.1:
                fs["fill_until"] = current_time + fill_duration
                found = True
                break
        if not found:
            filled_shapes.append({
                "contour": shape,
                "fill_until": current_time + fill_duration
            })
    filled_shapes = [fs for fs in filled_shapes if fs["fill_until"] > current_time]
    return filled_shapes

def draw_filled_shapes(frame, filled_shapes, fill_color=(0, 0, 255)):
    for fs in filled_shapes:
        cv2.drawContours(frame, [fs["contour"]], -1, fill_color, thickness=cv2.FILLED)

def detect_faces_and_hands(frame, face_cascade, hand_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    hands = hand_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces, hands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera-Index")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.camera)
    cv2.namedWindow("Shape Tracking")
    cv2.createTrackbar("Brightness", "Shape Tracking", 50, 100, lambda x: None)
    cv2.createTrackbar("Contrast", "Shape Tracking", 50, 100, lambda x: None)
    cv2.createTrackbar("Iterations", "Shape Tracking", 2, 10, lambda x: None)
    cv2.createTrackbar("BW Mode", "Shape Tracking", 0, 1, lambda x: None)
    
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    hand_cascade = cv2.CascadeClassifier('./cascades/haarcascade_hand.xml')
    
    if face_cascade.empty():
        print("Error: couldnt find face cascade")
        return
    if hand_cascade.empty():
        print("Warning: couldnt find hand cascade, deactivate hand and finger detection")
    
    trackers = []
    filled_shapes = []
    fill_duration = 5.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        brightness = cv2.getTrackbarPos("Brightness", "Shape Tracking")
        contrast = cv2.getTrackbarPos("Contrast", "Shape Tracking")
        iterations = cv2.getTrackbarPos("Iterations", "Shape Tracking")
        bw_mode = cv2.getTrackbarPos("BW Mode", "Shape Tracking")
        
        frame = adjust_brightness_contrast(frame, brightness, contrast)
        processed_frame, detected_shapes = detect_shapes(frame, iterations, bw_mode)
        
        current_time = time.time()
        filled_shapes = update_filled_shapes(detected_shapes, filled_shapes, current_time, fill_duration)
        draw_filled_shapes(processed_frame, filled_shapes, fill_color=(0, 0, 255))
        
        if len(trackers) == 0:
            for shape in detected_shapes:
                tracker = cv2.TrackerCSRT_create()
                x, y, w, h = cv2.boundingRect(shape)
                tracker.init(frame, (x, y, w, h))
                trackers.append(tracker)
        
        for tracker in trackers:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(processed_frame, "Tracked", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        faces, hands = detect_faces_and_hands(frame, face_cascade, hand_cascade)
        for (x, y, w, h) in faces:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(processed_frame, "Face", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if not hand_cascade.empty():
            for (x, y, w, h) in hands:
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(processed_frame, "Hand", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Shape Tracking", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()