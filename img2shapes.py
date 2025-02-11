#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"Cant find image: {image_path}")
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)
    return thresh

def get_contours(thresh):
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours

def classify_contour(contour, threshold_lines=0.01, min_contour_area=700):
    area = cv2.contourArea(contour)
    if area < min_contour_area:
        return None, None, None

    epsilon = threshold_lines * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if not cv2.isContourConvex(approx):
        return None, None, None

    if len(approx) == 3:
        shape_name = "Triangle"
        color = (0, 255, 255)
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.2:
            shape_name = "Square"
            color = (0, 0, 255)
        else:
            shape_name = "Rectangle"
            color = (0, 165, 255)
    elif len(approx) > 6:
        shape_name = "Circle"
        color = (255, 0, 0)
    else:
        return None, None, None

    return approx, color, shape_name

def draw_shape(image, approx, color, label=None):
    cv2.drawContours(image, [approx], -1, color, thickness=cv2.FILLED)
    if label:
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(
                image, label, (cX - 20, cY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

def detect_shapes(image_path):
    image = load_image(image_path)
    thresh = preprocess_image(image)
    contours = get_contours(thresh)
    
    # Parameter
    threshold_lines = 0.01
    min_contour_area = 700

    for contour in contours:
        approx, color, shape_name = classify_contour(contour, threshold_lines, min_contour_area)
        if approx is not None:
            draw_shape(image, approx, color, label=shape_name)
    
    cv2.imshow("Detected Shapes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 img2shapes.py <image_path>")
        sys.exit(1)
    detect_shapes(sys.argv[1])

if __name__ == "__main__":
    main()