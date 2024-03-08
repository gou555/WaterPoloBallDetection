import torch
import cv2
import numpy as np

# Load the trained model
m_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Using camera 1 (USB camera)
m_camera = cv2.VideoCapture(1)


m_frameRate = m_camera.get(cv2.CAP_PROP_FPS)
m_timePerFrame = 1.0 / m_frameRate

m_tracker = None

m_trackerActive = False

m_lastPosition = None

m_frameCount = 0

m_staticThreshold = 20

m_speed = 0

m_scaleFactor = 0.005  # 0.005 meters per pixel

try:
    while True:
        ret, frame = m_camera.read()
        if not ret:
            break
        
        if m_trackerActive:
            success, bbox = m_tracker.update(frame)
            if success:
                # Calculate the current position as the center of the bounding box
                curPosition = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                if m_lastPosition:
                    # Calculate movement by comparing positions
                    # [0] is the x coordinate and [1] is the y position
                    dx = curPosition[0] - m_lastPosition[0]
                    dy = curPosition[1] - m_lastPosition[1]
                    # Distance
                    movement = np.sqrt(dx**2 + dy**2)
                    # Calculate speed: distance/time
                    m_speed = movement / m_timePerFrame
                    if movement < 2:
                        m_frameCount += 1
                    else:
                        m_frameCount = 0 
                m_lastPosition = curPosition
                speedMeterPerSecond = m_speed * m_scaleFactor

                speedText = f"Speed: {speedMeterPerSecond:.2f} m/s"
                cv2.putText(frame, speedText, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Draw bounding box on the frame
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
                
                # If we don't move for 15frames we reset the detection (check the model again)
                if m_frameCount >= m_staticThreshold:
                    m_trackerActive = False
                    m_frameCount = 0 
            else:
                m_trackerActive = False
                m_lastPosition = None  
        else:
            results = m_model(frame)
            for *xyxy, conf, _ in results.xyxy[0]: # _ cause we only have 1class in our model (ball)
                if conf > 0.2:  
                    x1, y1, x2, y2 = map(int, xyxy)
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    
                    m_tracker = cv2.TrackerCSRT_create()
                    m_tracker.init(frame, bbox)
                    m_trackerActive = True
                    m_lastPosition = None  
                    break  

        
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    m_camera.release()
    cv2.destroyAllWindows()
