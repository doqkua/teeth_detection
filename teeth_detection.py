from ultralytics import YOLO, hub 
import cv2 

hub.login('a71906bb98e5bcbb2053baf5ea6d9b40bef22775f1')

model = YOLO('https://hub.ultralytics.com/models/XPJXCUZ5N3989s1Z86J2')

image_path = 'DENTISTRY/tooth_detection_and_numbering/teeth_xray4.jpg'


img = cv2.imread(image_path)
img_resized = cv2.resize(img, (1280, 720))
cv2.imwrite('xray_teeth_detection.jpg', img_resized)
results = model.predict('xray_teeth_detection.jpg', conf=0.01)[0]

detections = []

for box in results.boxes: 
    conf = float(box.conf[0])
    if conf < 0.01: 
        continue 

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    x_center = (x1 + x2) // 2 
    label = results.names[cls]
    detections.append((x_center, (x1, y1, x2, y2), label, conf))

detections.sort(key=lambda x:x[0])

for i, (_, (x1, y1, x2, y2), label, conf) in enumerate(detections): 
    cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
    text = f"Tooth {label} ({conf:.2f})"
    cv2.putText(img_resized, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2)
    
cv2.imshow("Detected Teeth", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()