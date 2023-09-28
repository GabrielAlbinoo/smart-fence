import cv2
import numpy as np

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
colors = np.random.uniform(0, 255, size=(80, 3))

cap = cv2.VideoCapture(0)

FRAME_WIDTH, FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
LINE_COLOR = (0, 0, 255)
CONFIDENCE_THRESHOLD = 0.5
class_mapping = { 0: 'pessoa', 16: 'cachorro'}

while True:
    ret, frame = cap.read()
    cv2.line(frame, (0, FRAME_HEIGHT // 2), (FRAME_WIDTH, FRAME_HEIGHT // 2), LINE_COLOR, 2)

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(net.getUnconnectedOutLayersNames())

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD and class_id in class_mapping:
                center_x, center_y, w, h = map(int, detection[:4] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT]))
                x, y = center_x - w // 2, center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, 0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = f'{class_mapping[class_ids[i]]} ({confidences[i]:.2f})'
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y_center = y + h // 2

        position = "ACIMA" if y_center < FRAME_HEIGHT // 2 else "ABAIXO"
        print(f"'{label}' está {position} da linha. Confiança: {confidences[i]}")

    cv2.imshow("Smart Fence", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
