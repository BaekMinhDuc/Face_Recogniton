import cv2

rtsp_url = "rtsp://192.168.1.100:554/user:hanet;pwd:hanet123"

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No")
        break
    cv2.imshow("Hanet", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
