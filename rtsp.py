import cv2
import threading
import time
import numpy as np

rtsp_urls = [
    "rtsp://admin:NNFVAJ@192.168.0.101:554/ch1/main",
    "rtsp://admin:NNFVAJ@192.168.0.101:554/ch1/main",
    "rtsp://admin:HLTHKD@192.168.0.102:554/ch1/main",
    "rtsp://admin:Edabk@408@192.168.0.125:554/ch1/main",
    "rtsp://admin:XLRPZQ@192.168.0.104:554/ch1/main",
    "rtsp://admin:DVCLRQ@192.168.0.105:554/ch1/main",
    "rtsp://admin:WSLRQC@192.168.0.106:554/ch1/main",
    "rtsp://admin:NXKPHU@192.168.0.119:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main",
    "rtsp://admin:JRGMMV@192.168.0.116:554/ch1/main",
    "rtsp://admin:LUXHLR@192.168.0.110:554/ch1/main",
    "rtsp://admin:TIJEQB@192.168.0.111:554/ch1/main",
    "rtsp://admin:XLRPZQ@192.168.0.104:554/ch1/main",
    "rtsp://admin:DVCLRQ@192.168.0.105:554/ch1/main",
    "rtsp://admin:WSLRQC@192.168.0.106:554/ch1/main",
    "rtsp://admin:NXKPHU@192.168.0.119:554/ch1/main",
    "rtsp://admin:WNRDVL@192.168.0.118:554/ch1/main",
    "rtsp://admin:JRGMMV@192.168.0.116:554/ch1/main",
    "rtsp://admin:LUXHLR@192.168.0.110:554/ch1/main",
    "rtsp://admin:TIJEQB@192.168.0.111:554/ch1/main"
]

W, H = 370, 270
ROWS, COLS = 4, 5

frames = [np.zeros((H, W, 3), dtype=np.uint8) for _ in rtsp_urls]
fps_texts = ["FPS: 0" for _ in rtsp_urls]

def read_camera(i, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return
    t0 = time.time()
    count = 0
    while True:
        ret, f = cap.read()
        if not ret:
            continue
        f = cv2.resize(f, (W, H))
        count += 1
        t1 = time.time()
        if t1 - t0 >= 1:
            fps_texts[i] = f"FPS: {count}"
            count = 0
            t0 = t1
        cv2.putText(f, fps_texts[i], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        frames[i] = f

for i, url in enumerate(rtsp_urls):
    threading.Thread(target=read_camera, args=(i, url), daemon=True).start()

while True:
    rows_img = []
    for i in range(ROWS):
        row = frames[i*COLS:(i+1)*COLS]
        rows_img.append(cv2.hconcat(row))
    grid = cv2.vconcat(rows_img)
    cv2.imshow("Cameras", grid)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()