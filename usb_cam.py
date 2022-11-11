import cv2
cap = cv2.VideoCapture(1)

print("?")
while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow("USB", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):  # 按下q(quit)键，程序退出
        break
cap.release()
cv2.destroyAllWindows()