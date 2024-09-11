import numpy as np
import cv2

def capture_images(cap, num_images):
    images = []
    while len(images) < num_images:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 좌우 대칭

        # 메시지 추가
        cv2.putText(frame, "Press 's' to capture, 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # 프레임을 윈도우에 표시
        cv2.imshow('Camera', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # 's' 키를 누르면 사진 저장
            if len(images) < num_images:
                images.append(frame)
                print(f"Photo {len(images)}/{num_images} saved")
        elif key == ord('q'):  # 'q' 키를 누르면 종료
            break

    return images

def create_collage(images):
    if len(images) != 4:
        raise ValueError("사진을 전부 촬영해주세요")
    
    # 이미지 크기 조정하기 
    images_resized = [cv2.resize(img, (640, 480)) for img in images]

    # Combine images into a 2x2 grid
    top_row = np.hstack(images_resized[0:2])
    bottom_row = np.hstack(images_resized[2:4])
    collage = np.vstack([top_row, bottom_row])

    return collage

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while True:
    print("Capture 4 photos. Press 's' to capture and 'q' to quit.")
    images = capture_images(cap, num_images=4)

    if len(images) == 4:
        # Create and save collage
        collage = create_collage(images)
        cv2.imwrite('photo_collage.jpg', collage)
        print("저장되었습니다.")

        # Display the collage
        cv2.imshow('Collage', collage)
        print("Press 'r' to retake photos or 'q' to quit.")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):  # 'r' 키를 누르면 다시 촬영
                break
            elif key == ord('q'):  # 'q' 키를 누르면 종료
                cap.release()
                cv2.destroyAllWindows()
                exit()

# Release resources
cap.release()
cv2.destroyAllWindows()
