import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                1, (0, 0, 255), 1)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, 	((3,57,108)), rt)
        # Top Left
        cv2.line(img, (x, y), (x + l, y), (0,0,225), t)
        cv2.line(img, (x, y), (x, y + l), (0,0,225), t)
        # Top Right
        cv2.line(img, (x1, y), (x1 - l, y), (0,0,225), t)
        cv2.line(img, (x1, y), (x1, y + l), (0,0,225), t)
        # Bottom Left
        cv2.line(img, (x, y1), (x + l, y1), (0,0,225), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0,0,225), t)
        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - l, y1), (0,0,225), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0,0,225), t)

        return img


def main():
    # Use 0 for webcam, or your video path
    cap = cv2.VideoCapture(0)

    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        img, bboxs = detector.findFaces(img)
        # print(bboxs)  # Optional

        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}',
                    (5, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 1)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
