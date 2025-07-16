import cv2
import mediapipe as mp
import time
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 75))

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, faceLms, FACEMESH_TESSELATION, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 75), 1)
                    # print(id, x, y)
                    face.append([x, y])
        faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or file.")
        return  # Exit if camera cannot be opened

    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame or end of stream.")
            break

        # Convert image to grayscale for Canny (Canny typically works best on grayscale)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_img, 145, 155)  # Canny applied here

        # You can choose to display the original image with face mesh OR the Canny edges
        # For now, let's display the Canny edges
        display_img = edges

        # If you still want face mesh, you'd apply it to the original 'img'
        # and then decide how to combine 'img' and 'edges'
        # For example, to overlay edges on the original image:
        # edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Convert edges to 3 channels to overlay
        # display_img = cv2.addWeighted(img, 0.7, edges_color, 0.3, 0) # Blend original with edges

        img_with_facemesh, faces = detector.findFaceMesh(img.copy())  # Pass a copy if you modify 'img' later

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(display_img, f'FPS: {int(fps)}', (5, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)  # Text color changed for grayscale display

        cv2.imshow('Canny Edges', display_img)  # Changed window name

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()