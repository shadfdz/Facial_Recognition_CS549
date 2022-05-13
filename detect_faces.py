import cv2
import mediapipe as mp


def get_vid_attributes(vid_cap):
    """
    Get video attributes
    Args:
        vid_cap: instance of cv2 Vid Capture
    Returns: width, height, length in seconds
    """
    width = vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    length = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)/vid_cap.get(cv2.CAP_PROP_FPS)

    return width, height, length


def get_face_image_list(video_file_path, detect_param=0.5):

    vid_cap = cv2.VideoCapture(video_file_path)
    if (vid_cap.isOpened() == False):
        raise Exception("Video not opened successfully")

    # get video properties
    width, height, length = get_vid_attributes(vid_cap)

    # list to hold video file paths
    faces_persecond_list = []

    for i in range(int(length)):
        # set frame time in video
        face_list = []
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, (i + 1) * 1000)
        # get frame
        frame_exists, sample_frame = vid_cap.read()

        if not frame_exists:
            raise Exception("Empty video frame")

        # create instance of face detection
        mp_face_detection = mp.solutions.face_detection

        # faces in frame
        with mp_face_detection.FaceDetection(
                min_detection_confidence=detect_param, model_selection=1) as face_detection:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))  # replace with frame

            if results.detections is not None:
                for detection in results.detections:
                    x = int(detection.location_data.relative_bounding_box.xmin * width) # get x coordinate
                    y = int(detection.location_data.relative_bounding_box.ymin * height) # get y coordinate
                    x_width = int(detection.location_data.relative_bounding_box.width * width) # get width
                    y_height = int(detection.location_data.relative_bounding_box.height * height) # get height
                    cropped_img = sample_frame[y:y+y_height,x:x+x_width].copy() # crop image
                    face_list.append(cropped_img)
                faces_persecond_list.append(face_list)

    return faces_persecond_list
