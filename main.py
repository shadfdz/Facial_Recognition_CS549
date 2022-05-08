import cv2
import glob
import mediapipe as mp
import pandas as pd


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

def main():

    # get paths of video files
    vid_list = glob.glob('./dataset/*.mp4')
    f_name = vid_list[5]

    vid_cap = cv2.VideoCapture(f_name)
    if (vid_cap.isOpened() == False):
        raise Exception("Video not opened successfully")

    # get video properties
    width, height, length = get_vid_attributes(vid_cap)

    # create instance of face detection
    mp_face_detection = mp.solutions.face_detection

    # list to hold video file paths
    face_info_list = []

    # iterate through video frames
    for i in range(int(10)):
        # set frame time in video
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, (i + 1) * 1000)
        # get frame
        frame_exists, sample_frame = vid_cap.read()

        if not frame_exists:
            raise Exception("Empty video frame")

        # faces in frame
        with mp_face_detection.FaceDetection(
                min_detection_confidence=0.07, model_selection=1) as face_detection:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB))  # replace with frame

            # get detections for frame and export face
            for j,detection in enumerate(results.detections):
                x = int(detection.location_data.relative_bounding_box.xmin * width) # get x coordinate
                y = int(detection.location_data.relative_bounding_box.ymin * height) # get y coordinate
                x_width = int(detection.location_data.relative_bounding_box.width * width) # get width
                y_height = int(detection.location_data.relative_bounding_box.height * height) # get height
                cropped_img = sample_frame[y:y+y_height,x:x+x_width].copy() # crop image
                out_file_name = './output/test'+ str(j) + '.jpg' # image file name
                face_info_list.append([f_name,out_file_name,i]) # add to list for info
                cv2.imwrite(out_file_name, cropped_img) # write file name

        if i == 20: # place holder for testing
            break

    df = pd.DataFrame(data=face_info_list,columns=['vid_name','file_path','time']) # convert to df
    df.to_csv('./output/frame_info.csv') # save df as csv


if __name__ == "__main__":
    main()