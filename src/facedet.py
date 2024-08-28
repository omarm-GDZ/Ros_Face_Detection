#!/usr/bin/env python3
import cv2
import cv_bridge
import rospy
from std_msgs.msg import Int32
def callback(x):
    rospy.loginfo(x)

if __name__ == "__main__":

    # Load the pre-trained face detection model from the specified XML file
    # The file contains the Haar Cascade classifier for detecting frontal faces
  #  face_cascade = cv2.CascadeClassifier('homepirpi_labhaarhaarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Open a connection to the camera (default camera index 0)
    cap = cv2.VideoCapture(0)

    # Initialize the ROS node named 'face_detector_node'
    rospy.init_node('face_detector_node')

    # Create a ROS publisher to send face detection status messages
    # Publishing to the 'face_detectorstatus' topic with message type Int32
    pub = rospy.Publisher('face_detectorstatus', Int32, queue_size=10)

    sub=rospy.Subscriber('face_detectorstatus',Int32,callback)
    while cap.isOpened():
        # Capture a single frame from the video feed
        ret, frame = cap.read()

        # Initialize the face detection flag for the current frame
        face_detected_in_this_frame = False

        if ret:
            # Convert the captured frame to grayscale
            # Face detection works better on grayscale images
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Parameters for the face detection algorithm
            scale_factor = 1.3  # Factor by which the image size is reduced at each image scale
            min_neighbours = 5  # Minimum number of neighbors each candidate rectangle should have to be retained

            # Perform face detection on the grayscale image
            faces = face_cascade.detectMultiScale(image_gray, scale_factor, min_neighbours)

            # Iterate through the detected faces
            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                # (x, y) is the top-left corner, (x+w, y+h) is the bottom-right corner
                cv2.rectangle(image_gray, (x, y), (x+w, y+h), (255, 255, 255), 2)

                # Set the face detection flag to True as a face is detected in this frame
                face_detected_in_this_frame = True

            # Publish the face detection status to the ROS topic
            # 1 indicates a face was detected, 0 indicates no face was detected
            pub.publish(Int32(1) if face_detected_in_this_frame else Int32(0))

            # Display the frame with detected faces highlighted
            cv2.imshow('frame', image_gray)

            # Wait for a key event for 1 millisecond
            # If the ESC key (key code 27) is pressed, exit the loop
            c = cv2.waitKey(1)
            if c == 27:
                # Close all OpenCV windows and exit the loop
                cv2.destroyAllWindows()
                break
