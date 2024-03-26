import cv2
import numpy as np
from Models.tracknet2 import trackNet

# Load the trackNet model
n_classes = 256
save_weights_path = 'WeightsTracknet/model.1'
width, height = 640, 360
m = trackNet(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# Function to predict the ball position
def predict_ball_position(frame):
    # Resize the frame to the input size of the model
    img = cv2.resize(frame, (width, height))
    img = img.astype(np.float32)

    # Change the axis order to 'channels_first'
    X = np.rollaxis(img, 2, 0)

    # Predict the heatmap
    pr = m.predict(np.array([X]))[0]

    # Reshape the output to (height, width, n_classes)
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # Convert to numpy.uint8
    pr = pr.astype(np.uint8)

    # Resize the heatmap to the original frame size
    heatmap = cv2.resize(pr, (frame.shape[1], frame.shape[0]))

    # Convert the heatmap to a binary image
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # Find the circles (ball positions) in the heatmap
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

    return circles

# Example usage
video = cv2.VideoCapture('517.mp4')
while True:
    ret, frame = video.read()
    if not ret:
        break

    circles = predict_ball_position(frame)

    # Draw the detected circles on the frame
    if circles is not None:
        for circle in circles[0]:
            x, y, r = circle
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()