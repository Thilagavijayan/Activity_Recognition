import numpy as np
import argparse
import imutils
import sys
import cv2

def recognize_activity(args):
    # Read the class labels from the file
    ACT = open(args["classes"]).read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112

    # Load the pre-trained model
    gp = cv2.dnn.readNet(args["model"])

    if args["gpu"] > 0:
        # Set the preferable backend and target to use GPU
        gp.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        gp.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Open the video stream
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    writer = None
    fps = vs.get(cv2.CAP_PROP_FPS)

    while True:
        frames = []  # Frames for processing
        originals = []  # Original frames

        for i in range(0, SAMPLE_DURATION):
            # Read frames from the video stream
            grabbed, frame = vs.read()

            if not grabbed:
                print("[INFO] No frame read from the stream - Exiting...")
                sys.exit(0)

            originals.append(frame)
            frame = imutils.resize(frame, width=100)
            frames.append(frame)

        # Prepare the input blob for the model
        blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
                                      swapRB=True, crop=True)
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis=0)

        # Set the input blob for the model and perform forward pass
        gp.setInput(blob)
        outputs = gp.forward()
        label = ACT[np.argmax(outputs)]  # Get the predicted activity label

        for frame in originals:
            # Draw the label on the frame
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if args["display"] > 0:
                # Display the frame with activity label
                cv2.imshow("Activity Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if args["output"] != "" and writer is None:
                # Create a video writer if output path is provided
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                new_width = int(frame.shape[1] * 0.5)
                new_height = int(frame.shape[0] * 0.5)
                writer = cv2.VideoWriter(args["output"], fourcc, fps, (new_width, new_height), True)

            if writer is not None:
                # Write the frame with activity label to the output video
                writer.write(frame)

    # Release resources
    vs.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Specify path to pre-trained model")
    ap.add_argument("-c", "--classes", required=True, help="Specify path to class labels file")
    ap.add_argument("-i", "--input", type=str, default="", help="Specify path to video file")
    ap.add_argument("-o", "--output", type=str, default="", help="Specify path to output video file")
    ap.add_argument("-d", "--display", type=int, default=1, help="Display output frame or not")
    ap.add_argument("-g", "--gpu", type=int, default=0, help="Whether or not to use GPU")
    args = vars(ap.parse_args())

    # Call the function to recognize activity
    recognize_activity(args)
