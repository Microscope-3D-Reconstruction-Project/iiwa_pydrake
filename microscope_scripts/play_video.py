import argparse
import os
import time

from datetime import datetime

import cv2


def play_video_stream(source=0):
    """
    Play a real-time video stream using OpenCV.

    Args:
        source: Video source (0 for default webcam, or path to video file)
    """
    # Open the video capture
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    # Get video properties
    cap.set(cv2.CAP_PROP_FPS, 30)  # Attempt to set FPS to 30 if supported
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Video stream opened: {width}x{height} @ {fps} FPS")
    print("Press 'q' to quit, 's' to save a snapshot, or click the button")

    snapshot_count = 0
    prev_time = time.time()

    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Button display properties (positioned under FPS)
    button_x, button_y = 10, 60
    button_width, button_height = 150, 40

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame or stream ended")
            break

        # Calculate real-time FPS
        current_time = time.time()
        realtime_fps = (
            1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        )
        prev_time = current_time

        # Add FPS text overlay to the frame
        fps_text = f"FPS: {realtime_fps:.2f}"
        cv2.putText(
            frame,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw screenshot button indicator (press 's' to use)
        cv2.rectangle(
            frame,
            (button_x, button_y),
            (button_x + button_width, button_y + button_height),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            frame,
            "Press 'S'",
            (button_x + 20, button_y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the frame
        cv2.imshow("Video Stream", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Exiting...")
            break
        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = os.path.join(outputs_dir, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_filename, frame)
            print(f"Screenshot saved: {snapshot_filename}")
            snapshot_count += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Play real-time video stream using OpenCV"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="4",
        help="Video source: 0 for default webcam, 1 for second camera, or path to video file",
    )

    args = parser.parse_args()

    # Convert source to int if it's a digit, otherwise keep as string (file path)
    source = int(args.source) if args.source.isdigit() else args.source

    play_video_stream(source)


if __name__ == "__main__":
    main()
