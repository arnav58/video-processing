import numpy as np
import cv2
import time
import argparse


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("video_file_path")
    # parser.add_argument("monochrome")
    # parser.add_argument("fps")
    # parser.add_argument("width")
    # parser.add_argument("height")
    #
    # args = parser.parse_args()
    #
    # video_file_path = args.video_file_path
    # monochrome = True if args.monochrome == "True" else False
    # fps = int(args.fps)
    # width = int(args.width)
    # height = int(args.height)

    video_file_path = 'data/video_1.mp4'
    monochrome = False
    fps = 10
    width = 400
    height = 240
    output_path = 'results/segmented_video.mp4'

    cap = cv2.VideoCapture(video_file_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        start = time.time()

        ret, frame = cap.read()

        if ret:
            if monochrome:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('monochrome', gray)
            else:
                cv2.imshow('original', frame)

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            low = np.array([0, 40, 0])
            high = np.array([179, 255, 255])
            mask = cv2.inRange(hsv_frame, low, high)

            result = cv2.bitwise_and(frame, frame, mask=mask)
            out.write(result)
            cv2.imshow('segmented', result)
        else:
            break

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('p'):
            while True:
                next_key = cv2.waitKey(0)

                if next_key & 0xFF == ord('p'):
                    break

                if next_key & 0xFF == ord('b'):
                    current_frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    prev_frame_no = current_frame_no

                    if current_frame_no > 1:
                        prev_frame_no -= 1

                    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_no)

        timeDiff = time.time() - start
        if timeDiff < 1.0 / fps:
            time.sleep(1.0 / fps - timeDiff)


    out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
