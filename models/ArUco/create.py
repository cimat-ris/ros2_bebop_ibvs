import cv2
import numpy as np
import argparse

# # resolution = 1890
# frame_width = 8
# margin = 60  # space between marker and frame

markers_list = ["4X4_50" ,
        "4X4_100" ,
        "4X4_250" ,
        "4X4_1000" ,
        "5X5_50" ,
        "5X5_100" ,
        "5X5_250" ,
        "5X5_1000" ,
        "6X6_50" ,
        "6X6_100" ,
        "6X6_250" ,
        "6X6_1000" ,
        "7X7_50" ,
        "7X7_100" ,
        "7X7_250" ,
        "7X7_1000" ,
        "ARUCO_ORIGINAL" ,
        "APRILTAG_16h5" ,
        "APRILTAG_25h9" ,
        "APRILTAG_36h10" ,
        "APRILTAG_36h11" ,
        "ARUCO_MIP_36h12"]

def main(arg):

    markers = markers_list.index(arg.dict)
    family = cv2.aruco.getPredefinedDictionary(markers)
    marker_size = arg.size - 2 * arg.margin

    for marker_id in range(arg.start, arg.end):  # includes m
        marker = cv2.aruco.generateImageMarker(family, marker_id, marker_size)

        # Black canvas makes the white frame visible.
        image = np.zeros((arg.size, arg.size), dtype=np.uint8)

        x = arg.margin
        y = arg.margin
        image[y:y + marker_size, x:x + marker_size] = marker

        # Draw the white frame around the marker.
        cv2.rectangle(
            image,
            (0, 0),
            (arg.size, arg.size),
            255,
            thickness=arg.margin,
        )

        cv2.imwrite(f"{arg.dict}-{marker_id}.png", image)

if __name__ ==  "__main__":
    description = "creating arucos as texture"
    parser = argparse.ArgumentParser(prog = 'python3 create.py',
                                     description = description)
    parser.add_argument( '--start', type=int,
        default = 0,
        help = "Start index (default 0)")
    parser.add_argument( '--end', type=int,
        default = 1,
        help = "Start index (default 1)")
    parser.add_argument( '--dict', type=str,
        default = '4X4_1000',
        help = "Marker dictionary (default 4X4_1000)")
    parser.add_argument( '--size', type=float,
        default = 1890,
        help = "Image edge size (default 1890)")
    parser.add_argument( '--margin', type=float,
        default = 60,
        help = "Image white margin width (default 60)")

    arg = parser.parse_args()
    main(arg)

