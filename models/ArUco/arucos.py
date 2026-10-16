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

def create(arg):

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

sdf_template = """<model name="ArUco_4">
    <static>true</static>
    <pose> -1 1 0.14 0 0 0.2 </pose>
    <link name="link">
      <collision name="collision">
        <geometry>
        <plane>
          <normal>0 0 1</normal>
          <size>0.5 0.5</size>
        </plane>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>100</mu>
              <mu2>50</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>0.5 0.5</size>
          </plane>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
          <pbr>
          <metal>
            <albedo_map>model://ArUco/4X4_1000-99.png</albedo_map>
          </metal>
        </pbr>
        </material>
      </visual>
    </link>
  </model>
"""
def sdf(arg):

    r = 3 # resolution

    lines = sdf_template.splitlines()

    dx = arg.range[1] - arg.range[0]
    dy = arg.range[3] - arg.range[2]
    dz = arg.range[5] - arg.range[4]
    dw = arg.range[7] - arg.range[6]

    # Create coordinates

    n =  arg.end - arg.start
    x = arg.range[0] + dx * np.random.rand(n)
    y = arg.range[2] + dy * np.random.rand(n)
    z = arg.range[4] + dz * np.random.rand(n)
    w = arg.range[6] + dw * np.random.rand(n)

    # _l = [i for i  in ]
    for i, k in enumerate(range(arg.start, arg.end)):
        lines[0] = f"  <model name=\"ArUco_{k}\">"
        lines[2] = f"    <pose> {x[i]:.{r}} {y[i]:.{r}} {z[i]:.{r}} 0 0 {w[i]:.{r}} </pose>"
        lines[34] = f"           <albedo_map>model://ArUco/{arg.dict}-{k}.png</albedo_map>"
        print("\n".join(lines))
        print()
        print()


if __name__ ==  "__main__":
    description = "ArUcos utilities"
    parser = argparse.ArgumentParser(prog = 'python3 arucos.py',
                                     description = description)
    subparsers = parser.add_subparsers(title="Commands", required=True)


    create_parser = subparsers.add_parser('create', help='Create ArUco images')
    create_parser.add_argument( '--start', type=int,
        default = 0,
        help = "Start index (default 0)")
    create_parser.add_argument( '--end', type=int,
        default = 1,
        help = "Start index (default 1)")
    create_parser.add_argument( '--dict', type=str,
        default = '4X4_1000',
        help = "Marker dictionary (default 4X4_1000)")
    create_parser.add_argument( '--size', type=float,
        default = 1890,
        help = "Image edge size (default 1890)")
    create_parser.add_argument( '--margin', type=float,
        default = 60,
        help = "Image white margin width (default 60)")
    create_parser.set_defaults(func = create)

    sdf_parser = subparsers.add_parser('sdf', help='Create ArUco sdf code with random placing')
    sdf_parser.add_argument( '--start', type=int,
        default = 0,
        help = "Start index (default 0)")
    sdf_parser.add_argument( '--end', type=int,
        default = 1,
        help = "Start index (default 1)")
    sdf_parser.add_argument( '--dict', type=str,
        default = '4X4_1000',
        help = "Marker dictionary (default 4X4_1000)")
    sdf_parser.add_argument( '--range', type=float, nargs=8,
        default = [0, 0, 0, 0, 0.1, 0.1, 0, 0],
        help = "Random intervals for X Y Z W, up and down for each (default 0's at Z = 0.1)")
    sdf_parser.set_defaults(func = sdf)

    arg = parser.parse_args()
    arg.func(arg)
    # main(arg)

