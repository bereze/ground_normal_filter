import numpy as np
import cv2
from os.path import join, isdir
from os import listdir, mkdir
from tqdm import tqdm
from filter import GroundNormalFilterIEKF
from visualizer import Visualization
import argparse
from scipy.spatial.transform import Rotation as R


def invT(transform):
    """inverse a transform matrix without using np.linalg.inv

    Args:
        transform (ndarray): input transform matrix with shape=(4,4)

    Returns:
        ndarray: output transform matrix with shape=(4,4)
    """
    R_Transposed = transform[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = R_Transposed
    result[:3, 3] = -R_Transposed @ transform[:3, 3]
    return result


def read_kitti_calib(calib_path):
    """Read kitti calibration file and get camera intrinsic matrix

    Args:
        calib_path (string): path to calibration file (xxx/calib.txt)

    Returns:
        ndarray: camera intrinsic matrix with shape=(3,3)
    """
    with open(calib_path, "r") as f:
        lines = f.readlines()
        p2 = lines[0].split()[1:]
        p2 = np.array(p2, dtype=np.float32).reshape(3, 4)
    return p2[:, :3]


def read_kitti_pose(pose_path):
    """Read kitti pose file and get relative transform

    Args:
        pose_path (string): path to pose file

    Returns:
        ndarray: relative transforms with shape=(N,4,4)
    """
    input_pose = np.loadtxt(pose_path)
    assert (input_pose.shape[1] == 12)
    # image_ids = input_pose[:, 0].astype(np.int32)
    input_pose = input_pose[1:, :]
    length = input_pose.shape[0]
    image_ids = [x for x in range(1, length)]
    input_pose = input_pose.reshape(-1, 3, 4)
    bottom = np.zeros((length, 1, 4))
    bottom[:, :, -1] = 1
    absolute_transform = np.concatenate((input_pose, bottom), axis=1)
    relative_transforms = []
    for idx in range(length - 1):
        relative_transform = invT(
            absolute_transform[idx + 1]) @ absolute_transform[idx]
        relative_transforms.append(relative_transform)
    return image_ids, relative_transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="00")
    parser.add_argument("--kitti_root", type=str, default="KITTI_odom/sequences")
    parser.add_argument("--pose_root", type=str, default="odometry/orbslam2")
    parser.add_argument("--output_root", type=str, default="results")
    args = parser.parse_args()

    # create output dir
    if not isdir(args.output_root):
        mkdir(args.output_root)
    vis_dir = join(args.output_root, "vis")
    if not isdir(vis_dir):
        mkdir(vis_dir)

    fourcc = cv2.VideoWriter.fourcc('m', 'p', '4', 'v')

    video_filename = "bev_{}.mp4".format(args.sequence)
    video = cv2.VideoWriter(join(vis_dir, video_filename), fourcc, 30, (960, 736))
    if not video.isOpened():
        print("video open failed")
        exit(0)

    # read poses
    sequence = args.sequence
    pose_file = join(args.pose_root, f"{sequence}.txt")
    image_ids, relative_transforms = read_kitti_pose(pose_file)

    # prepare image list
    image_dir = join(args.kitti_root, sequence, "image_0")
    image_list = listdir(image_dir)
    image_list.sort()

    # remove last image
    image_ids = image_ids[:-1]
    # image_list start with 000000.png while image_ids start with 1
    image_list = [image_list[i + 1] for i in image_ids]

    # read calibration
    calib_path = join(args.kitti_root, sequence, "calib.txt")
    camera_K = read_kitti_calib(calib_path)
    print(camera_K)

    # run
    gnf = GroundNormalFilterIEKF()
    vis = Visualization(K=camera_K, d=None, input_wh=(1241, 376))
    for idx, image_file in enumerate(tqdm(image_list)):
        image_path = join(image_dir, image_file)
        relative_so3 = relative_transforms[idx][:3, :3]
        print("\nT21\n", relative_transforms[idx])
        compensation_se3 = gnf.update(relative_so3)
        compensation_so3 = compensation_se3[:3, :3]
        print("compR: ", compensation_so3)
        image = cv2.imread(image_path)

        pitch = R.from_matrix(compensation_so3).as_euler('zxy', degrees=True)[1]
        roll = R.from_matrix(compensation_so3).as_euler('zxy', degrees=True)[0]
        Rx = R.from_euler('zxy', [roll, pitch, 0], degrees=True).as_matrix()
        combined_image = vis.get_frame(image, Rx)
        video.write(combined_image)
        cv2.imshow("combined", combined_image)
        key = cv2.waitKey(5)
        # output_path = join(vis_dir, f"{idx:06d}.jpg")
        # cv2.imwrite(output_path, combined_image)

    video.release()
