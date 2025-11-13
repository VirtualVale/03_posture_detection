"""
Module: detector.py
# Information:
#   Author: Valentin Böse
#   Created: 2025-11-12
#   Last Modified: 2025-11-13
#   Purpose: Detect human pose landmarks in video files using MediaPipe Pose Landmarker,
#            annotate frames with landmarks, write an annotated video and a JSON file
#            containing per-frame landmark detections.
#
"""

import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from typing import Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import json
import os


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result: any) -> np.ndarray:
  """Draw the detected landmarks to the image.

  Args:
      rgb_image (np.ndarray): image to draw on
      detection_result (any): detection result from the posture detection model

  Returns:
      np.ndarray: annotated image
  """

  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  if not pose_landmarks_list:
    return annotated_image

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def process_video(video_path: str, annotated_path: str) -> Tuple[str, str]:
  """Processes video from a given path with posture landmarks. Writes to the annotated path.

  Args:
      video_path (str): original video
      annotated_path (str): output path

  Returns:
      Tuple[str, str]: path to result video, path to json with landmarks
  """

  cap = cv2.VideoCapture(video_path)

  base_options = python.BaseOptions(model_asset_path='/root/projects/03_posture_detection/pose_landmarker_full.task')
  options = vision.PoseLandmarkerOptions(
      base_options=base_options,
      output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  
  frame_number = 0
  detections_data = []  # list of per-frame detections (poses -> landmarks)
  out = None

  # prepare output paths
  root, ext = os.path.splitext(video_path)
  video_name = os.path.basename(root)
  output_video_path = os.path.join(annotated_path, f"{video_name}-annotated.mp4") 
  output_json_path = os.path.join(annotated_path, f"{video_name}-detections.json")

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # convert BGR (OpenCV) to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create writer on first frame so we know dimensions/fps
    if frame_number == 0:
      height, width = frame.shape[:2]
      fps = cap.get(cv2.CAP_PROP_FPS)
      if fps <= 0 or np.isnan(fps):
        fps = 30.0
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # collect detection landmarks in a serializable structure
    frame_det = []
    if detection_result.pose_landmarks:
      for pose in detection_result.pose_landmarks:
        pose_list = []
        for lm in pose:
          pose_list.append({'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z)})
        frame_det.append(pose_list)
    detections_data.append(frame_det)

    # draw landmarks onto RGB image, convert back to BGR for writing
    annotated_rgb = draw_landmarks_on_image(rgb_frame, detection_result)
    annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    out.write(annotated_bgr)

    frame_number += 1

  cap.release()
  if out is not None:
    out.release()

  # save detections to JSON
  with open(output_json_path, 'w') as f:
    json.dump({'video_path': video_path, 'frame_count': frame_number, 'detections': detections_data}, f)

  return output_video_path, output_json_path


if __name__ == "__main__":

  video_directory = '/root/projects/03_posture_detection/original-videos'
  annotated_path = '/root/projects/03_posture_detection/annotated-videos'
  video_files = [f for f in os.listdir(video_directory) if f.endswith(('.mp4', '.MOV'))]

  for video_file in video_files:
    video_path = os.path.join(video_directory, video_file)
    annotated_video, detections_json = process_video(video_path, annotated_path)
    print("Saved:", annotated_video, detections_json)