""" LCR-Net: Localization-Classification-Regression for Human Pose
Copyright (C) 2017 Gregory Rogez & Philippe Weinzaepfel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>"""

import pickle
import pdb
import numpy as np
import cv2


def get_matrices(projMat, njts):
  projMat_block_diag = np.kron(np.identity(njts), projMat)
  M = np.kron(np.ones((njts, 1)), np.identity(4))
  formatted_joints3D = list(range(njts*4))
  for i in range(njts):
    formatted_joints3D[i*4+3] = 1.0
  formatted_joints2D = list(range(njts*3))
  for i in range(njts):
    formatted_joints2D[i*3+2] = 1.0
  return projMat_block_diag, M


def compute_reproj_delta_3d(detection, projMat_block_diag, M, njts):
  """
  returns (dx, dy, dz), apply this delta on 3D pose
  to visualize pose in world space
  """

  formatted_joints3D = np.ones( (4,njts), dtype=np.float32)
  formatted_joints3D[:3,:] = detection['pose3d'].reshape(3,njts)
  formatted_joints3D = formatted_joints3D.T.ravel()

  M[:, 3] = formatted_joints3D

  A = np.dot(projMat_block_diag, M)

  formatted_joints2D = np.ones( (3,njts), dtype=np.float32)
  formatted_joints2D[:2,:] = detection['pose2d'].reshape(2,njts)
  formatted_joints2D = formatted_joints2D.T.ravel()

  x,resid,rank,s = np.linalg.lstsq(A, formatted_joints2D)
  x /= x[3]

  return x[0:3]

