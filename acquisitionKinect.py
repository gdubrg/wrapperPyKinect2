from __future__ import division
import time
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import cv2
import numpy as np
import math

import ctypes
import _ctypes
import sys

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

# colors for drawing skeletons
SKELETON_COLORS = 255

# JOINTS_UPPER_BODY = [
#     PyKinectV2.JointType_HipLeft,
#     PyKinectV2.JointType_KneeLeft,
#     PyKinectV2.JointType_AnkleLeft,
#     PyKinectV2.JointType_FootLeft,
#     PyKinectV2.JointType_HipRight,
#     PyKinectV2.JointType_KneeRight,
#     PyKinectV2.JointType_AnkleRight,
#     PyKinectV2.JointType_FootRight
# ]

class AcquisitionKinect():
    def __init__(self, resolution_mode=1.0):
        self.resolution_mode = resolution_mode

        self._done = False

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body | PyKinectV2.FrameSourceTypes_Depth)

        # here we will store skeleton data
        self._bodies = None
        self.body_tracked = False
        self.joint_points = np.array([])
        self.joint_points3D = np.array([])
        self.joint_points_RGB = np.array([])
        self.joint_state = np.array([])

        self._frameRGB = None
        self._frameDepth = None
        self._frameDepthQuantized = None
        self._frameSkeleton = None
        self.frameNum = 0


    def get_frame(self, frame):
        self.acquireFrame()
        frame.ts = int(round(time.time() * 1000))

        self.frameNum += 1

        # try:
        #     frame.frameRGB = self._frameRGB.copy()
        #     frame.frameDepth = self._frameDepth.copy()
        #     frame.frameSkeleton = self._frameSkeleton.copy()
        # except:
        frame.frameRGB = self._frameRGB
        frame.frameDepth = self._frameDepth
        frame.frameDepthQuantized = self._frameDepthQuantized
        frame.frameSkeleton = self._frameSkeleton

        # frame.body_tracked = self.body_tracked
        # frame.bodyJoints = [self.joint_points[i] for i in range(len(self.joint_points)) if i not in JOINTS_UPPER_BODY]
        # frame.bodyJoints3D = [self.joint_points3D[i] for i in range(len(self.joint_points3D)) if i not in JOINTS_UPPER_BODY]
        # frame.bodyJointsRGB = [self.joint_points_RGB[i] for i in range(len(self.joint_points_RGB)) if i not in JOINTS_UPPER_BODY]
        # frame.bodyJointState = [self.joint_state[i] for i in range(len(self.joint_state)) if i not in JOINTS_UPPER_BODY]

        frame.frame_num = self.frameNum

        # get shoulder rotations (yaw, pitch and roll in Euler angles)
        euler, quat = self.get_shoulder_angles()
        if euler is None:
            return
        frame.shoulder_orientation_euler = euler
        frame.shoulder_orientation_quat = quat
        self.euler_to_quaternion(euler['roll']*np.pi/180, euler['pitch']*np.pi/180, euler['yaw']*np.pi/180)

    def get_shoulder_angles(self):

        # Find versors
        if self.joint_points3D is None or len(self.joint_points3D) == 0:
            return None, None

        p7 = self.joint_points3D[PyKinectV2.JointType_ShoulderRight]
        p4 = self.joint_points3D[PyKinectV2.JointType_ShoulderLeft]
        p13 = self.joint_points3D[PyKinectV2.JointType_SpineBase]

        # N1 versor
        normaN1 = math.sqrt((p7.x - p4.x)**2 + (p7.y - p4.y)**2 + (p7.z - p4.z)**2)
        N1 = PyKinectV2._Joint()
        N1.x = ((p7.x - p4.x) / normaN1)
        N1.y = ((p7.y - p4.y) / normaN1)
        N1.z = ((p7.z - p4.z) / normaN1)

        # Tmp versor
        normaU = math.sqrt((p7.x - p13.x)**2 + (p7.y - p13.y)**2 + (p7.z - p13.z)**2)
        U = PyKinectV2._Joint()
        U.x = ((p7.x - p13.x) / normaU)
        U.y = ((p7.y - p13.y) / normaU)
        U.z = ((p7.z - p13.z) / normaU)

        # N3 versor
        N3 = PyKinectV2._Joint()
        N3.x = N1.y * U.z - N1.z * U.y
        N3.y = N1.z * U.x - N1.x * U.z
        N3.z = N1.x * U.y - N1.y * U.x

        normaN3 = math.sqrt(N3.x**2 + N3.y**2 + N3.z**2)
        N3.x = N3.x / normaN3
        N3.y = N3.y / normaN3
        N3.z = N3.z / normaN3

        # N2 versor
        N2 = PyKinectV2._Joint()
        N2.x = N3.y * N1.z - N3.z * N1.y
        N2.y = N3.z * N1.x - N3.x * N1.z
        N2.z = N3.x * N1.y - N3.y * N1.x

        matrR = np.array([[N1.x,N1.y,N1.z],[N2.x,N2.y,N2.z],[N3.x,N3.y,N3.z]])
        euler = self.rotationMatrixToEulerAngles(matrR)
        shoulder_pitch = euler[0] * 180 / np.pi
        shoulder_yaw   = euler[1] * 180 / np.pi
        shoulder_roll  = euler[2] * 180 / np.pi

        matrR = np.array([[N1.x, N1.y, N1.z, 0], [N2.x, N2.y, N2.z, 0], [N3.x, N3.y, N3.z, 0], [0, 0, 0, 1]])
        quat = self.rotationMatrixToQuaternion(matrR)

        return {"roll": shoulder_roll, "pitch": shoulder_pitch, "yaw": shoulder_yaw}, quat

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles. The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])


    def rotationMatrixToQuaternion(self, matrix, isprecise=True):

        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 1, 2, 3
                if M[1, 1] > M[0, 0]:
                    i, j, k = 2, 3, 1
                if M[2, 2] > M[i, i]:
                    i, j, k = 3, 1, 2
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                             [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                             [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                             [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState
        joint1State = joints[joint1].TrackingState

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good 
        start = (int(jointPoints[joint0].x), int(jointPoints[joint0].y))
        end = (int(jointPoints[joint1].x), int(jointPoints[joint1].y))

        try:
            cv2.line(self._frameSkeleton, start, end, color, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        self._frameSkeleton = np.zeros_like(self._frameDepthQuantized, dtype=np.uint8)
        # Torso
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft)

        # Right Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft)
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight)
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight)
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight)
        #
        # # Left Leg
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft)
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft)
        # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft)

    def get_depth_frame(self):
        self._frameDepth = self._kinect.get_last_depth_frame()
        self._frameDepth = self._frameDepth.reshape(((424, 512))).astype(np.uint16)
        self._frameDepthQuantized = ((self._frameDepth.astype(np.int32)-500)/8.0).astype(np.uint8)

    def get_color_frame(self):
        self._frameRGB = self._kinect.get_last_color_frame()
        self._frameRGB = self._frameRGB.reshape((1080, 1920,-1)).astype(np.uint8)
        self._frameRGB = cv2.resize(self._frameRGB, (0,0), fx=1/self.resolution_mode, fy=1/self.resolution_mode)

    def draw_depth_frame(self):
        img[img<0]=0
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        image = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        cv2.imshow("depth",self._frameDepth.astype(np.uint8))

    def draw_color_frame(self):
        # cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        cv2.imshow("color",self._frameRGB.astype(np.uint8))

    def body_joint_to_depth_space(self, joint):
        return self._kinect._mapper.MapCameraPointToDepthSpace(joint.Position)

    def body_joint_to_color_space(self, joint):
        return self._kinect._mapper.MapCameraPointToColorSpace(joint.Position)

    def body_joints_to_depth_space(self, joints):
        joint_points = np.ndarray((PyKinectV2.JointType_Count), dtype=np.object)

        for j in range(0, PyKinectV2.JointType_Count):
            joint_points[j] = self.body_joint_to_depth_space(joints[j])

        return joint_points

    def body_joints_to_color_space(self, joints):
        joint_points = np.ndarray((PyKinectV2.JointType_Count), dtype=np.object)

        for j in range(0, PyKinectV2.JointType_Count):
            joint_points[j] = self.body_joint_to_color_space(joints[j])

        return joint_points

    def body_joints_to_camera_space(self, joints):
        joint_points = np.ndarray((PyKinectV2.JointType_Count), dtype=np.object)

        for j in range(0, PyKinectV2.JointType_Count):
            joint_points[j] = joints[j].Position

        return joint_points

    def body_joints_state(self, joints):
        joint_state = np.ndarray(PyKinectV2.JointType_Count, dtype='int')

        for j in range(0, PyKinectV2.JointType_Count):
             state = 0
             if joints[j].TrackingState == PyKinectV2.TrackingState_Inferred:
                 state = 1
             elif joints[j].TrackingState == PyKinectV2.TrackingState_Tracked:
                 state = 2
             joint_state[j] = state

        return joint_state


    def acquireFrame(self):

        if self._kinect.has_new_depth_frame():
            self.get_depth_frame()

        if self._kinect.has_new_color_frame():
            self.get_color_frame()

        if self._kinect.has_new_body_frame():
            self._bodies = self._kinect.get_last_body_frame()

        # --- draw skeletons to _frame_surface
        if self._bodies is not None:
            for i in range(0, self._kinect.max_body_count)[::-1]:
                body = self._bodies.bodies[i]
                if not body.is_tracked:
                    self.body_tracked = False
                    continue

                self.body_tracked = True

                joints = body.joints

                # convert joint coordinates to color space
                self.joint_points3D = self.body_joints_to_camera_space(joints)
                self.joint_points = self.body_joints_to_depth_space(joints)
                self.joint_points_RGB = self.body_joints_to_color_space(joints)
                self.joint_state = self.body_joints_state(joints)
                self.draw_body(joints, self.joint_points, SKELETON_COLORS)

    def close(self):
        self._kinect.close()
        self._frameDepth = None
        self._frameRGB = None
        self._frameSkeleton = None
