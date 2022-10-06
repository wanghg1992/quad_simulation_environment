#!/usr/bin/env python3

import time

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import sys
import pybullet as p
import pybullet_data
import math
import numpy as np
import random

import pyquaternion as pyq
from scipy.spatial.transform import Rotation as R

sys.path.append("../../../")
import robot_description

USE_ROS = True

if USE_ROS:
    import rospy
    from std_msgs.msg import Float32MultiArray
    from sensor_msgs.msg import JointState

class QuadrupedEnv(gym.Env):

    def __init__(self):
        # input size and output size for reinforce learning
        self.action_space = spaces.Box(np.array([-0.1] * 12), np.array([0.1] * 12))
        self.observation_space = spaces.Box(np.array([-2.0] * 32), np.array([2.0] * 32))

        # load simulation environment
        p.connect(p.GUI)    # p.GUI for render; p.DIRECT for non-graphical version
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,                                  #reset view
                                     cameraPitch=-40, cameraTargetPosition=[0.55, -0.35, 0.2])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        # robot_urdf_path = os.path.dirname(os.path.abspath(__file__)) + "/../../../robot_description/"
        self.quadrupedUid = p.loadURDF(robot_description.getDataPath() + "/urdf/quadruped_robot/quadruped_robot.urdf", [0, 0, .45],
                                       [0, 0.0, 0.0, 1], useFixedBase=False)
        self.sim_step_time = 1.0 / 240
        self.sim_timer = 0
        self.leg_length = [0.046, 0.066, 0.065]

        # get joints and links
        self.allJointIds = []
        self.actuatedJointIds = []
        self.footJointIds = []
        for j in range(p.getNumJoints(self.quadrupedUid)):
            info = p.getJointInfo(self.quadrupedUid, j)
            jointType = info[2]
            self.allJointIds.append(j)
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.actuatedJointIds.append(j)
        self.footJointIds.append(3)
        self.footJointIds.append(7)
        self.footJointIds.append(11)
        self.footJointIds.append(15)
        # enable foot joint sensors
        for i in self.footJointIds:
            p.enableJointForceTorqueSensor(self.quadrupedUid, i, 1)

        print("joint number:",p.getNumJoints(self.quadrupedUid))

        # joint control
        self.joint_pos = np.array([0.0] * 12)
        self.joint_vel = np.array([0.0] * 12)
        self.init_joint_pos = np.array([0, -0.6435, 1.2870, 0, -0.6435, 1.2870, 0, -0.6435, 1.2870, 0, -0.6435, 1.2870])
        self.joint_pos_last = np.array([0.0] * 12)

        # foot control
        self.step_freq = 3
        self.step_period = 1.0/self.step_freq
        self.step_phase = [0, 1, 1, 0]
        self.step_timer = 0
        self.foot_pos = np.array([0.0] * 12)
        self.foot_pos_cmd = np.array([0.0] * 12)
        self.foot_pos_adjust = np.array([0.0] * 12)
        self.foot_vel_adjust = np.array([0.0] * 12)
        self.stand_height = 0.13

        # debug function
        self.StopSimulationID = p.addUserDebugParameter("StopSimulation", 1, 0, 0)
        self.StepSimulationID = p.addUserDebugParameter("StepSimulation", 1, 0, 0)
        self.StepSimulation = 0
        self.StepNumID = p.addUserDebugParameter("StepNum", 0, 30, 5)
        self.StepNum = 0

        self.reset()

        # msg publish
        if USE_ROS:
            # self.ros_debug_pub = rospy.Publisher('debug_data', Float32MultiArray, queue_size=10)
            self.ros_debug_pub = rospy.Publisher('debug_data', JointState, queue_size=10)
            rospy.init_node('talker', anonymous=True)

        # record action of classical control
        self.obs_act = [0]*12

    def step(self, action):
        return self.step_foot_velocity(action)

    def step_foot_velocity(self, action):

        # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)  #open render

        # robot planner
        self.foot_pos[2:12:3] = -self.stand_height
        self.sim_timer = self.sim_timer + self.sim_step_time
        self.step_timer = self.step_timer + self.sim_step_time
        # step_freq = 2
        # foot_pos_z = math.sin(self.timer * self.step_freq * 6.28) * 0.04
        # print("foot_pos_z:",foot_pos_z)
        # step_phase = [0]*4
        lf_swing = False
        rf_swing = False
        # if foot_pos_z > 0.02 or lf_swing:
        #     lf_swing = False
        #     rf_swing = False
        #     self.foot_pos[2] = self.foot_pos[11] = -self.stand_height + (foot_pos_z-0.02)
        #     step_phase[0] = step_phase[3] = 0
        #     step_phase[1] = step_phase[2] = 1
        # elif foot_pos_z < -0.02 or rf_swing:
        #     rf_swing = False
        #     lf_swing = False
        #     self.foot_pos[5] = self.foot_pos[8] = -self.stand_height - (foot_pos_z+0.02)
        #     step_phase[0] = step_phase[3] = 1
        #     step_phase[1] = step_phase[2] = 0
        # else:
        #     step_phase[0] = step_phase[3] = step_phase[1] = step_phase[2] = 1
        # switch
        if self.step_timer > self.step_period:
            self.step_timer = 0
            if self.step_phase[0] == 0:
                self.step_phase[0] = self.step_phase[3] = 1
                self.step_phase[1] = self.step_phase[2] = 0
            else:
                self.step_phase[0] = self.step_phase[3] = 0
                self.step_phase[1] = self.step_phase[2] = 1

        # planner
        if self.step_phase[0] == 0:
            self.foot_pos[2] = self.foot_pos[11] = -self.stand_height + (1 + math.cos(self.step_timer / self.step_period * 6.28 - 3.14)) * 0.02
        elif self.step_phase[1] == 0:
            self.foot_pos[5] = self.foot_pos[8] = -self.stand_height + (1 + math.cos(self.step_timer / self.step_period * 6.28 - 3.14)) * 0.02

        # action=[1, 2]
        max_adj = [0.05, 0.05, 0.01] * 4
        min_adj = [-0.05, -0.05, -0.01] * 4
        # middle_value=[action[0] ,action[1],action[1],action[0]]
        # print("action:",action)
        # middle = action.copy()
        # action[0:12:6] = [middle[0]]*2
        # action[3:12:6] = [middle[3]]*2
        # action[1:12:6] = [middle[1]]*2
        # action[4:12:6] = [middle[4]]*2
        # self.foot_vel_adjust = self.foot_vel_adjust + np.array(action) * self.step_time
        self.foot_pos_adjust = np.clip(self.foot_pos_adjust + np.array(action) * self.sim_step_time,
                                       min_adj, max_adj)
        # if max(self.foot_pos_adjust[1:12:3]) > 0.1:
        #     print("self.foot_pos_adjust[1:12:3]:",self.foot_pos_adjust[1:12:3])
        self.foot_pos_cmd = self.foot_pos + self.foot_pos_adjust

        # print("self.foot_pos_adjust:",self.foot_pos_adjust)
        # print("self.foot_pos_cmd:",self.foot_pos_cmd)

        # self.foot_pos_final = self.foot_pos+np.array(action)

        #robot control
        for leg in range(4):
            self.inverse_kinematic(leg)
        self.joint_vel = (self.joint_pos - self.joint_pos_last)/self.sim_step_time
        max_force = [500]*12
        kp = [0.1]*12
        kd = [0.4]*12
        p.setJointMotorControlArray(self.quadrupedUid, list(self.actuatedJointIds), p.POSITION_CONTROL,
                                    targetPositions=list(self.joint_pos), targetVelocities=list(self.joint_vel),
                                    forces=max_force, positionGains=kp, velocityGains=kd)

        self.joint_pos_last = self.joint_pos

        # simulation
        # print(p.readUserDebugParameter(self.StopSimulationID),p.readUserDebugParameter(self.StepSimulationID))
        while (p.readUserDebugParameter(self.StopSimulationID) % 2) == 1 and self.StepNum<1:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            p.setTimeStep(0.000)
            p.stepSimulation()
            if p.readUserDebugParameter(self.StepSimulationID) > self.StepSimulation:
                self.StepSimulation = p.readUserDebugParameter(self.StepSimulationID)
                self.StepNum = p.readUserDebugParameter(self.StepNumID)
                break
        self.StepNum = self.StepNum - 1
        p.setTimeStep(self.sim_step_time)
        p.stepSimulation()

        # get feedback
        state_robot_joint = list(p.getJointState(self.quadrupedUid, joint) for joint in self.actuatedJointIds)
        state_robot_base = list(p.getBasePositionAndOrientation(self.quadrupedUid))
        state_robot_base[1] = self.getHerizonEulerFromQuaternion(state_robot_base[1])
        state_robot_base_vel_local = self.getBaseVelocityLocal(self.quadrupedUid)
        # print("robot euler:",state_robot_base[1])
        # print("robot euler vel:", state_robot_base_vel[1])

        foot_pos_fdb=[0]*12
        for leg in range(4):
            foot_pos_fdb[leg*3:leg*4] = self.forward_kinematic([state_robot_joint[leg*3+0][0],state_robot_joint[leg*3+1][0],state_robot_joint[leg*3+2][0]])

        # get foot states
        foot_contact = []
        for i in self.footJointIds:
            foot_joint_force = p.getJointState(self.quadrupedUid, i)[2]
            print("foot joint",i,":",foot_joint_force)
            foot_contact.append(math.sqrt(foot_joint_force[0]**2+foot_joint_force[1]**2+foot_joint_force[2]**2) > 2)

        # evaluation
        def k(x):
            return -1.0 / (math.exp(x) + 2.0 + math.exp(-x))
        c_w = -6.0 * self.sim_step_time
        cost_ang_vel = c_w * k(np.linalg.norm(np.array(state_robot_base_vel_local[1]) - np.array([0, 0, 0]), ord=1))
        c_v1 = -10.0 * self.sim_step_time
        c_v2 = -4.0 * self.sim_step_time
        cost_lin_vel = c_v1 * k(np.linalg.norm(c_v2 * (np.array(state_robot_base_vel_local[0]) - np.array([0, 0, 0])), ord=1))
        k_c = 0
        c_tor = 0.005 * self.sim_step_time
        cost_torque = k_c * c_tor * (np.linalg.norm(np.array(state_robot_joint)[:, 3])) ** 2
        c_js = 0.03 * self.sim_step_time
        cost_joint_vel = k_c * c_js * (np.linalg.norm(np.array(state_robot_joint)[:, 1])) ** 2
        c_f = 0.1 * self.sim_step_time
        p_fiz_d = 0.07
        c_fv = 2.0 * self.sim_step_time
        c_o = 0.4 * self.sim_step_time
        cost_ori = 0.3 * c_o * np.linalg.norm(np.array(state_robot_base[1]))
        c_h = c_o
        cost_hgt = 0.1 * c_h * np.linalg.norm(np.array(state_robot_base[0][2] - self.stand_height))
        c_s = 0.5 * self.sim_step_time
        cost_balance = state_robot_base[1][0] * state_robot_base_vel_local[1][0] + state_robot_base[1][1] * state_robot_base_vel_local[1][1] + \
                       state_robot_base[1][2] * state_robot_base_vel_local[1][2]*1


        # reward = - np.linalg.norm(np.asarray(action) - np.asarray(self.obs_act))
        # reward = -(
        #         cost_ang_vel * 0 + cost_lin_vel * 0 + cost_torque * 0 + cost_joint_vel * 0 + cost_ori * 0 + cost_hgt * 1 + cost_balance * 1)

        # cost_lin_vel = c_v1 * k(
        #     np.linalg.norm(c_v2 * (np.array(state_robot_base_vel_local[0]) - np.array([0.2, 0, 0])), ord=1))
        # reward = -(
        #             cost_ang_vel * 0 + cost_lin_vel * 1 + cost_torque * 0 + cost_joint_vel * 0 + cost_ori * 0 + cost_hgt * 1 + cost_balance * 1)

        # cost_lin_vel = c_v1 * k(
        #     np.linalg.norm(c_v2 * (np.array(state_robot_base_vel_local[0]) - np.array([0.2, 0, 0])), ord=1))
        # cost_balance = 0.3 * c_o *k(state_robot_base[1][0] * state_robot_base_vel_local[1][0] + state_robot_base[1][1] * state_robot_base_vel_local[1][1] + \
        #                state_robot_base[1][2] * state_robot_base_vel_local[1][2]*1)
        # reward = -(
        #             cost_ang_vel * 0 + cost_lin_vel * 1 + cost_torque * 0 + cost_joint_vel * 0 + cost_ori * 0 + cost_hgt * 1 + cost_balance * 1)

        cost_lin_vel = 2*c_v1 * k(
            np.linalg.norm(c_v2 * (np.array(state_robot_base_vel_local[0]) - np.array([0.2, 0, 0])), ord=1))
        cost_balance = 0.3 * c_o *k(state_robot_base[1][0] * state_robot_base_vel_local[1][0]*5 + state_robot_base[1][1] * state_robot_base_vel_local[1][1]*5 + \
                       state_robot_base[1][2] * state_robot_base_vel_local[1][2]*1)
        reward = -(
                    cost_ang_vel * 0 + cost_lin_vel * 1 + cost_torque * 0 + cost_joint_vel * 0 + cost_ori * 0 + cost_hgt * 1 + cost_balance * 1)


        # print("cost_ori:",cost_ori)
        # print("reward:",reward)
        # print("action:",action)

        if state_robot_base[0][2] < 0.05 \
                or abs(state_robot_base[1][0]) > 0.4 or abs(state_robot_base[1][1]) > 0.8\
                or self.sim_timer>5.0:
            done = True
            print("step end:", state_robot_base[0], np.array(state_robot_base[1]))
        else:
            done = False
            # print("step not end")

        observation = [s for elem in state_robot_base[0:2] for s in elem]
        for i in state_robot_base_vel_local[0:2]:
            for j in i:
                observation.append(j)
        for i in self.step_phase:
            observation.append(i)
        for i in foot_contact:
            observation.append(i)
        for i in self.foot_pos_cmd:
            observation.append(i)

        state_object, _ = p.getBasePositionAndOrientation(self.quadrupedUid)
        info = state_object

        info = {
            'reward_forward': 1
        }

        if USE_ROS:
            # debug_data = Float32MultiArray(data=[self.foot_pos_cmd[2], foot_pos_fdb[2]])
            debug_data = JointState()
            # debug_data.header.stamp = rospy.get_time()
            debug_data.header.stamp = rospy.rostime.Time(self.sim_timer)
            debug_data.position = [self.foot_pos_cmd[0], self.foot_pos_cmd[1], self.foot_pos_cmd[2],
                                   foot_pos_fdb[0], foot_pos_fdb[1], foot_pos_fdb[2],
                                   self.step_phase[0],foot_contact[0]]
            self.ros_debug_pub.publish(debug_data)

        # print("observation:",observation)
        # print("reward:",reward)
        # print("done:",done)
        # print("info:",info)

        self.obs_act = self.classical_control(observation)

        return observation, reward, done, info

    def step_joint_torque(self, action):
        # close joint lock
        p.setJointMotorControlArray(self.quadrupedUid, list(self.actuatedJointIds), p.VELOCITY_CONTROL,
                                    forces=[0]*12)
        p.setJointMotorControlArray(self.quadrupedUid, list(self.actuatedJointIds), p.TORQUE_CONTROL,
                                    forces=action)
        # simulation
        # print(p.readUserDebugParameter(self.StopSimulationID),p.readUserDebugParameter(self.StepSimulationID))
        while (p.readUserDebugParameter(self.StopSimulationID) % 2) == 1 and self.StepNum<1:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            p.setTimeStep(0.000)
            p.stepSimulation()
            if p.readUserDebugParameter(self.StepSimulationID) > self.StepSimulation:
                self.StepSimulation = p.readUserDebugParameter(self.StepSimulationID)
                self.StepNum = p.readUserDebugParameter(self.StepNumID)
                break
        self.StepNum = self.StepNum - 1

        p.setTimeStep(self.sim_step_time)
        p.stepSimulation()
        # get feedback
        state_robot_base = list(p.getBasePositionAndOrientation(self.quadrupedUid))
        # state_robot_base[1] = self.getHerizonEulerFromQuaternion(state_robot_base[1])
        state_robot_base_vel_local = self.getBaseVelocityLocal(self.quadrupedUid)
        state_robot_joint = list(p.getJointState(self.quadrupedUid, joint) for joint in self.actuatedJointIds)

        # get foot states
        foot_force = []
        foot_contact = []
        for i in self.footJointIds:
            foot_joint_force = p.getJointState(self.quadrupedUid, i)[2]
            # print("foot joint",i,":",foot_joint_force)
            foot_force.append(foot_joint_force)
            foot_contact.append(math.sqrt(foot_joint_force[0]**2+foot_joint_force[1]**2+foot_joint_force[2]**2) > 2)
        fs = [foot_force, foot_contact]

        observation = [s for elem in state_robot_base[0:2] for s in elem]
        for i in state_robot_base_vel_local[0:2]:
            for j in i:
                observation.append(j)
        for i in state_robot_joint:
            observation.append(i[0])
        return observation, state_robot_base, state_robot_base_vel_local, state_robot_joint, fs

    def reset(self):
        # simulation setting reset
        # p.resetSimulation()
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # planeId = p.loadURDF("plane.urdf")
        # self.quadrupedUid = p.loadURDF("./urdf/quadruped_robot/quadruped_robot.urdf", [0, 0, .45],
        #                                [0, 0.0, 0.0, 1], useFixedBase=False)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)       #close render
        p.setGravity(0, 0, -10)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # print(pybullet_data.getDataPath())

        # control reset
        self.joint_pos = np.array([0.0] * 12)
        self.joint_vel = np.array([0.0] * 12)
        self.foot_pos_adjust = np.array([0.0] * 12)
        self.foot_vel_adjust = np.array([0.0] * 12)
        self.foot_pos[:] = 0
        self.foot_pos[2:12:3] = -self.stand_height
        self.foot_pos_cmd = self.foot_pos + self.foot_pos_adjust
        for leg in range(4):
            self.inverse_kinematic(leg)
        self.sim_timer = 0
        self.step_timer = 0

        # reset robot position and orientation
        p.resetBasePositionAndOrientation(self.quadrupedUid,[0, 0, self.stand_height+0.01],
                                          [0, 0.0, 0.0, 1])
        for leg in range(4):
            self.inverse_kinematic(leg)
        for index, jointId in enumerate(self.actuatedJointIds):
            p.resetJointState(self.quadrupedUid, jointId, self.joint_pos[index])

        self.joint_pos_last = self.joint_pos

        # # get feedback
        # state_robot_joint = list(p.getJointState(self.quadrupedUid, joint) for joint in self.actuatedJointIds)
        # foot_pos_fdb=[0]*12
        # for leg in range(4):
        #     foot_pos_fdb[leg*3:leg*4] = self.forward_kinematic([state_robot_joint[leg*3+0][0],state_robot_joint[leg*3+1][0],state_robot_joint[leg*3+2][0]])
        # print("foot_pos_fdb:",foot_pos_fdb)

        # get robot states
        ## get euler with reference to herizon
        state_robot_base = list(p.getBasePositionAndOrientation(self.quadrupedUid))
        state_robot_base[1] = self.getHerizonEulerFromQuaternion(state_robot_base[1])
        print("robot euler:", state_robot_base[1])

        step_phase = [1, 1, 1, 1]
        contact = [1, 1, 1, 1]

        observation = [s for elem in state_robot_base for s in elem]
        for i in range(6):
            observation.append(0)
        for i in step_phase:
            observation.append(i)
        for i in contact:
            observation.append(i)
        for i in self.foot_pos_cmd:
            observation.append(i)
        print("observation:",observation)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)   #open render

        self.obs_act = self.classical_control(observation)

        # input("press any key to continue...")
        return observation

    def render(self, mode='human'):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)  # open render
        return None

    def close(self):
        p.disconnect()

    def show_joint(self):
        for i in range(p.getNumJoints(self.quadrupedUid)):
            joint_info = p.getJointInfo(self.quadrupedUid, i)
            print(joint_info)

    def forward_kinematic(self, joint_angle):
        theta0 = joint_angle[0]
        theta1 = joint_angle[1]
        theta2 = joint_angle[2]
        l0 = self.leg_length[0]
        l1 = self.leg_length[1]
        l2 = self.leg_length[2]
        x = -math.sin(theta1) * l1 - math.sin(theta1 + theta2) * l2
        y = math.sin(theta0) * l0 + math.sin(theta0) * math.cos(theta1) * l1 + math.sin(theta0) * math.cos(theta1 + theta2) * l2
        z = -(math.cos(theta0) * l0 + math.cos(theta0) * math.cos(theta1) * l1 + math.cos(theta0) * math.cos(theta1 + theta2) * l2)
        return [x,y,z]

    def inverse_kinematic(self, leg_id):
        x = self.foot_pos_cmd[0 + leg_id * 3]
        y = self.foot_pos_cmd[1 + leg_id * 3]
        z = self.foot_pos_cmd[2 + leg_id * 3]
        l0 = self.leg_length[0]
        l1 = self.leg_length[1]
        l2 = self.leg_length[2]

        if z < 0:
            self.joint_pos[0 + leg_id * 3] = math.atan(-y / z)
        elif y > 0:
            self.joint_pos[0 + leg_id * 3] = math.atan(-y / z) + 3.1415926
        else:
            self.joint_pos[0] = math.atan(-y / z) - 3.1415926

        middle_value = -z * math.cos(self.joint_pos[0 + leg_id * 3]) + y * math.sin(self.joint_pos[0 + leg_id * 3]) - l0
        baita = math.atan(x / middle_value)

        if (math.sqrt(y * y + z * z) - l0) * (math.sqrt(y * y + z * z) - l0) + x * x > (l1 + l2) * (l1 + l2):
            self.joint_pos[1 + leg_id * 3] = -math.atan(x / (math.sqrt(y * y + z * z) - l0))
            self.joint_pos[2 + leg_id * 3] = 0
        else:
            if leg_id==0 or leg_id==1 or True:    # < config
                self.joint_pos[2 + leg_id * 3] = -math.acos(
                    (middle_value * middle_value + x * x - l1 * l1 - l2 * l2) / 2 / l1 / l2)
            else:       # > config
                self.joint_pos[2 + leg_id * 3] = math.acos(
                    (middle_value * middle_value + x * x - l1 * l1 - l2 * l2) / 2 / l1 / l2)
            self.joint_pos[1 + leg_id * 3] = math.asin(
                -l2 * math.sin(self.joint_pos[2 + leg_id * 3]) / math.sqrt(middle_value * middle_value + x * x)) - baita

    def getHerizonEulerFromQuaternion(self, quaternion):
        yaw = p.getEulerFromQuaternion(quaternion)[2]
        coordinate_herizon = pyq.Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw)
        coordinate_robot = pyq.Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2],
                                          w=quaternion[3])
        middle = (coordinate_robot / coordinate_herizon).elements
        dis_quaternion =  [middle[1],middle[2],middle[3],middle[0]]
        middle = p.getEulerFromQuaternion(dis_quaternion)
        return [middle[0], middle[1], yaw]

    def getHerizonAngularVelocity(self, quaternion, angularVelocity):
        yaw = p.getEulerFromQuaternion(quaternion)[2]
        r = R.from_euler('xyz', [0, 0, yaw])
        r = np.asmatrix(r.as_matrix())
        angularVelocity = np.asarray(angularVelocity).reshape(3,1)
        # print("r*angularVelocity:",(r*angularVelocity).reshape(1,3).tolist()[0])
        return (r*angularVelocity).reshape(1,3).tolist()[0]

    def getBaseVelocityLocal(self, robotUid):
        state_robot_base = list(p.getBasePositionAndOrientation(self.quadrupedUid))
        quaternion = state_robot_base[1]
        yaw = p.getEulerFromQuaternion(quaternion)[2]
        r = R.from_euler('xyz', [0, 0, yaw])
        r = np.asmatrix(r.as_matrix())

        state_robot_base_vel = list(p.getBaseVelocity(self.quadrupedUid))
        linearVelocity = np.asarray(state_robot_base_vel[0]).reshape(3,1)
        angularVelocity = np.asarray(state_robot_base_vel[1]).reshape(3,1)

        return [(r * linearVelocity).reshape(1, 3).tolist()[0],
                (r * angularVelocity).reshape(1, 3).tolist()[0]]

    def classical_control(self, observation):
        baseLinearVelocityLocal = observation[6:9]
        # print("baseLinearVelocityLocal:", baseLinearVelocityLocal)
        foot_pos_cmd = observation[20:32]
        step_phase = observation[12:16]
        contact = observation[16:20]
        footHold = np.asarray(baseLinearVelocityLocal[0:2]) * 1.0 / self.step_freq * (0.5 + 0.05)
        action = [0] * 12
        kp = [20,-20,0]
        kd = [2,-2,0]
        for leg in range(4):
            if contact[leg] == 1:
                action[leg * 3 + 0] = kp[1] * (observation[4]) + kd[1] * (observation[10]) * 1
                action[leg * 3 + 1] = kp[0] * (observation[3]) + kd[0] * (observation[9]) * 1
                # action[leg * 3 + 2] = 2.0 * (-self.stand_height - foot_pos_cmd[leg * 3 + 2])
            elif step_phase[leg] == 0:
                action[leg * 3 + 0] = 10.0 * (footHold[0] - foot_pos_cmd[leg * 3 + 0])
                action[leg * 3 + 1] = 10.0 * (footHold[1] - foot_pos_cmd[leg * 3 + 1])
            # else:
            #     action[leg * 3 + 2] = -0.05*2
        return action

if __name__ == '__main__':

    def getHerizonEulerFromQuaternion(quaternion):
        yaw = p.getEulerFromQuaternion(quaternion)[2]
        coordinate_herizon = pyq.Quaternion(axis=(0.0, 0.0, 1.0), radians=yaw)
        coordinate_robot = pyq.Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2],
                                          w=quaternion[3])
        middle = (coordinate_robot / coordinate_herizon).elements
        dis_quaternion =  [middle[1],middle[2],middle[3],middle[0]]
        middle = p.getEulerFromQuaternion(dis_quaternion)
        return [middle[0], middle[1], middle[2]]

    def getHerizonAngularVelocity(quaternion, angularVelocity):
        yaw = p.getEulerFromQuaternion(quaternion)[2]
        r = R.from_euler('xyz', [0, 0, yaw])
        r = np.asmatrix(r.as_matrix())
        angularVelocity = np.asarray(angularVelocity).reshape(3,1)
        # print("r*angularVelocity:",(r*angularVelocity).reshape(1,3).tolist()[0])
        return (r*angularVelocity).reshape(1,3).tolist()[0]

    def getBaseVelocityLocal(robotUiD):
        state_robot_base = list(p.getBasePositionAndOrientation(robotUiD))
        quaternion = state_robot_base[1]
        yaw = p.getEulerFromQuaternion(quaternion)[2]
        r = R.from_euler('xyz', [0, 0, yaw])
        r = np.asmatrix(r.as_matrix())

        state_robot_base_vel = list(p.getBaseVelocity(robotUiD))
        linearVelocity = np.asarray(state_robot_base_vel[0]).reshape(3,1)
        angularVelocity = np.asarray(state_robot_base_vel[1]).reshape(3,1)

        return [(r * linearVelocity).reshape(1, 3).tolist()[0],
                (r * angularVelocity).reshape(1, 3).tolist()[0]]


    test_cmd = 6

    if test_cmd == 1:  # test gym environment
        id = "gym_env:Quadruped-v0"
        env = gym.make(id)
        env.reset()
        env.step(range(12))
        env.render()
        input("press any key to continue...")
        env.close()

    elif test_cmd == 2:  # test function
        quadruped = QuadrupedEnv()
        quadruped.reset()
        for i in range(10000000):
            a = [0] * 12
            quadruped.step(a)
        quadruped.show_joint()

    elif test_cmd == 3:  # test urdf
        # load simulation environment
        physicsClient = p.connect(p.GUI)  # p.GUI for render; p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        # p.setAdditionalSearchPath(pybullet_data.getDataPath() + '/quadruped') #optionally
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        urdfFlags = p.URDF_USE_SELF_COLLISION
        quadrupedUid = p.loadURDF("./urdf/quadruped_robot/quadruped_robot.urdf", [0, 0, .0], [0, 0, 0, 1],
                                  flags=urdfFlags,
                                  useFixedBase=False)
        step_time = 1./240.
        p.setTimeStep(step_time)

        # get joints and links
        allJointIds = []
        actuatedJointIds = []
        footJointIds = []
        for j in range(p.getNumJoints(quadrupedUid)):
            info = p.getJointInfo(quadrupedUid, j)
            print(info)
            # p.changeDynamics(quadrupedUid, j, linearDamping=0, angularDamping=0)
            # jointName = info[1]
            jointType = info[2]

            allJointIds.append(j)
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                actuatedJointIds.append(j)
        footJointIds.append(3)
        footJointIds.append(7)
        footJointIds.append(11)
        footJointIds.append(15)

        # control joint
        jointAngles = np.array([0, 0.7, -1.4,
                                0, 0.7, -1.4,
                                0, 0.7, -1.4,
                                0, 0.7, -1.4 ])
        for index, jointId in enumerate(actuatedJointIds):
            print("index:",index,",",jointId)
            p.resetJointState(quadrupedUid, jointId, jointAngles[index])

        # get robot states
        ## get euler with reference to herizon
        state_robot_base = list(p.getBasePositionAndOrientation(quadrupedUid))
        state_robot_base[1] = getHerizonEulerFromQuaternion(state_robot_base[1])
        print("robot euler:", state_robot_base[1])
        state_robot_base_vel = list(p.getBaseVelocity(quadrupedUid))
        state_robot_base_vel[1] = getHerizonAngularVelocity(p.getBasePositionAndOrientation(quadrupedUid)[1],state_robot_base_vel[1])
        print("robot euler vel:", state_robot_base_vel[1])
        # get foot states
        foot_contact = []
        foot_joint_force = []
        for i in footJointIds:
            p.enableJointForceTorqueSensor(quadrupedUid, i, 1)
            foot_joint_force = p.getJointState(quadrupedUid, i)[2]
            foot_contact.append(math.sqrt(foot_joint_force[0]**2+foot_joint_force[1]**2+foot_joint_force[2]**2) > 2)
            print("joint name:", i, ",", foot_joint_force)
        print("foot_contact:", foot_contact)



        # step simulation
        for i in range(10000):
            for index, jointId in enumerate(actuatedJointIds):
                p.setJointMotorControl2(quadrupedUid,
                                        jointId,
                                        p.POSITION_CONTROL,
                                        jointAngles[index],
                                        force=10000)
            state_robot_base = list(p.getBasePositionAndOrientation(quadrupedUid))
            state_robot_base[1] = getHerizonEulerFromQuaternion(state_robot_base[1])
            # print("robot euler:",state_robot_base[1])
            state_robot_base_vel = list(p.getBaseVelocity(quadrupedUid))
            state_robot_base_vel[1] = getHerizonAngularVelocity(p.getBasePositionAndOrientation(quadrupedUid)[1],
                                                                state_robot_base_vel[1])
            # print("robot euler vel:", state_robot_base_vel[1])

            foot_contact = []
            for j in footJointIds:
                foot_joint_force = p.getJointState(quadrupedUid, j)[2]
                foot_contact.append(
                    math.sqrt(foot_joint_force[0] ** 2 + foot_joint_force[1] ** 2 + foot_joint_force[2] ** 2) > 2)
                # print("joint name:", j, ",", foot_joint_force)
            print("foot_contact:", foot_contact)


            p.stepSimulation()
            time.sleep(step_time)

        input("press any key to continue...")
        p.disconnect()

    elif test_cmd == 4:  # test urdf
        robot = QuadrupedEnv()
        robot.show_joint()
        robot.reset()
        robot.step(list(range(12)))
        input("press any key to continue...")

    elif test_cmd == 5:                         # classical control debug
        robot = QuadrupedEnv()
        robot.show_joint()
        robot.reset()
        print("input:",list([0]*12))
        observation, reward, done, info = robot.step(list([0]*12))
        robot.render()
        kp = [100,-20,0]
        kd = [10,-2,0]
        for i in range(10000):
            baseVelocityLocal = getBaseVelocityLocal(robot.quadrupedUid)
            footHold = np.asarray(baseVelocityLocal[0][0:2])*1.0/robot.step_freq*(0.5+0.15)
            foot_pos_cmd = observation[20:32]
            step_phase = observation[12:16]
            contact = observation[16:20]
            # print("baseVelocityLocal:", baseVelocityLocal)
            action = [0] * 12
            for leg in range(4):
                if contact[leg] == 1:
                    action[leg*3+0] = kp[1]*(observation[4]) + kd[1]*(observation[10]) *0
                    action[leg*3+1] = kp[0]*(observation[3]) + kd[0]*(observation[9]) *0
                elif step_phase[leg]==0:
                    action[leg * 3 + 0] = 8.0*(footHold[0] - foot_pos_cmd[leg*3+0])
                    action[leg * 3 + 1] = 8.0*(footHold[1] - foot_pos_cmd[leg*3+1])
                else:
                    action[leg * 3 + 2] = -0.05

            print("observation:",observation[12:])
            observation, reward, done, info = robot.step(action)
            robot.render()

    elif test_cmd == 6:                         # classical control debug
        robot = QuadrupedEnv()
        robot.show_joint()
        robot.reset()
        print("input:",list([0]*12))
        observation, reward, done, info = robot.step(np.asarray(list([0]*12)))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        for i in range(10000):
            action = robot.classical_control(observation)
            # action = [0]*12
            observation, reward, done, info = robot.step(np.asarray(action))
            # robot.render()

    elif test_cmd == 7:  # test torque mode
        quadruped = QuadrupedEnv()
        quadruped.reset()
        for i in range(10000000):
            # a = [0] * 12
            a = [0, 0.26, 3.45,   0, 0.26, 3.45,   0, 0.26, 3.45,   0, 0.26, 3.45]
            quadruped.step_torque(a)
        quadruped.show_joint()




