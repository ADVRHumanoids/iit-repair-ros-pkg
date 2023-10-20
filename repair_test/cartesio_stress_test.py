#!/usr/bin/env python

import xbot_interface.config_options as co
import xbot_interface.xbot_interface as xb
from cartesian_interface.pyci_all import *
import numpy as np
import rospy

def add_wp(mat, time, wp_list):

    wp = pyci.WayPoint(mat, time)
    wp_list.append(wp)

def main():
    rospy.init_node('repair_stress_test')

    time = 3.0

    # robot = get_robot()
    # Rise arms and send them to the back
    # q0 = robot.getMotorPosition()

    ci = pyci.CartesianInterfaceRos()
    ci.update()

    waypoints = []

    # translation: [0.2448, 0.3962, 1.125]
    # rotation: [0.7011, 0.4426, 0.4744, 0.2959]

    # translation: [0.2448, 0.2002, 1.552]
    # rotation: [0.7231, 0.1131, 0.6803, 0.03735]

    # translation: [0.2448, -0.09736, 1.123]
    # rotation: [0.7634, 0.1623, 0.5486, 0.2997]
    ci.getTask('arm_2_tcp').disable()
    arm_start, _, _ = ci.getPoseReference('arm_1_tcp')

    add_wp(Affine3(pos=[0.2448, 0.3962, 1.125], rot=[0.7011, 0.4426, 0.4744, 0.2959]), 1*time, waypoints)
    add_wp(Affine3(pos=[0.2448, 0.2002, 1.552], rot=[0.7231, 0.1131, 0.6803, 0.03735]), 2*time, waypoints)
    add_wp(Affine3(pos=[0.2448, -0.09736, 1.123], rot=[0.7634, 0.1623, 0.5486, 0.2997]), 3*time, waypoints)
    add_wp(arm_start, 5*time, waypoints)

    ci.setWaypoints('arm_1_tcp', waypoints)
    ci.waitReachCompleted('arm_1_tcp')

    print('Motion completed!')


    # niter = 1

    # t0 = rospy.Time.now()
    #
    # while not rospy.is_shutdown():
    #     print('Started loop ', niter, ', elapsed time ', (rospy.Time.now() - t0).to_sec())
    #     niter += 1
    #
    #     q1 = np.array([-2.5, -2.5, -2.5, 0.7, -2.7, -2.7, -2.7])
    #     move_to_q(robot, q0, q1, time)
    #
    #     q2 = np.array([0.0, -1.5, 0.0, -1.0, 0.0, -1.4, 0.0])
    #     move_to_q(robot, q1, q2, time)
    #
    #     q3 = np.array([2.5, -0.5, 2.5, -2.3, 2.7, 2.0, 2.7])
    #     move_to_q(robot, q2, q3, time)
    #
    #     move_to_q(robot, q3, q0, time)
    #
    #
    #
    # print('Exiting..')


if __name__ == '__main__':
    main()