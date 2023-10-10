#!/usr/bin/env python

import xbot_interface.config_options as co
import xbot_interface.xbot_interface as xb
import numpy as np
import rospy


def get_robot():
    cfg = co.ConfigOptions()
    prefix = 'xbotcore/'
    urdf = rospy.get_param(prefix + 'robot_description')
    srdf = rospy.get_param(prefix + 'robot_description_semantic')

    cfg = co.ConfigOptions()
    cfg.set_urdf(urdf)
    cfg.set_srdf(srdf)
    cfg.generate_jidmap()
    cfg.set_string_parameter('framework', 'ROS')
    cfg.set_string_parameter('model_type', 'RBDL')
    cfg.set_bool_parameter('is_model_floating_base', True)
    robot = xb.RobotInterface(cfg)
    robot.sense()
    robot.setControlMode(xb.ControlMode.Position())

    return robot
def move_to_q(robot, q0, q1, time):

    current_time = 0.0
    dt = 0.01

    while current_time <= time:
        alpha = current_time/time
        alpha = alpha**2*(2-alpha)**2
        q = alpha*q1 + (1-alpha)*q0
        robot.setPositionReference(q)
        robot.move()
        rospy.sleep(rospy.Duration(dt))
        current_time += dt


def main():
    rospy.init_node('repair_stress_test')

    time = 3.0

    robot = get_robot()

    # Rise arms and send them to the back
    q1 = robot.getMotorPosition()

    niter = 1
    t0 = rospy.Time.now()

    while not rospy.is_shutdown():
        print('Started loop ', niter, ', elapsed time ', (rospy.Time.now() - t0).to_sec())
        niter += 1
    #
        q0 = np.array(q1)
        q1 = np.array([0.6, 2.3, 0.0, 0.0, -2.4, -1.4, -2.0])
        move_to_q(robot, q0, q1, time)
    #
    #     q0 = np.array(q1)
    #     la_q = np.array([1.4, 0.1, 2.3, -2.3, -2.4, -1.4, -2.0])
    #     q1 = la_to_robot(robot, q0, la_q, s)
    #     move_to_q(robot, q0, q1, time)
    #
    #     q0 = np.array(q1)
    #     la_q = np.array([0.6, 0.3, 0.0, -1.9, 0.0, -0.5, -2.0])
    #     q1 = la_to_robot(robot, q0, la_q, s)
    #     move_to_q(robot, q0, q1, time)
    #


    print('Exiting..')


if __name__ == '__main__':
    main()