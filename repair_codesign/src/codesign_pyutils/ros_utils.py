import rospy

from geometry_msgs.msg import PoseStamped

from codesign_pyutils import math_utils

import multiprocessing

class PoseStampedPub:
    def __init__(self, node_name, anonymous = False):

        self.node_name = node_name
        self.anonymous = anonymous

        self.pose_stamped_pubs = []

        self.positions = []
        self.rots = []
        self.topics = []
        self.base_link = []
        self.poses = []

    def add_pose(self, pos, rot, topic, frame):

        self.positions.append(pos)
        self.rots.append(rot)
        self.topics.append(topic)
        self.base_link.append(frame)

        Q = math_utils.rot2quat(rot)

        pose = PoseStamped()
        pose.header.frame_id = frame
        pose.pose.position.x = pos[0]
        pose.pose.position.y = pos[1]
        pose.pose.position.z = pos[2]
        pose.pose.orientation.x = Q[1]
        pose.pose.orientation.y = Q[2]
        pose.pose.orientation.z = Q[3]
        pose.pose.orientation.w = Q[0]

        self.poses.append(pose)

    def trgt_poses_pub(self):

        rospy.init_node(self.node_name, anonymous = self.anonymous)

        for i in range(len(self.topics)):
            
            self.pose_stamped_pubs.append(rospy.Publisher(self.topics[i], PoseStamped, queue_size = 10))
        
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():

            for i in range(len(self.topics)):

                self.pose_stamped_pubs[i].publish(self.poses[i])

            rate.sleep()

    def pub_frames(self):

        self.process = multiprocessing.Process(target = self.trgt_poses_pub)
        self.process.start()