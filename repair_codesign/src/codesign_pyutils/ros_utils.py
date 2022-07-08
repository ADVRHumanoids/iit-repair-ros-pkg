from sympy import true

import rospy
import rospkg

import numpy as np

from geometry_msgs.msg import PoseStamped

from codesign_pyutils import math_utils

import multiprocessing

import subprocess

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster

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

class GenPosesFromRViz:

    def __init__(self, launch_name, package_name, base_link_name = "world", marker_scale = 0.5, position = [0, 0, 0]):

        self.node = rospy.init_node("marker_spawner", anonymous = True)

        self.launch_name = launch_name 
        self.package_name = package_name
        self.base_link_name = base_link_name
        self.marker_scale = marker_scale

        # self.launch_rviz()

        self.markers = [] # list of spawned markers
        self.servers = []
        self.marker_counter = 0

        self.position = None
        self.orientation = None 

        self.br = TransformBroadcaster()

        self.add_marker(position, "repair_poses")

        rospy.spin()

    def add_marker(self, position, topic_base_name = "repair_poses"):

        self.marker_counter = self.marker_counter + 1

        self.servers.append(InteractiveMarkerServer(topic_base_name + f"{self.marker_counter}"))

        self.add_dropdown_menu()

        self.make6DofMarker(self.marker_counter - 1, InteractiveMarkerControl.NONE, Point(position[0], position[1], position[2]), True)

        self.servers[self.marker_counter - 1].applyChanges()


    def getPose(self):

        return self.position, self.orientation

    def processFeedback(self, feedback):

        self.position = np.array([feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z])
        self.orientation = np.array([feedback.pose.orientation.w, feedback.pose.orientation.x, feedback.pose.orientation.y, feedback.pose.orientation.z]) # by default, real part is the first element of the quaternion

        if feedback.menu_entry_id == 1:
            
            print("sdsdfs")

        if feedback.menu_entry_id == 2:

            print("sdsdfs")
            
        return true
    
    def add_dropdown_menu(self):

        self.menu_handler = MenuHandler()

        self.menu_handler.insert( "Spawn other marker", callback = self.processFeedback )
        self.menu_handler.insert( "is_trgt_pose", callback = self.processFeedback )
        # sub_menu_handle = self.menu_handler.insert( "Submenu" )
        # self.menu_handler.insert( "First Entry", parent = sub_menu_handle, callback = self.processFeedback )


    def makeBoxControl(self, msg ):

        control =  InteractiveMarkerControl()
        control.always_visible = True
        msg.controls.append( control )

        return control

    def make6DofMarker(self, srv_index, interaction_mode, position, show_6dof = True):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_link_name
        int_marker.pose.position = position
        int_marker.scale = self.marker_scale

        int_marker.name = "repair_pose_marker"
        int_marker.description = ""

        # insert a box
        self.makeBoxControl(int_marker)
        int_marker.controls[0].interaction_mode = interaction_mode
        
        if show_6dof: 
            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "rotate_x"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 1
            control.orientation.y = 0
            control.orientation.z = 0
            control.name = "move_x"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS

            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "rotate_z"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 1
            control.orientation.z = 0
            control.name = "move_z"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "rotate_y"
            control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            int_marker.controls.append(control)

            control = InteractiveMarkerControl()
            control.orientation.w = 1
            control.orientation.x = 0
            control.orientation.y = 0
            control.orientation.z = 1
            control.name = "move_y"
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            int_marker.controls.append(control)

        self.servers[srv_index].insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.servers[srv_index], int_marker.name )

        self.markers.append(int_marker)

    def launch_rviz(self):

        rviz_window = subprocess.Popen(["roslaunch", self.package_name, self.launch_name + ".launch"])



