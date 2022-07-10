from sympy import true

import rospy
import rospkg

import numpy as np

from geometry_msgs.msg import PoseStamped

from codesign_pyutils import math_utils

import multiprocessing

from interactive_markers.interactive_marker_server import *
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from geometry_msgs.msg import Point
from tf.broadcaster import TransformBroadcaster

import warnings

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

class MarkerGen:

    def __init__(self):

        self.node = rospy.init_node("marker_spawner", anonymous = True)

        self.base_link_name = None
        self.marker_scale = None

        self.was_spin_called = False

        self.markers = [] # list of spawned markers
        self.servers = []
        self.markers_access_map = {}

        self.marker_counter = 0

        self.positions = None
        self.orientations = None

        self.br = TransformBroadcaster()

    def add_marker(self, base_link_name = "world", position = [0, 0, 0], topic_base_name = None, marker_scale = 0.5):
        
        if self.was_spin_called == False:

            self.marker_scale = marker_scale
            self.base_link_name = base_link_name
            self.marker_counter = self.marker_counter + 1

            if topic_base_name == None:

                topic_base_name = "marker_gen"
                self.servers.append(InteractiveMarkerServer(topic_base_name + f"{self.marker_counter}"))
            
            else:

                self.servers.append(InteractiveMarkerServer(topic_base_name))

            self.markers_access_map[topic_base_name] = self.marker_counter - 1 # 0-based indexing by convention

            self.add_dropdown_menu()

            self.make6DofMarker(self.markers_access_map[topic_base_name], InteractiveMarkerControl.NONE, Point(position[0], position[1], position[2]), True)

            self.servers[self.markers_access_map[topic_base_name]].applyChanges()
        
        else:

            warnings.warn("Marker " + topic_base_name + " won't be added. add_marker(*) needs to be called before spin()!")

    def spin(self):
        
        self.positions = [None] * len(self.markers)
        self.orientations = [None] * len(self.markers)

        self.was_spin_called = True # once spin is called, markers cannot added anymore

        self.process = multiprocessing.Process(target = rospy.spin) # spin node on separate process to avoid blocking the main program
        self.process.start()

    def getPose(self, marker_topic_name):
        
        marker_index = self.markers_access_map[marker_topic_name]

        return self.positions[marker_index], self.orientations[marker_index]

    def processFeedback(self, feedback):
        
        self.positions[self.markers_access_map[feedback.marker_name]] = [feedback.pose.position.x, feedback.pose.position.y, feedback.pose.position.z]
        self.orientations[self.markers_access_map[feedback.marker_name]] = [feedback.pose.orientation.w, feedback.pose.orientation.x, feedback.pose.orientation.y, feedback.pose.orientation.z] # by default, real part is the first element of the quaternion
       
        if feedback.menu_entry_id == 1:
            
            print("First entry pressed\n")

        if feedback.menu_entry_id == 2:

            print("Second entry pressed\n")
            
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

    def make6DofMarker(self, marker_index, interaction_mode, position, show_6dof = True):

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_link_name
        int_marker.pose.position = position
        int_marker.scale = self.marker_scale

        int_marker.name = list(self.markers_access_map.keys())[list(self.markers_access_map.values()).index(marker_index)]
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

        self.servers[marker_index].insert(int_marker, self.processFeedback)
        self.menu_handler.apply(self.servers[marker_index], int_marker.name )

        self.markers.append(int_marker)



