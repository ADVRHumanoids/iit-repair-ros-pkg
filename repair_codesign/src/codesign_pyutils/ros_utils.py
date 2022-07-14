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

import casadi as cs
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import geometry_msgs.msg
import time
from casadi_kin_dyn import pycasadi_kin_dyn as cas_kin_dyn
from copy import deepcopy

try:
    import tf as ros_tf
except ImportError:
    from . import tf_broadcaster_simple as ros_tf
    print('will not use tf publisher')

class FramePub:

    def __init__(self, node_name = "frame_publisher", anonymous = False):

        self.node_name = node_name
        self.anonymous = anonymous

        self.pose_stamped_pubs = []

        self.positions = []
        self.orientations = []
        self.topics = []
        self.base_link = []
        self.poses = []
        self.topics_map = {}

        self.frame_counter = 0

    def add_pose(self, pos, Q, topic, frame):

        self.frame_counter = self.frame_counter + 1

        self.positions.append(pos)
        self.orientations.append(Q)
        self.topics.append(topic)
        self.base_link.append(frame)
        self.topics_map[topic] = self.frame_counter - 1 # 0-based indexing

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

    def set_pose(self, frame_name, pos, Q):

        # doesn't work!!!

        self.poses[self.topics_map[frame_name]].pose.position.x = pos[0]
        self.poses[self.topics_map[frame_name]].pose.position.x = pos[1]
        self.poses[self.topics_map[frame_name]].pose.position.x = pos[2]
        self.poses[self.topics_map[frame_name]].pose.orientation.x = Q[1]
        self.poses[self.topics_map[frame_name]].pose.orientation.y = Q[2]
        self.poses[self.topics_map[frame_name]].pose.orientation.z = Q[3]
        self.poses[self.topics_map[frame_name]].pose.orientation.w = Q[0]

    def trgt_poses_pub(self):

        rospy.init_node(self.node_name, anonymous = self.anonymous)

        for i in range(len(self.topics)):
            
            self.pose_stamped_pubs.append(rospy.Publisher(self.topics[i],\
                                          PoseStamped, queue_size = 10))
        
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():

            for i in range(len(self.topics)):

                self.pose_stamped_pubs[i].publish(self.poses[i])

            rate.sleep()

    def spin(self):

        self.process = multiprocessing.Process(target = self.trgt_poses_pub)
        self.process.start()

class MarkerGen:

    def __init__(self, node_name = "marker_spawner"):

        self.node = rospy.init_node(node_name, anonymous = False)

        self.base_link_name = None
        self.marker_scale = None

        self.was_spin_called = False

        self.markers = [] # list of spawned markers
        self.servers = []
        self.markers_access_map = {}
        self.description_map = {}

        self.marker_counter = 0

        self.positions = None
        self.orientations = None

    def add_marker(self, base_link_name = "world", position = [0, 0, 0],\
                   topic_base_name=None, description = "", marker_scale = 0.5):
        
        if self.was_spin_called == False:

            self.marker_scale = marker_scale
            self.base_link_name = base_link_name
            self.marker_counter = self.marker_counter + 1

            if topic_base_name == None:

                topic_base_name = "marker_gen"
                self.servers.append(InteractiveMarkerServer(topic_base_name + \
                                                            f"{self.marker_counter}"))
            
            else:

                self.servers.append(InteractiveMarkerServer(topic_base_name))

            self.markers_access_map[topic_base_name] = self.marker_counter - 1 # 0-based indexing by convention
            self.description_map[topic_base_name] = description

            self.add_dropdown_menu()

            self.make6DofMarker(self.markers_access_map[topic_base_name],\
                                InteractiveMarkerControl.NONE, \
                                Point(position[0], position[1], position[2]),\
                                True)

            self.servers[self.markers_access_map[topic_base_name]].applyChanges()
        
        else:

            warnings.warn("Marker " + topic_base_name + \
                          " won't be added. add_marker(*) needs to be called before spin()!")

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
        
        self.positions[self.markers_access_map[feedback.marker_name]] = \
                                                    [feedback.pose.position.x, \
                                                     feedback.pose.position.y, \
                                                     feedback.pose.position.z]

        self.orientations[self.markers_access_map[feedback.marker_name]] = \
                                                    [feedback.pose.orientation.w, \
                                                     feedback.pose.orientation.x, \
                                                     feedback.pose.orientation.y, \
                                                     feedback.pose.orientation.z] # by default, real part is the first element of the quaternion
       
        if feedback.menu_entry_id == 1:
            
            print("First entry pressed\n")

        if feedback.menu_entry_id == 2:

            print("Second entry pressed\n")
            
        return true
    
    def add_dropdown_menu(self):

        self.menu_handler = MenuHandler()

        self.menu_handler.insert("Spawn other marker", \
                                 callback = self.processFeedback)
        self.menu_handler.insert("is_trgt_pose",\
                                 callback = self.processFeedback )

        # sub_menu_handle = self.menu_handler.insert( "Submenu" )
        # self.menu_handler.insert( "First Entry", parent = sub_menu_handle, callback = self.processFeedback )


    def makeBoxControl(self, msg ):

        control =  InteractiveMarkerControl()
        control.always_visible = True
        msg.controls.append( control )

        return control

    def make6DofMarker(self, marker_index, interaction_mode, position,\
                       show_6dof = True):

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.base_link_name
        int_marker.pose.position = position
        int_marker.scale = self.marker_scale

        int_marker.name = \
            list(self.markers_access_map.keys())[list(self.markers_access_map.values()).index(marker_index)]
        int_marker.description = self.description_map[int_marker.name]

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

class ReplaySol:

    def __init__(self, dt, joint_list, q_replay, frame_force_mapping=None,\
                 force_reference_frame = cas_kin_dyn.CasadiKinDyn.LOCAL,\
                 kindyn=None, srt_msg = "\nReplaying trajectory ...\n",\
                 stop_msg = "Finished.\n"):
        
        """
        Contructor
        Args:
            dt: time of replaying trajectory
            joint_list: list of joints names
            q_replay: joints position to replay
            frame_force_mapping: map between forces and frames where the force is acting
            force_reference_frame: frame w.r.t. the force is expressed. If LOCAL_WORLD_ALIGNED then forces are rotated in LOCAL frame before being published
            kindyn: needed if forces are in LOCAL_WORLD_ALIGNED

        """
        self.srt_msg = srt_msg
        self.stop_msg = stop_msg

        self.is_floating_base = False
        self.base_link = ""
        self.play_once = False
        
        self.process = []

        if frame_force_mapping is None:
            frame_force_mapping = {}
        self.dt = dt
        self.joint_list = joint_list
        self.q_replay = q_replay
        self.__sleep = 0.
        self.force_pub = []
        self.frame_force_mapping = {}
        self.slow_down_rate = 1.
        self.frame_fk = dict()

        if frame_force_mapping is not None:
            self.frame_force_mapping = deepcopy(frame_force_mapping)

        # WE CHECK IF WE HAVE TO ROTATE CONTACT FORCES:
        if force_reference_frame is cas_kin_dyn.CasadiKinDyn.LOCAL_WORLD_ALIGNED:
            if kindyn is None:
                raise Exception('kindyn input can not be None if force_reference_frame is LOCAL_WORLD_ALIGNED!')
            for frame in self.frame_force_mapping: # WE LOOP ON FRAMES
                FK = cs.Function.deserialize(kindyn.fk(frame))
                self.frame_fk[frame] = FK

                # rotate frame
                # w_all = self.frame_force_mapping[frame]
                # for k in range(0, w_all.shape[1]):
                #     w_R_f = FK(q=self.q_replay[:, k])['ee_rot']
                #     w = w_all[:, k].reshape(-1, 1)
                #     if w.shape[0] == 3:
                #         self.frame_force_mapping[frame][:, k] = np.dot(w_R_f.T,  w).T
                #     else:
                #         A = np.zeros((6, 6))
                #         A[0:3, 0:3] = A[3:6, 3:6] = w_R_f.T
                #         self.frame_force_mapping[frame][:, k] = np.dot(A,  w).T

        try:
            rospy.init_node('joint_state_publisher')
        except rospy.exceptions.ROSException as e:
            pass
        self.pub = rospy.Publisher('joint_states', JointState, queue_size=10)
        self.br = ros_tf.TransformBroadcaster()

        if self.frame_force_mapping:
            for key in self.frame_force_mapping:
                self.force_pub.append(rospy.Publisher(key+'_forces',\
                                                      geometry_msgs.msg.WrenchStamped,\
                                                      queue_size=10))


    def normalize_quaternion(q):

        def normalize(v):
            return v / np.linalg.norm(v)

        quat = normalize([q[3], q[4], q[5], q[6]])
        q[3:7] = quat[0:4]

        return q

    def publishContactForces(self, time, qk, k):
        i = 0
        for frame in self.frame_force_mapping:

            f_msg = geometry_msgs.msg.WrenchStamped()
            f_msg.header.stamp = time
            f_msg.header.frame_id = frame

            f = self.frame_force_mapping[frame][:, k]

            w_R_f = self.frame_fk[frame](q=qk)['ee_rot'].toarray()
            
            if f.shape[0] == 3:
                f = np.dot(w_R_f.T,  f).T
            else:
                A = np.zeros((6, 6))
                A[0:3, 0:3] = A[3:6, 3:6] = w_R_f.T
                f = np.dot(A,  f).T

            f_msg.wrench.force.x = f[0]
            f_msg.wrench.force.y = f[1]
            f_msg.wrench.force.z = f[2]

            if f.shape[0] == 3:
                f_msg.wrench.torque.x = 0.
                f_msg.wrench.torque.y = 0.
                f_msg.wrench.torque.z = 0.
            else:
                f_msg.wrench.torque.x = f[3]
                f_msg.wrench.torque.y = f[4]
                f_msg.wrench.torque.z = f[5]

            self.force_pub[i].publish(f_msg)
            i += 1

    def sleep(self, secs):
        '''
        Set sleep time between trajectory sequences
        Args:
            secs: time to sleep in seconds
        '''
        self.__sleep = secs

    def setSlowDownFactor(self, slow_down_factor):
        '''
        Set a slow down factor for the replay of the trajectory
        Args:
             slow_down_factor: fator to slow down
        '''
        self.slow_down_rate = 1./slow_down_factor

    def replay_sol(self):

        rate = rospy.Rate(self.slow_down_rate / self.dt)
        joint_state_pub = JointState()
        joint_state_pub.header = Header()
        joint_state_pub.name = self.joint_list

        if self.is_floating_base:
            br = ros_tf.TransformBroadcaster()
            m = geometry_msgs.msg.TransformStamped()
            m.header.frame_id = 'world'
            m.child_frame_id = self.base_link

        nq = np.shape(self.q_replay)[0]
        ns = np.shape(self.q_replay)[1]

        if not self.play_once:

            while not rospy.is_shutdown():
                
                print(self.srt_msg)

                k = 0
                for qk in self.q_replay.T:

                    t = rospy.Time.now()

                    if self.is_floating_base:
                        qk = self.normalize_quaternion(qk)

                        m.transform.translation.x = qk[0]
                        m.transform.translation.y = qk[1]
                        m.transform.translation.z = qk[2]
                        m.transform.rotation.x = qk[3]
                        m.transform.rotation.y = qk[4]
                        m.transform.rotation.z = qk[5]
                        m.transform.rotation.w = qk[6]

                        br.sendTransform((m.transform.translation.x, m.transform.translation.y, m.transform.translation.z),
                                        (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z,
                                        m.transform.rotation.w),
                                        t, m.child_frame_id, m.header.frame_id)

                    
                    joint_state_pub.header.stamp = t
                    joint_state_pub.position = qk[7:nq] if self.is_floating_base else qk
                    joint_state_pub.velocity = []
                    joint_state_pub.effort = []
                    self.pub.publish(joint_state_pub)
                    if self.frame_force_mapping:
                        if k != ns-1:
                            self.publishContactForces(t, qk, k)
                    rate.sleep()
                    k += 1
                if self.__sleep > 0.:
                    time.sleep(self.__sleep)
                    print(self.stop_msg)
        else:

            print(self.srt_msg)

            k = 0
            for qk in self.q_replay.T:

                t = rospy.Time.now()

                if self.is_floating_base:
                    qk = self.normalize_quaternion(qk)

                    m.transform.translation.x = qk[0]
                    m.transform.translation.y = qk[1]
                    m.transform.translation.z = qk[2]
                    m.transform.rotation.x = qk[3]
                    m.transform.rotation.y = qk[4]
                    m.transform.rotation.z = qk[5]
                    m.transform.rotation.w = qk[6]

                    br.sendTransform((m.transform.translation.x, m.transform.translation.y, m.transform.translation.z),
                                    (m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z,
                                    m.transform.rotation.w),
                                    t, m.child_frame_id, m.header.frame_id)

                
                joint_state_pub.header.stamp = t
                joint_state_pub.position = qk[7:nq] if self.is_floating_base else qk
                joint_state_pub.velocity = []
                joint_state_pub.effort = []
                self.pub.publish(joint_state_pub)
                if self.frame_force_mapping:
                    if k != ns-1:
                        self.publishContactForces(t, qk, k)

                rate.sleep()
                k += 1

            if self.__sleep > 0.:
                
                time.sleep(self.__sleep)
                print(self.stop_msg)


    def replay(self, is_floating_base = True, base_link = 'base_link', play_once = False, is_blocking = True):

        self.is_floating_base = is_floating_base
        self.base_link = base_link
        self.play_once = play_once

        if not is_blocking:
            self.process = multiprocessing.Process(target = self.replay_sol) # run replayer on different process to avoid blocking main code
            self.process.start()
            
        self.replay_sol()