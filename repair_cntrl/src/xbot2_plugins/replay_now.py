#!/usr/bin/python3 

### A minimal node for exploiting the my_ell_traj_srvc service ###

import rospy

from repair_cntrl.srv import ReplayNow, ReplayNowRequest

rospy.init_node("replay_now_repair_srvc")

replay_proxy= rospy.ServiceProxy("/traj_replayer_rt/replay_now_srvc_proxy", ReplayNow)

replay_proxy.wait_for_service()

req = ReplayNowRequest()

req.replay_now = True

res = replay_proxy(req)

print(res)