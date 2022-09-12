
#!/usr/bin/python3 

### A minimal node for exploiting the my_ell_traj_srvc service ###

import rospy

from repair_cntrl.srv import 

from xbot_msgs.srv import PluginStatus, PluginStatusRequest

rospy.init_node("start_replay_plugin")

start_plugin_proxy= rospy.ServiceProxy("/xbotcore/traj_replayer/switch", PluginStatus)

start_plugin_proxy.wait_for_service()

req = PluginStatusRequest()

req.data = True

res = start_plugin_proxy(req)

print(res)
