# iit-repair-ros-pkg

### To visualize RePair on RViz:

``` roslaunch repair_urdf repair_full_slider.launch '''

## steps to use RePair on Gazebo (with XBot2 integration):

- ``` roslaunch repair_gazebo repair_gazebo.launch '''

- then set the path to the XBot2 config file with ``` set_xbot2_config /${path_to_workspace_src}/iit-
repair-ros-pkg/repair_cntrl/config/repair_basic.yaml '''

- unpause Gazebo if paused

- run  ``` xbot2-core -S ''' in a terminal and ``` xbot2-gui ''' in another one. By enabling the plugin "ros_control" you should be able to control the joints of the platform using the low-level impedance controller
