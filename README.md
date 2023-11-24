# iit-repair-ros-pkg

### To visualize RePair on RViz and play with its joints:

``` roslaunch repair_urdf repair_full_slider.launch ```

### Steps to use RePair on Gazebo (with XBot2 integration)

- ``` roslaunch repair_gazebo repair_gazebo.launch ```

- then set the path to the XBot2 config file with ``` set_xbot2_config ${path_to_workspace_src}/iit-
repair-ros-pkg/repair_cntrl/config/repair_basic.yaml ```

- check that you have generated the ```repair_full.urdf``` file (if not, you can do it running the ```generate_urdfs.sh``` snippet in ```${path_to_workspace_src}/iit-
repair-ros-pkg/repair_urdf/urdf/```)
- unpause Gazebo if paused

- run  ``` xbot2-core -S ``` in a terminal and ``` xbot2-gui ``` in another one. By enabling the plugin "ros_control" you should be able to control the joints of the platform using the low-level impedance controller
