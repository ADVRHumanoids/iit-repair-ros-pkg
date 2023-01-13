# iit-repair-ros-pkg

### To visualize RePair on RViz and play with its joints:

``` roslaunch repair_urdf repair_full_slider.launch ```

### To launch RePair on RViz without the joint_state_publisher_gui:

``` roslaunch repair_urdf repair_full.launch ```

### To use the robot in Gazebo:

<<<<<<< HEAD
- check that you have generated the ```repair_full.urdf``` file (if not, you can do it running the ```generate_urdf.sh``` snippet in ```${path_to_workspace_src}/iit-
repair-ros-pkg/repair_urdf/urdf/```)
- unpause Gazebo if paused
=======
``` roslaunch repair_gazebo repair_gazebo.launch ```
>>>>>>> master

