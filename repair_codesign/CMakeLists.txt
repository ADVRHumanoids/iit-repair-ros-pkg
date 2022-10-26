cmake_minimum_required(VERSION 3.0.2)
project(repair_codesign)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  test_results
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS src/replay_sol.py src/run_1st_step_opt_on_workstation.py  
            src/run_2nd_3rd_step_opt_on_workstation.py 
            src/run_codesign_pìpeline.py 
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY src/codesign_pyutils  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
