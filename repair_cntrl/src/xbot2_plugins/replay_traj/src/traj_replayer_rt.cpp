#include "traj_replayer.h"

#include "matlogger2/mat_data.h"

#include <math.h> 

void TrajReplayerRt::init_clocks()
{
    _loop_time = 0.0; // reset loop time clock
    _pause_time = 0.0;
}

void TrajReplayerRt::reset_flags()
{

    _first_run = true; // reset flag in case the plugin is run multiple times

    _approach_traj_started = false;
    _approach_traj_finished = false;
    _recompute_approach_traj = true;

    _traj_started = false;
    _traj_finished = false;

    _pause_started = false;
    _pause_finished = false;

    _replay = false;

    _sample_index = 0; // resetting samples index, in case the plugin stopped and started again

}

void TrajReplayerRt::update_clocks()
{
    // Update time(s)
    _loop_time += _plugin_dt;
    
    if(_pause_started && !_pause_finished)
    {
        _pause_time += _plugin_dt;
    }

    // Reset timers, if necessary
    if (_loop_time >= _loop_timer_reset_time)
    {
        _loop_time = _loop_time - _loop_timer_reset_time;
    }

    if(_pause_time >= _traj_pause_time)
    {
        _pause_finished = true;
        _pause_time = _pause_time - _traj_pause_time;
    }
    
}

void TrajReplayerRt::get_params_from_config()
{
    // Reading some parameters from XBot2 config. YAML file

    _mat_path = getParamOrThrow<std::string>("~mat_path"); 
    _mat_name = getParamOrThrow<std::string>("~mat_name"); 
    _dump_dir = getParamOrThrow<std::string>("~dump_dir"); 
    _stop_stiffness = getParamOrThrow<Eigen::VectorXd>("~stop_stiffness");
    _stop_damping = getParamOrThrow<Eigen::VectorXd>("~stop_damping");
    _delta_effort_lim = getParamOrThrow<double>("~delta_effort_lim");
    _approach_traj_exec_time = getParamOrThrow<double>("~approach_traj_exec_time");
    _replay_stiffness = getParamOrThrow<Eigen::VectorXd>("~replay_stiffness"); 
    _replay_damping = getParamOrThrow<Eigen::VectorXd>("~replay_damping");
    _looped_traj = getParamOrThrow<bool>("~looped_traj");
    _traj_pause_time = getParamOrThrow<double>("~traj_pause_time");
    _send_pos_ref = getParamOrThrow<bool>("~send_pos_ref");
    _send_vel_ref = getParamOrThrow<bool>("~send_vel_ref");
    _send_eff_ref = getParamOrThrow<bool>("~send_eff_ref");
    
}

void TrajReplayerRt::update_state()
{
    // "sensing" the robot
    _robot->sense();
    // Getting robot state
    _robot->getJointPosition(_q_p_meas);
    _robot->getMotorVelocity(_q_p_dot_meas);  
    _robot->getJointEffort(_tau_meas);
    
}

void TrajReplayerRt::init_dump_logger()
{

    // // Initializing logger for debugging
    MatLogger2::Options opt;
    opt.default_buffer_size = 1e6; // set default buffer size
    opt.enable_compression = true; // enable ZLIB compression
    _dump_logger = MatLogger2::MakeLogger(_dump_dir + "TrajReplayerRt", opt); // date-time automatically appended
    _dump_logger->set_buffer_mode(XBot::VariableBuffer::Mode::circular_buffer);

    _dump_logger->add("plugin_dt", _plugin_dt);

    _dump_logger->add("traj_dt_before_res", _traj_dt_before_res);
     _dump_logger->add("q_p_bf_res", _q_p_bf_res);
    _dump_logger->add("q_p_dot_bf_res", _q_p_dot_bf_res);
    _dump_logger->add("tau_bf_res", _tau_bf_res);

    _dump_logger->add("stop_stiffness", _stop_stiffness);
    _dump_logger->add("stop_damping", _stop_damping);

    _dump_logger->create("plugin_time", 1);
    _dump_logger->create("replay_stiffness", _n_jnts_robot);
    _dump_logger->create("replay_damping", _n_jnts_robot);
    _dump_logger->create("q_p_meas", _n_jnts_robot);
    _dump_logger->create("q_p_dot_meas", _n_jnts_robot);
    _dump_logger->create("tau_meas", _n_jnts_robot);
    _dump_logger->create("q_p_cmd", _n_jnts_robot);
    _dump_logger->create("q_p_dot_cmd", _n_jnts_robot);
    _dump_logger->create("tau_cmd", _n_jnts_robot);

    
    // auto jnt_names_cell = XBot::matlogger2::MatData::make_cell(_n_jnts_robot);
    // for(int i = 0; i < _n_jnts_robot; i++)
    // {
    //     jnt_names_cell[i] = _jnt_names[i];
    // }

    // _dump_logger->save("jnt_names_cell", jnt_names_cell);

}

void TrajReplayerRt::add_data2dump_logger()
{
    
    _dump_logger->add("replay_stiffness", _replay_stiffness);
    _dump_logger->add("replay_damping", _replay_damping);

    _dump_logger->add("q_p_meas", _q_p_meas);
    _dump_logger->add("q_p_dot_meas", _q_p_dot_meas);
    _dump_logger->add("tau_meas", _tau_meas);

    _dump_logger->add("plugin_time", _loop_time);

    _dump_logger->add("q_p_cmd", _q_p_cmd);
    _dump_logger->add("q_p_dot_cmd", _q_p_dot_cmd);
    _dump_logger->add("tau_cmd", _tau_cmd);

}

void TrajReplayerRt::init_nrt_ros_bridge()
{    
    ros::NodeHandle nh(getName());

    _ros = std::make_unique<RosSupport>(nh);

    /* Service server */
    _replay_now_srv = _ros->advertiseService(
        "replay_now_srvc_proxy",
        &TrajReplayerRt::on_replay_msg_rcvd,
        this,
        &_queue);

}

bool  TrajReplayerRt::on_replay_msg_rcvd(const repair_cntrl::ReplayNowRequest& req,
                    repair_cntrl::ReplayNowResponse& res)
{

    _replay = req.replay_now;

    if (req.replay_now)
    {
        jhigh().jprint(fmt::fg(fmt::terminal_color::magenta),
                   "\n Received trajectory replay signal! Finger crossed the robot won't break ;) ...\n");

        res.message = "Starting replaying of trajectory!";
        
    }

    if (!req.replay_now)
    {
        jhigh().jprint(fmt::fg(fmt::terminal_color::magenta),
                   "\n Stopping trajectory replay ...\n");

        res.message = "Stopping trajectory replay!";

        _recompute_approach_traj = true; // resetting flag so that a new approaching traj can be computed

        reset_flags();
    }

    res.success = true;

    return true; 
}

void TrajReplayerRt::load_opt_data()
{   

    _traj = plugin_utils::TrajLoader(_mat_path + _mat_name, true, 0.01, false);

    int n_traj_jnts = _traj.get_n_jnts();

    if(n_traj_jnts != _n_jnts_robot) 
    {
        jwarn("The loaded trajectory has {} joints, while the robot has {} .\n Make sure to somehow select the right components!!",
        n_traj_jnts, _n_jnts_robot);
    }

    // resample input data at the plugin frequency (for now it very crude implementation)

    _traj.resample(_plugin_dt, _q_p_ref, _q_p_dot_ref, _tau_ref); // just brute for linear interpolation for now (for safety, better to always use the same plugin_dt as the loaded trajectory)
    _traj_ref_time_vector = _traj.compute_res_times(_plugin_dt); // used for post-processing

    _traj.get_loaded_traj(_q_p_bf_res, _q_p_dot_bf_res, _tau_bf_res, _traj_dt_before_res);
}

void TrajReplayerRt::saturate_effort()
{
    int input_sign = 1; // defaults to positive sign 

    for(int i = 0; i < _n_jnts_robot; i++)
    {
        if (abs(_tau_cmd[i]) >= abs(_effort_lims[i]))
        {
            input_sign = (signbit(_tau_cmd[i])) ? -1: 1; 

            _tau_cmd[i] = input_sign * (abs(_effort_lims[i]) - _delta_effort_lim);
        }
    }
}

void TrajReplayerRt::compute_approach_traj()
{

    _approach_traj = plugin_utils::PeisekahTrans(_q_p_meas, _q_p_ref.col(_sample_index), _approach_traj_exec_time, _plugin_dt); 

    _dump_logger->add("approach_traj", _approach_traj.get_traj());

    _recompute_approach_traj = false;

}

void TrajReplayerRt::send_approach_trajectory()
{

    if (_sample_index > (_approach_traj.get_n_nodes() - 1))
    { // reached the end of the trajectory

        _approach_traj_finished = true;
        _traj_started = true; // start to publish the loaded trajectory starting from the next control loop
        _sample_index = 0; // reset publish index (will be used to publish the loaded trajectory)

        jhigh().jprint(fmt::fg(fmt::terminal_color::magenta),
                   "\n Approach trajectory finished...\n");

    }
    else
    {
        _q_p_cmd = _approach_traj.eval_at(_sample_index);

        _robot->setPositionReference(_q_p_cmd);
        
        _robot->move(); // Send commands to the robot
    }
    
}

void TrajReplayerRt::send_trajectory()
{
    // first, set the control mode and stiffness when entering the first control loop (to be used during the approach traj)
    if (_first_run)
    { // first time entering the control loop
        
        _sample_index = 0;
        _approach_traj_started = true; // flag signaling the start of the approach trajectory
        
        jhigh().jprint(fmt::fg(fmt::terminal_color::magenta),
                "\n Starting approach traj. ...\n");
    }

    // Loop again thorugh the trajectory, if it is finished and the associated flag is active
    if (_traj_finished && _looped_traj)
    { // finished publishing trajectory

        _pause_started = true;

        if (!_pause_finished)
        { // do nothing in this control loop

        }
        else 
        {
            _sample_index = 0; // reset publishing index
            _traj_finished = false; // reset flag
            
            _pause_started = false;
            _pause_finished = false;
        }
        
    }

    // When the plugin is stopped from service, recompute a transition trajectory
    // from the current state
    if (_recompute_approach_traj)
    {
        jhigh().jprint(fmt::fg(fmt::terminal_color::magenta),
                "\n Recomputing approach trajectory ...\n");
        compute_approach_traj(); // necessary if traj replay is stopped and started again from service (probably breaks rt performance)
    }

    if (_approach_traj_started && !_approach_traj_finished)
    { // still publishing the approach trajectory

        send_approach_trajectory();
    }

    // Publishing loaded traj samples
    if (_traj_started && !_traj_finished)
    { // publish current trajectory sample
        
        if (_sample_index > (_traj.get_n_nodes() - 1))
        { // reached the end of the trajectory

            
            _traj_finished = true;
            _sample_index = 0; // reset publish index (will be used to publish the loaded trajectory)
            
            jhigh().jprint(fmt::fg(fmt::terminal_color::magenta),
                "\n Finished replaying trajectory ...\n");

        }
        else
        { // send command
            
            // by default assign all commands anyway
            _q_p_cmd = _q_p_ref.col(_sample_index);
            _q_p_dot_cmd = _q_p_dot_ref.col(_sample_index);
            _tau_cmd = _tau_ref.col(_sample_index);
            
            if (_send_eff_ref)
            {
                _robot->setEffortReference(_tau_cmd);
            }

            if(_send_pos_ref)
            {  

                _robot->setPositionReference(_q_p_cmd);
            }

            if(_send_vel_ref)
            {  

                _robot->setVelocityReference(_q_p_dot_cmd);
            }
            
            _robot->setStiffness(_replay_stiffness); // necessary at each loop (for some reason)
            _robot->setDamping(_replay_damping);

            saturate_effort(); // perform input torque saturation

            _robot->move(); // Send commands to the robot

            add_data2dump_logger();
        
        }

    }

}

bool TrajReplayerRt::on_initialize()
{   
    
    _plugin_dt = getPeriodSec();

    _n_jnts_robot = _robot->getJointNum();

    // std::map<std::string, XBot::KinematicChain::Ptr> chain_map = _robot->getChainMap();

    // for (std::map<std::string, XBot::KinematicChain::Ptr>::iterator it = chain_map.begin(); it != chain_map.end(); ++it)
    // {
    //     _jnt_names.push_back(it->first);
    // }

    _robot->getEffortLimits(_effort_lims);

    init_nrt_ros_bridge();

    get_params_from_config(); // load params from yaml file
    
    load_opt_data(); // load trajectory from file (to be placed here in starting because otherwise
    // a seg fault will arise)


    return true;
}

void TrajReplayerRt::starting()
{

    init_dump_logger(); // needs to be here

    _dump_logger->add("traj_ref_time_vector", _traj_ref_time_vector);
    
    _dump_logger->add("q_p_ref", _q_p_ref);
    _dump_logger->add("q_p_dot_ref", _q_p_dot_ref);
    _dump_logger->add("tau_ref", _tau_ref);

    reset_flags();

    init_clocks(); // initialize clocks timers

    // setting the control mode to effort + velocity + stiffness + damping
    _robot->setControlMode(ControlMode::Position() + ControlMode::Velocity() + ControlMode::Effort() + ControlMode::Stiffness() + 
            ControlMode::Damping());
    _robot->setStiffness(_replay_stiffness);
    _robot->setDamping(_replay_damping);

    update_state(); // read current jnt positions and velocities
    
    compute_approach_traj(); // based on the current state, compute a smooth transition to the\\
    first trajectory position sample

    // Move on to run()
    start_completed();
    
}

void TrajReplayerRt::run()
{  

    update_state(); // read current jnt positions and velocities

    _queue.run();

    if (_replay) // only jump if a positive jump signal was received
    {
        send_trajectory();
    }
    
    update_clocks(); // last, update the clocks (loop + any additional one)

    if (_first_run == true & _replay)
    { // this is the end of the first control loop
        _first_run = false;
    }

    _sample_index++; // incrementing loop counter

}

void TrajReplayerRt::on_stop()
{
    // Read the current state
    _robot->sense();

    // Setting references before exiting
    _robot->setControlMode(ControlMode::Position() + ControlMode::Stiffness() + ControlMode::Damping());

    _robot->setStiffness(_stop_stiffness);
    _robot->setDamping(_stop_damping);
    _robot->getPositionReference(_q_p_meas); // to avoid jumps in the references when stopping the plugin
    _robot->setPositionReference(_q_p_meas);

    // Sending references
    _robot->move();

    // Destroy logger and dump .mat file (will be recreated upon plugin restart)
    _dump_logger.reset();
}

void TrajReplayerRt::stopping()
{
    _first_run = true; 

    stop_completed();
}

void TrajReplayerRt::on_abort()
{
    // Read the current state
    _robot->sense();

    // Setting references before exiting
    _robot->setControlMode(ControlMode::Position() + ControlMode::Stiffness() + ControlMode::Damping());

    _robot->setStiffness(_stop_stiffness);
    _robot->setDamping(_stop_damping);
    _robot->getPositionReference(_q_p_meas); // to avoid jumps in the references when stopping the plugin
    _robot->setPositionReference(_q_p_meas);

    // Sending references
    _robot->move();

    // Destroy logger and dump .mat file
    _dump_logger.reset();
}

void TrajReplayerRt::on_close()
{
    jinfo("Closing TrajReplayerRt");
}

XBOT2_REGISTER_PLUGIN(TrajReplayerRt, traj_replayer_rt)