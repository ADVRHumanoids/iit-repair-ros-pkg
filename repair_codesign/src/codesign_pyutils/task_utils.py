import numpy as np

from codesign_pyutils.tasks import TaskGen

from codesign_pyutils.miscell_utils import gen_y_sampling

from termcolor import colored

from horizon.transcriptions.transcriptor import Transcriptor

from horizon.solvers import solver

def gen_task_copies(weight_global_manip: np.double, weight_class_manip: np.double, 
                    filling_nodes: int,
                    wrist_offset: np.double, 
                    y_samples: int, y_ub: np.double, 
                    urdf_path: str,
                    t_exec: np.double,
                    rot_err_epsi: np.double,
                    use_classical_man = False,
                    sliding_wrist = False, 
                    coll_path = "", 
                    is_second_lev_opt = False):

    
    y_sampling = gen_y_sampling(y_samples, y_ub)

    right_arm_picks = [True] * len(y_sampling)

    for i in range(len(y_sampling)):
        
        if y_sampling[i] <= 0 : # on the right
            
            right_arm_picks[i] = True
        
        else:

            right_arm_picks[i] = False
            
    # initialize problem task
    task = TaskGen(filling_n_nodes = filling_nodes, \
                    is_sliding_wrist = sliding_wrist,\
                    sliding_wrist_offset = wrist_offset,\
                    coll_yaml_path = coll_path)

    task.add_tasks(y_sampling, right_arm_picks)

    # initialize problem
    task.init_prb(urdf_path,
                    weight_glob_man = weight_global_manip, weight_class_man = weight_class_manip,\
                    tf_single_task = t_exec)

    print(colored("Task node list: " + str(task.nodes_list), "magenta"))
    print(colored("Task list: " + str(task.task_list), "magenta"))
    print(colored("Task names: " + str(task.task_names), "magenta"))
    print(colored("Task dict: " + str( task.task_dict), "magenta"))
    print(colored("Total employed nodes: " + str(task.total_nnodes), "magenta"))
    print(colored("Number of added subtasks:" + str(task.n_of_tasks) + "\n", "magenta"))

    # set constraints and costs
    task.setup_prb(rot_err_epsi, is_classical_man = use_classical_man,
                    is_second_lev_opt=is_second_lev_opt)

    return task

def gen_slvr_copies(task: TaskGen,
                    solver_type: str,
                    transcription_method: str, 
                    transcription_opts: dict, 
                    slvr_opt: dict):

    if solver_type != "ilqr":

        Transcriptor.make_method(transcription_method,\
                                task.prb,\
                                transcription_opts)
    
    ## Creating the solver
    slvr = solver.Solver.make_solver(solver_type, task.prb, slvr_opt)

    if solver_type == "ilqr":

        slvr.set_iteration_callback()

    return slvr

def compute_cl_man_list(task: TaskGen, q: np.ndarray):

    Jl = task.jac_arm_l( q = q )["J"]
    Jr = task.jac_arm_r( q = q)["J"]
    cl_man_ltrasl, cl_man_lrot, cl_man_ltot = task.compute_cl_man(Jl)
    cl_man_rtrasl, cl_man_rrot, cl_man_rtot = task.compute_cl_man(Jr)

    return cl_man_ltrasl, cl_man_lrot, cl_man_ltot, \
            cl_man_rtrasl, cl_man_rrot, cl_man_rtot   
