from codesign_pyutils.dump_utils import SolDumper
from codesign_pyutils.load_utils import LoadSols
from codesign_pyutils.math_utils import compute_man_index
from codesign_pyutils.task_utils import gen_task_copies, compute_ms_cl_man
from codesign_pyutils.miscell_utils import correct_list, extract_q_design
from codesign_pyutils.clustering_utils import Clusterer

from horizon.utils import mat_storer

import os

from termcolor import colored

import numpy as np

import rospkg

import subprocess

import matplotlib.pyplot as plt

def compute_man_cost(task_node_list: list, 
                    q_dot: list, man_weight = None):

  # costs are computed by Horizon summing costs over each problem node

  if man_weight is None:

    man_weight = 1.0

  if type(q_dot[0]) != np.ndarray:

    raise Exception("compute_man_cost: q_dot should be a list of ndarrays!")

  man_cost = [0.0] * len(q_dot)

  for ms_idx in range(len(q_dot)):

    for task in range(len(task_node_list)):

      for node in task_node_list[task]:
      
        if node != task_node_list[task][-1]: # transition between tasks is not considered (tasks are optimizer for in parallel)
          
          man_cost[ms_idx] = man_cost[ms_idx] + man_weight * np.sum(np.square(q_dot[ms_idx][:, node]))

  return man_cost

class PostProcL1:

    def __init__(self, load_path, 
                additional_info_pattern="info", 
                l1_dirname="first_level", 
                dump_dirname="l1_postproc", 
                opt_dirname="opt", 
                fail_dirname="failed", 
                cl_man_post_proc = False):

        self.cl_man_post_proc = cl_man_post_proc # whether to compute and print
        # also cl. man. postproc. info (can take time on large n. samples)

        self._load_path = load_path + "/" + l1_dirname + "/"
        self._dump_path = load_path + "/" + dump_dirname

        # useful paths
        rospackage = rospkg.RosPack() # Only for taking the path to the leg package
        urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
        urdf_name = "repair_full"
        self._urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
        self._xacro_full_path = urdfs_path + "/" + urdf_name + ".urdf.xacro"
        coll_yaml_name = "arm_coll.yaml"
        self._coll_yaml_path = rospackage.get_path("repair_urdf") + "/config/" + coll_yaml_name

        print(colored("\n--->Initializing first level postprocess object from folder \"" + 
                        self._load_path + "\n", 
                        "blue"))

        # Loading solutions
        self._sol_loader = LoadSols(self._load_path,
                                opt_dir_name=opt_dirname, 
                                fail_dir_name=fail_dirname, 
                                additional_info_pattern=additional_info_pattern)
        
        self._opt_data = self._sol_loader.opt_data
        self._fail_data = self._sol_loader.fail_data
        self._n_opt_sols = self._sol_loader.n_opt_sols
        self._n_fail_sols = self._sol_loader.n_fail_sols

        self._prb_info_data = self._sol_loader.task_info_data

        self._dump_vars = {}
        
        self.__read_info_data()
        self.__read_opt_data()

        self._man_cost = self.__get_man_cost()
        self._man_index = self.__get_man_index(self._man_cost)

        self.__compute_solver_stats()

        if self.cl_man_post_proc:

            self.__gen_urdf()
            print(colored("\nGenerating task copy...\n", "magenta"))
            self._task_copy = gen_task_copies(self._man_w_base, self._class_man_w_base,
                                            self._filling_nnodes, 
                                            self._wrist_off,
                                            self._ny_sampl,
                                            self._y_sampl_ub, 
                                            self._urdf_full_path, 
                                            self._t_exec_task, 
                                            self._rot_error_epsi, 
                                            self._is_class_man, 
                                            self._is_sliding_wrist, 
                                            self._coll_yaml_path, 
                                            is_second_lev_opt=False)

            self._cl_man_llist, self._cl_man_rlist = self.__get_cl_man_list()

            self.__compute_cl_man_data()

    def __read_info_data(self):

        # reading other stuff from add info data
        self._task_dt = self._prb_info_data["dt"][0][0]
        self._filling_nnodes = self._prb_info_data["filling_nodes"][0][0]
        self._integrator = self._prb_info_data["integrator"][0]
        self._ig_seed = self._prb_info_data["ig_seed"][0][0]
        self._max_retry_n = self._prb_info_data["max_retry_n"][0][0]
        self._ms_trgt = self._prb_info_data["n_msrt_trgt"][0][0]
        self._ny_sampl = self._prb_info_data["n_y_samples"][0]
        self._y_sampl_ub = self._prb_info_data["y_sampl_ub"][0]
        self._nodes_list = correct_list(self._prb_info_data["nodes_list"])
        self._n_int = self._nodes_list[-1][-1]
        self._proc_sol_divs = correct_list(self._prb_info_data["proc_sol_divs"])
        self._rot_error_epsi = self._prb_info_data["rot_error_epsi"][0][0]

        self._slvr_opts_tol = self._prb_info_data["slvr_opts"]["ipopt.tol"][0][0][0][0]
        self._slvr_opts_maxiter = self._prb_info_data["slvr_opts"]["ipopt.max_iter"][0][0][0][0]
        self._slvr_opts_cnstr_viol = self._prb_info_data["slvr_opts"]["ipopt.constr_viol_tol"][0][0][0][0]
        self._slvr_opts_print_l = self._prb_info_data["slvr_opts"]["ipopt.print_level"][0][0][0][0]
        self._slvr_opts_lin_solv = self._prb_info_data["slvr_opts"]["ipopt.linear_solver"][0][0][0]
        self._solver_type = self._prb_info_data["solver_type"][0]

        self._right_arm_picks = self._prb_info_data["right_arm_picks"][0]
        self._is_biman_pick = bool(self._prb_info_data["is_biman_pick"][0][0])
        self._is_in_place_flip = bool(self._prb_info_data["is_in_place_flip"][0][0])
        self._t_exec_task = self._prb_info_data["t_exec_task"][0][0]
        self._task_base_nnodes = self._prb_info_data["task_base_nnodes"]
        self._task_dict = self._prb_info_data["tasks_dict"]
        self._tasks_list = self._prb_info_data["tasks_list"]
        self._transcription_method = self._prb_info_data["transcription_method"][0]
        self._unique_id = self._prb_info_data["unique_id"][0]
        self._is_class_man = bool(self._prb_info_data["use_classical_man"][0][0])
        self._class_man_w_base = self._prb_info_data["w_clman_base"][0][0]
        self._class_man_w_a = self._prb_info_data["w_clman_actual"][0][0]
        self._man_w_base = self._prb_info_data["w_man_base"][0][0]
        self._man_w_a = self._prb_info_data["w_man_actual"][0][0]
        self._wrist_off = self._prb_info_data["sliding_wrist_offset"][0][0]
        self._is_sliding_wrist = bool(self._prb_info_data["is_sliding_wrist"][0][0])

    def __read_opt_data(self):

        # reading stuff from opt data
        self._coll_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("coll", self._coll_cnstrnt_data,
                                self._opt_data, 
                                is_dict = True)

        self._lambd_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("lambd", self._lambd_cnstrnt_data,
                                self._opt_data, 
                                is_dict = True)

        self._pos_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("pos", self._pos_cnstrnt_data, 
                                self._opt_data,
                                is_dict = True)

        self._rot_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("rot", self._rot_cnstrnt_data, 
                                self._opt_data,
                                is_dict = True)

        self._ws_lim_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("keep", self._ws_lim_cnstrnt_data, 
                                self._opt_data,
                                is_dict = True)

        self._codes_simmetry_cnstrt = [{}] * len(self._opt_data)
        self.__get_data_matching("same", self._codes_simmetry_cnstrt, 
                                self._opt_data,
                                is_dict = True)

        self._mult_shoot_cnstrnt_data = [None] * len(self._opt_data)
        self.__get_data_matching("multiple_shooting", self._mult_shoot_cnstrnt_data, 
                                self._opt_data,
                                patter_is_varname = True)

        self._codes_var_cnstr = [None] * len(self._opt_data)
        self.__get_data_matching("single_var", self._codes_var_cnstr, 
                                self._opt_data,
                                is_dict = True)

        self._niters2sol = [-1] * len(self._opt_data)
        self.__get_data_matching("n_iter2sol", self._niters2sol, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._opt_costs = [-1] * len(self._opt_data)
        self.__get_data_matching("opt_cost", self._opt_costs,
                                self._opt_data, 
                                patter_is_varname = True, 
                                is_scalar = True)

        self._q = [None] * len(self._opt_data)
        self.__get_data_matching("q", self._q, 
                                self._opt_data,
                                patter_is_varname = True)
        
        self._q_ig = [None] * len(self._opt_data)
        self.__get_data_matching("q_ig", self._q_ig,
                                self._opt_data, 
                                patter_is_varname = True)
                                    
        self._q_dot = [None] * len(self._opt_data)
        self.__get_data_matching("q_dot", self._q_dot,
                                self._opt_data, 
                                patter_is_varname = True)
        
        self._q_dot_ig = [None] * len(self._opt_data)
        self.__get_data_matching("q_dot_ig", self._q_dot_ig, 
                                self._opt_data,
                                patter_is_varname = True)

        self._ms_indxs = [None] * len(self._opt_data)
        self.__get_data_matching("multistart_index", self._ms_indxs, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._sol_times = [None] * len(self._opt_data)
        self.__get_data_matching("solution_time", self._sol_times, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._solve_failed = [None] * len(self._opt_data)
        self.__get_data_matching("solve_failed", self._solve_failed,
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._trial_idxs = [None] * len(self._opt_data)
        self.__get_data_matching("trial_index", self._trial_idxs, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._u_opt = [None] * len(self._opt_data)
        self.__get_data_matching("u_opt", self._u_opt,
                                self._opt_data, 
                                patter_is_varname = True)

        self._x_opt = [None] * len(self._opt_data)
        self.__get_data_matching("x_opt", self._x_opt, 
                                self._opt_data,
                                patter_is_varname = True)

        self._sols_run_ids = [None] * len(self._opt_data)
        self.__get_data_matching("run_id", self._sols_run_ids, 
                                self._opt_data,
                                patter_is_varname = True)

        self._q_design = extract_q_design(self._q)

    def __compute_solver_stats(self):
        
        self._tot_sol_time = np.sum(np.array(self._sol_times))
        self._avrg_sol_time = np.mean(np.array(self._sol_times))
        self._max_sol_time = np.max(np.array(self._sol_times))
        self._min_sol_time = np.min(np.array(self._sol_times))
        self._rmse_sol_time = self.__rmse(self._avrg_sol_time, self._sol_times)

        self._tot_niters2sol = np.sum(np.array(self._niters2sol))
        self._avrg_niters2sol = np.mean(np.array(self._niters2sol))
        self._max_niters2sol = np.max(np.array(self._niters2sol))
        self._min_niters2sol = np.min(np.array(self._niters2sol))
        self._rmse_niters2sol = self.__rmse(self._avrg_niters2sol, self._niters2sol)

        self._tot_trial_idxs = np.sum(np.array(self._trial_idxs))
        self._avrg_trial_idxs = np.mean(np.array(self._trial_idxs))
        self._max_trial_idxs = np.max(np.array(self._trial_idxs))
        self._min_trial_idxs = np.min(np.array(self._trial_idxs))
        self._rmse_trial_idxs = self.__rmse(self._avrg_trial_idxs, self._trial_idxs)

        self._avrg_man_cost = np.mean(np.array(self._man_cost))
        self._max_man_cost = np.max(np.array(self._man_cost))
        self._min_man_cost = np.min(np.array(self._man_cost))
        self._rmse_man_cost = self.__rmse(self._avrg_man_cost, self._man_cost)

        self._avrg_man_index = np.mean(np.array(self._man_index))
        self._max_man_index = np.max(np.array(self._man_index))
        self._min_man_index = np.min(np.array(self._man_index))
        self._rmse_man_index = self.__rmse(self._avrg_man_index, self._man_index)

        self._avrg_opt_costs = np.mean(np.array(self._opt_costs))
        self._max_opt_costs = np.max(np.array(self._opt_costs))
        self._min_opt_costs = np.min(np.array(self._opt_costs))
        self._rmse_opt_costs = self.__rmse(self._avrg_opt_costs, self._opt_costs)

    def __compute_cl_man_data(self):

        self._avrg_cl_man_llist, self._avrg_cl_man_rlist = self.__compute_avrg_cl_man_list()
        self._max_cl_man_llist, self._max_cl_man_rlist = self.__compute_max_cl_man_list()
        self._min_cl_man_llist, self._min_cl_man_rlist = self.__compute_min_cl_man_list()
        self._rmse_cl_man_llist, self._rmse_cl_man_rlist = self.____compute_rmse_cl_man_list()

        self._2avrg_cl_man_l_trasl = np.sum(np.array(self._avrg_cl_man_llist[0]))/ self._n_opt_sols
        self._2avrg_cl_man_l_rot =  np.sum(np.array(self._avrg_cl_man_llist[1]))/ self._n_opt_sols
        self._2avrg_cl_man_r_trasl = np.sum(np.array(self._avrg_cl_man_rlist[0]))/ self._n_opt_sols
        self._2avrg_cl_man_r_rot = np.sum(np.array(self._avrg_cl_man_rlist[1]))/ self._n_opt_sols
        self._2max_cl_man_l_trasl = np.max(np.array(self._max_cl_man_llist[0]))
        self._2max_cl_man_l_rot = np.max(np.array(self._max_cl_man_llist[1]))
        self._2max_cl_man_r_trasl = np.max(np.array(self._max_cl_man_rlist[0]))
        self._2max_cl_man_r_rot = np.max(np.array(self._max_cl_man_rlist[1]))
        self._2min_cl_man_l_trasl = np.min(np.array(self._min_cl_man_llist[0]))
        self._2min_cl_man_l_rot = np.min(np.array(self._min_cl_man_llist[1]))
        self._2min_cl_man_r_trasl = np.min(np.array(self._min_cl_man_rlist[0]))
        self._2min_cl_man_r_rot = np.min(np.array(self._min_cl_man_rlist[1]))
        self._avrg_rmse_cl_man_l_trasl = np.sum(np.array(self._rmse_cl_man_llist[0]))/ self._n_opt_sols
        self._avrg_rmse_cl_man_l_rot = np.sum(np.array(self._rmse_cl_man_llist[1]))/ self._n_opt_sols
        self._avrg_rmse_cl_man_r_trasl = np.sum(np.array(self._rmse_cl_man_rlist[0]))/ self._n_opt_sols
        self._avrg_rmse_cl_man_r_rot = np.sum(np.array(self._rmse_cl_man_rlist[1]))/ self._n_opt_sols

    def __gen_urdf(self):

        sliding_wrist_command = "is_sliding_wrist:=" + "true"
        show_softhand_command = "show_softhand:=" + "true"
        show_coll_command = "show_coll:=" + "true"

        try:
        
            # print(sliding_wrist_command)
            xacro_gen = subprocess.check_call(["xacro",\
                                            self._xacro_full_path, \
                                            sliding_wrist_command, \
                                            show_softhand_command, \
                                            show_coll_command, \
                                            "-o", 
                                            self._urdf_full_path])

        except:

            print(colored('Failed to generate URDF.', "red"))                             

    def __rmse(self, ref, vals):
        
        rmse = 0.0
        sum_sqrd = 0.0
        for i in range(len(vals)):
            sum_sqrd = sum_sqrd + (vals[i] - ref)**2

        rmse = np.sqrt(sum_sqrd / len(vals))

        return rmse

    def __get_man_index(self, man_cost):

        man_ind = compute_man_index(man_cost, self._q_dot[0].shape[1])

        return man_ind

    def __get_man_cost(self):

        man_cost_raw = compute_man_cost(self._nodes_list, self._q_dot, self._man_w_a)

        man_cost = [man_cost_raw[i] for i in range(len(man_cost_raw))] # correction for scalar data

        return man_cost

    def __get_cl_man_list(self):

        cl_man_llist = []
        cl_man_rlist = []

        for ms in range(len(self._q)):

            cl_man_lft, cl_man_rght  = compute_ms_cl_man(self._q[ms], self._nodes_list, self._task_copy)

            cl_man_llist.append(cl_man_lft)
            cl_man_rlist.append(cl_man_rght)

        return cl_man_llist, cl_man_rlist

    def __compute_avrg_cl_man_list(self):

        avrg_cl_man_list_l_trasl = []
        avrg_cl_man_list_l_rot = []
        avrg_cl_man_list_r_trasl = []
        avrg_cl_man_list_r_rot = []

        for ms in range(len(self._q)):

            n_sampl = self._q[ms].shape[1]

            avrg_cl_man_list_l_trasl.append(np.sum(self._cl_man_llist[ms][0, :]) / n_sampl) 
            avrg_cl_man_list_l_rot.append(np.sum(self._cl_man_llist[ms][1, :]) / n_sampl)

            avrg_cl_man_list_r_trasl.append(np.sum(self._cl_man_rlist[ms][0, :]) / n_sampl) 
            avrg_cl_man_list_r_rot.append(np.sum(self._cl_man_rlist[ms][1, :]) / n_sampl)

        avrg_cl_man_list_l = [avrg_cl_man_list_l_trasl, avrg_cl_man_list_l_rot]
        avrg_cl_man_list_r = [avrg_cl_man_list_r_trasl, avrg_cl_man_list_r_rot]

        return avrg_cl_man_list_l, avrg_cl_man_list_r

    def ____compute_rmse_cl_man_list(self):

        rmse_cl_man_list_l_trasl = []
        rmse_cl_man_list_l_rot = []
        rmse_cl_man_list_r_trasl = []
        rmse_cl_man_list_r_rot = []

        for ms in range(len(self._q)):

            n_sampl = self._q[ms].shape[1]

            rmse_cl_man_list_l_trasl.append(self.__rmse(self._avrg_cl_man_llist[0][ms], self._cl_man_llist[ms][0, :])) 
            rmse_cl_man_list_l_rot.append(self.__rmse(self._avrg_cl_man_llist[1][ms], self._cl_man_llist[ms][1, :]))

            rmse_cl_man_list_r_trasl.append(self.__rmse(self._avrg_cl_man_rlist[0][ms], self._cl_man_rlist[ms][0, :])) 
            rmse_cl_man_list_r_rot.append(self.__rmse(self._avrg_cl_man_rlist[1][ms], self._cl_man_rlist[ms][1, :]))

        rmse_cl_man_list_l = [rmse_cl_man_list_l_trasl, rmse_cl_man_list_l_rot]
        rmse_cl_man_list_r = [rmse_cl_man_list_r_trasl, rmse_cl_man_list_r_rot]

        return rmse_cl_man_list_l, rmse_cl_man_list_r

    def __compute_max_cl_man_list(self):
        
        max_cl_man_list_l_trasl = []
        max_cl_man_list_l_rot = []
        max_cl_man_list_r_trasl = []
        max_cl_man_list_r_rot = []

        for ms in range(len(self._q)):

            n_sampl = self._q[ms].shape[1]

            max_cl_man_list_l_trasl.append(np.max(self._cl_man_llist[ms][0, :])) 
            max_cl_man_list_l_rot.append(np.max(self._cl_man_llist[ms][1, :]))

            max_cl_man_list_r_trasl.append(np.max(self._cl_man_rlist[ms][0, :]))
            max_cl_man_list_r_rot.append(np.max(self._cl_man_rlist[ms][1, :]))

        max_cl_man_list_l = [max_cl_man_list_l_trasl, max_cl_man_list_l_rot]
        max_cl_man_list_r = [max_cl_man_list_r_trasl, max_cl_man_list_r_rot]

        return max_cl_man_list_l, max_cl_man_list_r

    def __compute_min_cl_man_list(self):
        
        min_cl_man_list_l_trasl = []
        min_cl_man_list_l_rot = []
        min_cl_man_list_r_trasl = []
        min_cl_man_list_r_rot = []

        for ms in range(len(self._q)):

            n_sampl = self._q[ms].shape[1]

            min_cl_man_list_l_trasl.append(np.min(self._cl_man_llist[ms][0, :])) 
            min_cl_man_list_l_rot.append(np.min(self._cl_man_llist[ms][1, :]))

            min_cl_man_list_r_trasl.append(np.min(self._cl_man_rlist[ms][0, :]))
            min_cl_man_list_r_rot.append(np.min(self._cl_man_rlist[ms][1, :]))

        min_cl_man_list_l = [min_cl_man_list_l_trasl, min_cl_man_list_l_rot]
        min_cl_man_list_r = [min_cl_man_list_r_trasl, min_cl_man_list_r_rot]

        return min_cl_man_list_l, min_cl_man_list_r

    def __array2double_correction(self, input_data):

        for i in range(len(input_data)):

            input_data[i] = input_data[i][0][0] # needed because scalar data are loaded as matrices

    def __get_data_matching(self, pattern: str,
                            dest: list,
                            source: dict,
                            is_dict = False, 
                            patter_is_varname = False, 
                            is_scalar = False):

        print(colored("Exctracting data matching pattern \"" + pattern + "\"...", 
                        "magenta"))

        for i in range(len(source)):
            
            varnames_i = source[i].keys()

            for varname in varnames_i:
                
                if not patter_is_varname:
                    
                    if pattern in varname:
                    
                        if is_dict and (type(dest[i]) == type({})):

                            dest[i][varname] = source[i][varname]
                        
                        if not is_dict:
                            
                            dest[i] = source[i][varname]

                else:

                    if varname == pattern:
                    
                        if is_dict and (type(dest[i]) == type({})):

                            dest[i][varname] = source[i][varname]
                        
                        if not is_dict:
                            
                            dest[i] = source[i][varname]
        
        if is_scalar:

            try:
                
                self.__array2double_correction(dest)

            except:
                
                print(colored("Failed to apply matrix to scalar correction", "yellow"))
    
    def clusterize(self, n_clusters):

        self._clusterer = Clusterer(self._q_design.T, self._opt_costs, self._n_int, n_clusters = n_clusters)

    def print_sol_run_info(self, round2 = 8):
        
        print(colored("############ SOLUTION RUN " + self._unique_id + " ############\n", "blue"))

        print(colored(" TASK INFO:\n", "white"))

        print(colored(" urdf_full_path:", "white"), self._urdf_full_path)

        print(colored(" coll_yaml_path:", "white"), self._coll_yaml_path)

        print(colored(" ms_trgt:", "white"), self._ms_trgt)

        print(colored(" is_biman_pick:", "white"), self._is_biman_pick)

        print(colored(" is_in_place_flip:", "white"), self._is_in_place_flip)

        print(colored(" filling_nnodes:", "white"), self._filling_nnodes)
        
        print(colored(" task_dt:", "white"), self._task_dt)
        
        print(colored(" ig_seed:", "white"), self._ig_seed)

        print(colored(" max_retry_n:", "white"), self._max_retry_n)

        print(colored(" ny_sampl:", "white"), self._ny_sampl)

        print(colored(" y_sampl_ub:", "white"), self._y_sampl_ub)

        print(colored(" nodes_list:", "white"), self._nodes_list)

        print(colored(" t_exec_task:", "white"), self._t_exec_task)

        print(colored(" is_class_man:", "white"), self._is_class_man)

        print(colored(" is_sliding_wrist:", "white"), self._is_sliding_wrist)

        print(colored(" wrist_off:", "white"), self._wrist_off)

        print("\n")

        print(colored(" SOLVER INFO:\n", "white"))
        
        print(colored(" integrator:", "white"), self._integrator)
        
        print(colored(" tol:", "white"), self._slvr_opts_tol)

        print(colored(" max_iter:", "white"), self._slvr_opts_maxiter)

        print(colored(" constraint violation:", "white"), self._slvr_opts_cnstr_viol)
        
        print(colored(" verbosity level:", "white"), self._slvr_opts_print_l)
        
        print(colored(" lin. solver:", "white"), self._slvr_opts_lin_solv)

        print(colored(" solver type:", "white"), self._solver_type)

        print(colored(" transcription method:", "white"), self._transcription_method)

        print("\n")

        print(colored(" SOLUTION STATISTICS:\n", "white"))
        
        print(colored(" tot_sol_time:", "white"), np.round(self._tot_sol_time, round2))

        print(colored(" avrg_sol_time:", "white"), np.round(self._avrg_sol_time, round2))

        print(colored(" max_sol_time:", "white"), np.round(self._max_sol_time, round2))

        print(colored(" min_sol_time:", "white"), np.round(self._min_sol_time, round2))

        print(colored(" rmse_sol_time:", "white"), np.round(self._rmse_sol_time, round2))

        print(colored(" n_opt_sols:", "white"), self._n_opt_sols)

        print(colored(" n_fail_sols:", "white"), self._n_fail_sols)

        print(colored(" optimal solutions to multistarts number ratio:", "white"),\
                                    np.round(self._n_opt_sols/self._ms_trgt, round2))

        print("\n")

        print(colored(" tot_niters2sol:", "white"), np.round(self._tot_niters2sol, round2))

        print(colored(" avrg_niters2sol:", "white"), np.round(self._avrg_niters2sol, round2))

        print(colored(" max_niters2sol:", "white"), np.round(self._max_niters2sol, round2))

        print(colored(" min_niters2sol:", "white"), np.round(self._min_niters2sol, round2))

        print(colored(" rmse_niters2sol:", "white"), np.round(self._rmse_niters2sol, round2))

        print("\n")

        print(colored(" tot_trial_idxs:", "white"), np.round(self._tot_trial_idxs, round2))

        print(colored(" avrg_trial_idxs:", "white"), np.round(self._avrg_trial_idxs, round2))

        print(colored(" max_trial_idxs:", "white"), np.round(self._max_trial_idxs, round2))

        print(colored(" min_trial_idxs:", "white"), np.round(self._min_trial_idxs, round2))

        print(colored(" rmse_trial_idxs:", "white"), np.round(self._rmse_trial_idxs, round2))

        print("\n")

        print(colored(" avrg_opt_costs:", "white"), np.round(self._avrg_opt_costs, round2))

        print(colored(" max_opt_costs:", "white"), np.round(self._max_opt_costs, round2))

        print(colored(" min_opt_cost:", "white"), np.round(self._min_opt_costs, round2))

        print(colored(" rmse_opt_costs:", "white"), np.round(self._rmse_opt_costs, round2))

        print("\n")

        print(colored(" avrg_man_cost:", "white"), np.round(self._avrg_man_cost, round2))

        print(colored(" max_man_cost:", "white"), np.round(self._max_man_cost, round2))

        print(colored(" min_man_cost:", "white"), np.round(self._min_man_cost, round2))

        print(colored(" rmse_man_cost:", "white"), np.round(self._rmse_man_cost, round2))

        print("\n")

        print(colored(" avrg_man_index:", "white"), np.round(self._avrg_man_index, round2))

        print(colored(" max_man_index:", "white"), np.round(self._max_man_index, round2))

        print(colored(" min_man_index:", "white"), np.round(self._min_man_index, round2))

        print(colored(" rmse_man_index:", "white"), np.round(self._rmse_man_index, round2))

        print("\n")
        
        if self.cl_man_post_proc:

            print(colored(" left avrg_cl_man_index:", "white"), np.round(self._avrg_cl_man_llist, round2))
            print(colored(" left max_cl_man_index:", "white"), np.round(self._max_cl_man_llist, round2))
            print(colored(" left min_cl_man_index:", "white"), np.round(self._min_cl_man_llist, round2))
            print(colored(" left rmse_cl_man_index:", "white"), np.round(self._rmse_cl_man_llist, round2))

            print(colored(" right avrg_cl_man_index:", "white"), np.round(self._avrg_cl_man_rlist, round2))
            print(colored(" right max_cl_man_index:", "white"), np.round(self._max_cl_man_rlist, round2))
            print(colored(" right min_cl_man_index:", "white"), np.round(self._min_cl_man_rlist, round2))
            print(colored(" right rmse_cl_man_index:", "white"), np.round(self._rmse_cl_man_rlist, round2))

            print("\n")

        print(colored("\n########################################################\n", "blue"))

    def make_solver_stat_plots(self, bin_scale_factor = 20.0, round2 = 2):

        green_diamond = dict(markerfacecolor='g', marker='D')

        # sol time
        leg_title = "average: " + str(round(self._avrg_sol_time, round2)) + "\n" + \
                    "RMSE: " + str(round(self._rmse_sol_time, round2)) + "\n" + \
                    "max: " + str(round(self._max_sol_time, round2)) + "\n" + \
                    "min: " + str(round(self._min_sol_time, round2)) + "\n" 

        _, ax_sol_t = plt.subplots(1)
        ax_sol_t.hist(self._sol_times, bins = int(len(self._sol_times)/bin_scale_factor))
        leg_t = ax_sol_t.legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        ax_sol_t.set_xlabel(r"sol_t[s]")
        ax_sol_t.set_ylabel(r"N samples")
        ax_sol_t.set_title(r"Solution time", fontdict=None, loc='center')
        ax_sol_t.grid()
        
        _, ax_sol_t_box = plt.subplots(1)
        ax_sol_t_box.boxplot(self._sol_times, flierprops = green_diamond, vert=True, 
                        # whis = (0, 100),
                        autorange = True)
        leg_box_t = ax_sol_t_box.legend(loc="upper right", 
            title = leg_title)
        leg_box_t.set_draggable(True)                
        ax_sol_t_box.set_ylabel("sol. time[s]")
        ax_sol_t_box.set_title(r"Solution time", fontdict=None, loc='center')
        ax_sol_t_box.grid()

        # n.iterations to solution
        leg_title = "average: " + str(round(self._avrg_niters2sol, round2)) + "\n" + \
                    "RMSE: " + str(round(self._rmse_niters2sol, round2)) + "\n" + \
                    "max: " + str(round(self._max_niters2sol, round2)) + "\n" + \
                    "min: " + str(round(self._min_niters2sol, round2)) + "\n" 

        _, ax_hist_niter = plt.subplots(1)
        ax_hist_niter.hist(self._niters2sol, bins = int(len(self._niters2sol)/bin_scale_factor))
        leg_niter = ax_hist_niter.legend(loc="upper right", 
            title = leg_title)
        leg_niter.set_draggable(True)
        ax_hist_niter.set_xlabel(r"n. iterations")
        ax_hist_niter.set_ylabel(r"N samples")
        ax_hist_niter.set_title(r"Number of iterations to solution", fontdict=None, loc='center')
        ax_hist_niter.grid()
        
        _, ax_niter_box = plt.subplots(1)
        ax_niter_box.boxplot(self._niters2sol, flierprops = green_diamond, vert=True, 
                        # whis = (0, 100),
                        autorange = True)
        leg_niter_box = ax_niter_box.legend(loc="upper right", 
            title = leg_title)
        leg_niter_box.set_draggable(True)
        ax_niter_box.set_ylabel("n. iterations")
        ax_niter_box.set_title(r"Number of iterations to solution", fontdict=None, loc='center')
        ax_niter_box.grid()

        # n. of solution retries
        leg_title = "average: " + str(round(self._avrg_trial_idxs, round2)) + "\n" + \
                    "RMSE: " + str(round(self._rmse_trial_idxs, round2)) + "\n" + \
                    "max: " + str(round(self._max_trial_idxs, round2)) + "\n" + \
                    "min: " + str(round(self._min_trial_idxs, round2)) + "\n" 
        _, ax_retr_sc = plt.subplots(1)
        ax_retr_sc.scatter(self._ms_indxs, self._trial_idxs)
        leg_retries = ax_retr_sc.legend(loc="upper right", 
            title = leg_title)
        leg_retries.set_draggable(True)
        ax_retr_sc.set_xlabel(r"multistart node index")
        ax_retr_sc.set_ylabel(r"retry number")
        ax_retr_sc.set_title(r"Number of solution retries per ms node", fontdict=None, loc='center')
        ax_retr_sc.grid()
        
        _, ax_retr_box = plt.subplots(1)
        ax_retr_box.boxplot(self._trial_idxs, flierprops = green_diamond, vert=True, 
                        # whis = (0, 100),
                        autorange = True)
        leg_retries_box = ax_retr_box.legend(loc="upper right", 
            title = leg_title)
        leg_retries_box.set_draggable(True)
        ax_retr_box.set_ylabel(r"retry number")
        ax_retr_box.set_title(r"Number of solution retries per ms node", fontdict=None, loc='center')
        ax_retr_box.grid()

    def make_cost_stat_plots(self, bin_scale_factor = 20.0, round2 = 5):

        green_diamond = dict(markerfacecolor='g', marker='D')

        # opt cost
        leg_title = "average: " + str(round(self._avrg_opt_costs, round2)) + "\n" + \
                    "RMSE: " + str(round(self._rmse_opt_costs, round2)) + "\n" + \
                    "max: " + str(round(self._max_opt_costs, round2)) + "\n" + \
                    "min: " + str(round(self._min_opt_costs, round2)) + "\n" 

        _, ax_opt_c_hist = plt.subplots(1)
        ax_opt_c_hist.hist(self._opt_costs, bins = int(len(self._opt_costs)/bin_scale_factor))
        leg_opt_c = ax_opt_c_hist.legend(loc="upper right", 
            title = leg_title)
        leg_opt_c.set_draggable(True)
        ax_opt_c_hist.set_xlabel(r"opt cost")
        ax_opt_c_hist.set_ylabel(r"N samples")
        ax_opt_c_hist.set_title(r"Opt. cost", fontdict=None, loc='center')
        ax_opt_c_hist.grid()
        
        _, ax_opt_c_box = plt.subplots(1)
        ax_opt_c_box.boxplot(self._opt_costs, flierprops = green_diamond, vert=True, 
                        # whis = (0, 100),
                        autorange = True)
        leg_opt_c_box = ax_opt_c_box.legend(loc="upper right", 
            title = leg_title)
        leg_opt_c_box.set_draggable(True)
        ax_opt_c_box.set_ylabel(r"opt cost")
        ax_opt_c_box.set_title(r"Opt. cost", fontdict=None, loc='center')
        ax_opt_c_box.grid()

        # man. cost
        leg_title = "average: " + str(round(self._avrg_man_cost, round2)) + "\n" + \
                    "RMSE: " + str(round(self._rmse_man_cost, round2)) + "\n" + \
                    "max: " + str(round(self._max_man_cost, round2)) + "\n" + \
                    "min: " + str(round(self._min_man_cost, round2)) + "\n" 

        _, ax_opt_m_hist = plt.subplots(1)
        ax_opt_m_hist.hist(self._man_cost, bins = int(len(self._man_cost)/bin_scale_factor))
        leg_opt_m = ax_opt_m_hist.legend(loc="upper right", 
            title = leg_title)
        leg_opt_m.set_draggable(True)
        ax_opt_m_hist.set_xlabel(r"man. cost")
        ax_opt_m_hist.set_ylabel(r"N samples")
        ax_opt_m_hist.set_title(r"Man. cost", fontdict=None, loc='center')
        ax_opt_m_hist.grid()
        
        _, ax_opt_c_box = plt.subplots(1)
        ax_opt_c_box.boxplot(self._man_cost, flierprops = green_diamond, vert=True, 
                        # whis = (0, 100),
                        autorange = True)
        leg_opt_m_box = ax_opt_c_box.legend(loc="upper right", 
            title = leg_title)
        leg_opt_m_box.set_draggable(True)
        ax_opt_c_box.set_ylabel(r"man. cost")
        ax_opt_c_box.set_title(r"Man. cost", fontdict=None, loc='center')
        ax_opt_c_box.grid()

        # man. index
        leg_title = "average: " + str(round(self._avrg_man_index, round2)) + "\n" + \
                    "RMSE: " + str(round(self._rmse_man_index, round2)) + "\n" + \
                    "max: " + str(round(self._max_man_index, round2)) + "\n" + \
                    "min: " + str(round(self._min_man_index, round2)) + "\n" 

        _, ax_opt_mi_hist = plt.subplots(1)
        ax_opt_mi_hist.hist(self._man_index, bins = int(len(self._man_index)/bin_scale_factor))
        leg_opt_m = ax_opt_mi_hist.legend(loc="upper right", 
            title = leg_title)
        leg_opt_m.set_draggable(True)
        ax_opt_mi_hist.set_xlabel(r"man. index")
        ax_opt_mi_hist.set_ylabel(r"N samples")
        ax_opt_mi_hist.set_title(r"Man. index", fontdict=None, loc='center')
        ax_opt_mi_hist.grid()
        
        _, ax_opt_mi_box = plt.subplots(1)
        ax_opt_mi_box.boxplot(self._man_index, flierprops = green_diamond, vert=True, 
                        # whis = (0, 100),
                        autorange = True)
        leg_opt_m_box = ax_opt_mi_box.legend(loc="upper right", 
            title = leg_title)
        leg_opt_m_box.set_draggable(True)
        ax_opt_mi_box.set_xlabel(r"man. index")
        ax_opt_mi_box.set_title(r"Man. index", fontdict=None, loc='center')
        ax_opt_mi_box.grid()

        # cl. man. index

        if self.cl_man_post_proc:

            leg_title= "average left arm: " + str(np.round([self._2avrg_cl_man_l_trasl, self._2avrg_cl_man_l_rot], round2)) + "\n" + \
                        "RMSE left arm: " + str(np.round([self._avrg_rmse_cl_man_l_trasl, self._avrg_rmse_cl_man_l_rot], round2)) + "\n" + \
                        "max left arm: " + str(np.round([self._2max_cl_man_l_trasl, self._2max_cl_man_l_rot], round2)) + "\n" + \
                        "min left arm: " + str(np.round([self._2min_cl_man_l_trasl, self._2min_cl_man_l_rot], round2)) + "\n" 

            _, ax_opt_mi_hist = plt.subplots(1)
            ax_opt_mi_hist.hist(np.array(self._avrg_cl_man_llist + self._avrg_cl_man_rlist).T,\
                                    bins = int(len(self._avrg_cl_man_llist[0])/bin_scale_factor), 
                                    label = ["trasl. left", "rot. left", \
                                        "trasl. right", "rot. right"])
            leg = ax_opt_mi_hist.legend(loc="upper right", 
                title = leg_title)
            leg.set_draggable(True)
            ax_opt_mi_hist.set_xlabel(r"avrg. cl. man. index")
            ax_opt_mi_hist.set_ylabel(r"N samples")
            ax_opt_mi_hist.set_title(r"Average classical man. across multistart samples", fontdict=None, loc='center')
            ax_opt_mi_hist.grid()

    def make_plots(self, bin_scale_factor = 20.0):
        
        self.make_solver_stat_plots(bin_scale_factor=bin_scale_factor)

        self.make_cost_stat_plots(bin_scale_factor=bin_scale_factor)

        self._clusterer.create_cluster_plot(plt_red_factor=4)

    def show_plots(self):

        plt.show()

        return True

    def save_plots():

        return True

    def dump_data2file(self):

        attr_dict = self.__dict__

        attr_names = attr_dict.keys()

        for attr in attr_names:

            if "_" not in attr:

                self._dump_vars[attr] = attr_dict[attr]

class PostProcL2:

    def __init__(self, load_path, 
                clust_dir_basename = "clust", 
                additional_info_pattern = "info", 
                dump_dirname = "l2_postproc"):

        self._load_path = load_path + "/second_level/"
        self._clust_dir_basename = clust_dir_basename
        self._dump_path = load_path + "/" + dump_dirname

        # read sol paths from directory
        additional_info_name = additional_info_pattern
        additional_info_candidates = os.listdir(self._load_path)
        self.add_info_filename = ""

        add_info_filename_aux = []

        for i in range(len(additional_info_candidates)):

            if additional_info_name in additional_info_candidates[i]:

                add_info_filename_aux.append(additional_info_candidates[i]) 

        if len(add_info_filename_aux) == 0:

            raise Exception("PostProc2ndLev: didn't find any solution information file matching pattern \"" +
                            additional_info_name + "\"" + " in base directory " + self._load_path + ".\n" +
                            "Please provide a valid solution information file.")

        if len(add_info_filename_aux) > 1:

            raise Exception("PostProc2ndLev: too many solution information files provided.\n" + 
                            "Please make sure the loading directory only contains coherent data.")


        self._add_info_path = self._load_path + "/" + add_info_filename_aux[0]
        self._info_data = mat_storer.matStorer(self._add_info_path).load()
        self.__read_info_data()

        self.loaders = [] * self._n_clust # list of loaders (one for each cluster)
        self._second_lev_opt_costs = [] 
        self._second_lev_man_measure = []

        self._opt_mult_indeces = [] # index of the solution (w.r.t. the n multistarts per cluster)
        
        self._confidence_coeffs = [-1] * self._n_clust

        self._second_lev_true_costs = [-1] * self._n_clust
        self._second_lev_best_ms_index = -1
        self.second_lev_true_man = [-1] * self._n_clust
        self.n_of_improved_costs = 0

        self._second_lev_weighted_costs = [-1] * self._n_clust

        self._rmse_man_meas = [-1] * self._n_clust
        self._rmse_opt_cost = [-1] * self._n_clust

        self.__load_clust_sols()
        self.__compute_second_lev_true_costs()
        self._l2_cl_cands_man_measure = compute_man_index(self._l2_cl_cands_opt_cost, self._n_int)

        self.__compute_conf_coeff()
        self.__compute_weighted_costs()
        self.__compute_rmse()

        self.best_second_lev_cost, self.best_second_lev_cl_index,\
            self.best_second_lev_man_measure, self.best_second_lev_qcodes = self.__compute_second_lev_best_sol()

        self.best_second_lev_weight_cost, self.best_second_lev_weight_cl_index,\
            self.best_second_lev_weight_man_measure, self.best_second_lev_weight_qcodes =\
                self.__compute_second_lev_best_sol(use_weighted=True)

        self.print_best_sol()
        self.print_best_sol(weighted=True)

        self.__dump_results()

    def __read_info_data(self):

        self._n_clust = self._info_data["n_clust"][0][0] # double loaded as matrix --> [0][0] necessary
        
        self._task_dt = self._info_data["dt"][0][0]
        self._filling_nnodes = self._info_data["filling_nodes"][0][0]

        self._l2_cl_cand_inds = self._info_data["l2_cl_cand_inds"][0]
        self._l2_cl_cands_opt_cost = self._info_data["l2_cl_opt_costs"][0]
        self._l2_cl_candidates = self._info_data["l2_cl_best_candidates"]
        self._l2_cl_cands_man_measure = self._info_data["l2_cl_cand_man_measure"]

        self._integrator = self._info_data["integrator"][0]
        self._ig_seed = self._info_data["ig_seed"][0][0]
        self._max_retry_n = self._info_data["max_retry_n"][0][0]
        self._ms_trgt = self._info_data["n_msrt_trgt"][0][0]
        self._ny_sampl = self._info_data["n_y_samples"][0]
        self._y_sampl_ub = self._info_data["y_sampl_ub"][0]
        self._nodes_list = correct_list(self._info_data["nodes_list"])
        self._n_int = self._nodes_list[-1][-1]
        self._proc_sol_divs = correct_list(self._info_data["proc_sol_divs"])
        self._rot_error_epsi = self._info_data["rot_error_epsi"][0][0]

        self._slvr_opts_tol = self._info_data["slvr_opts"]["ipopt.tol"][0][0][0][0]
        self._slvr_opts_maxiter = self._info_data["slvr_opts"]["ipopt.max_iter"][0][0][0][0]
        self._slvr_opts_cnstr_viol = self._info_data["slvr_opts"]["ipopt.constr_viol_tol"][0][0][0][0]
        self._slvr_opts_print_l = self._info_data["slvr_opts"]["ipopt.print_level"][0][0][0][0]
        self._slvr_opts_lin_solv = self._info_data["slvr_opts"]["ipopt.linear_solver"][0][0][0]
        self._solver_type = self._info_data["solver_type"][0]

        self._right_arm_picks = self._info_data["right_arm_picks"][0]
        self._is_biman_pick = bool(self._info_data["is_biman_pick"][0][0])
        self._is_in_place_flip = bool(self._info_data["is_in_place_flip"][0][0])
        self._t_exec_task = self._info_data["t_exec_task"][0][0]
        self._task_base_nnodes = self._info_data["task_base_nnodes"]
        self._task_dict = self._info_data["tasks_dict"]
        self._tasks_list = self._info_data["tasks_list"]
        self._transcription_method = self._info_data["transcription_method"][0]
        self._unique_id = self._info_data["unique_id"][0]
        self._is_class_man = bool(self._info_data["use_classical_man"][0][0])
        self._class_man_w_base = self._info_data["w_clman_base"][0][0]
        self._class_man_w_a = self._info_data["w_clman_actual"][0][0]
        self._man_w_base = self._info_data["w_man_base"][0][0]
        self._man_w_a = self._info_data["w_man_actual"][0][0]
        self._wrist_off = self._info_data["sliding_wrist_offset"][0][0]
        self._is_sliding_wrist = bool(self._info_data["is_sliding_wrist"][0][0])

    def __read_opt_data(self):

        # reading stuff from opt data
        self._coll_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("coll", self._coll_cnstrnt_data,
                                self._opt_data, 
                                is_dict = True)

        self._lambd_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("lambd", self._lambd_cnstrnt_data,
                                self._opt_data, 
                                is_dict = True)

        self._pos_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("pos", self._pos_cnstrnt_data, 
                                self._opt_data,
                                is_dict = True)

        self._rot_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("rot", self._rot_cnstrnt_data, 
                                self._opt_data,
                                is_dict = True)

        self._ws_lim_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("keep", self._ws_lim_cnstrnt_data, 
                                self._opt_data,
                                is_dict = True)

        self._codes_simmetry_cnstrt = [{}] * len(self._opt_data)
        self.__get_data_matching("same", self._codes_simmetry_cnstrt, 
                                self._opt_data,
                                is_dict = True)

        self._mult_shoot_cnstrnt_data = [None] * len(self._opt_data)
        self.__get_data_matching("multiple_shooting", self._mult_shoot_cnstrnt_data, 
                                self._opt_data,
                                patter_is_varname = True)

        self._codes_var_cnstr = [None] * len(self._opt_data)
        self.__get_data_matching("single_var", self._codes_var_cnstr, 
                                self._opt_data,
                                is_dict = True)

        self._niters2sol = [-1] * len(self._opt_data)
        self.__get_data_matching("n_iter2sol", self._niters2sol, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._opt_costs = [-1] * len(self._opt_data)
        self.__get_data_matching("opt_cost", self._opt_costs,
                                self._opt_data, 
                                patter_is_varname = True, 
                                is_scalar = True)

        self._q = [None] * len(self._opt_data)
        self.__get_data_matching("q", self._q, 
                                self._opt_data,
                                patter_is_varname = True)
        
        self._q_ig = [None] * len(self._opt_data)
        self.__get_data_matching("q_ig", self._q_ig,
                                self._opt_data, 
                                patter_is_varname = True)
                                    
        self._q_dot = [None] * len(self._opt_data)
        self.__get_data_matching("q_dot", self._q_dot,
                                self._opt_data, 
                                patter_is_varname = True)
        
        self._q_dot_ig = [None] * len(self._opt_data)
        self.__get_data_matching("q_dot_ig", self._q_dot_ig, 
                                self._opt_data,
                                patter_is_varname = True)

        self._ms_indxs = [None] * len(self._opt_data)
        self.__get_data_matching("multistart_index", self._ms_indxs, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._sol_times = [None] * len(self._opt_data)
        self.__get_data_matching("solution_time", self._sol_times, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._solve_failed = [None] * len(self._opt_data)
        self.__get_data_matching("solve_failed", self._solve_failed,
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._trial_idxs = [None] * len(self._opt_data)
        self.__get_data_matching("trial_index", self._trial_idxs, 
                                self._opt_data,
                                patter_is_varname = True, 
                                is_scalar = True)

        self._u_opt = [None] * len(self._opt_data)
        self.__get_data_matching("u_opt", self._u_opt,
                                self._opt_data, 
                                patter_is_varname = True)

        self._x_opt = [None] * len(self._opt_data)
        self.__get_data_matching("x_opt", self._x_opt, 
                                self._opt_data,
                                patter_is_varname = True)

        self._sols_run_ids = [None] * len(self._opt_data)
        self.__get_data_matching("run_id", self._sols_run_ids, 
                                self._opt_data,
                                patter_is_varname = True)

        self._q_design = extract_q_design(self._q)
        
    def __load_clust_sols(self):

        for cl in range(self._n_clust):
            
            #clusters are loaded in order
            self.loaders.append(LoadSols(self._load_path + self._clust_dir_basename + str(cl),\
                                        base_info_path = self._load_path))  

            opt_cost_aux_list = []
            opt_mult_aux_indeces = []

            for opt_sol_index in range(len(self.loaders[cl].opt_data)):

                opt_cost_aux_list.append(self.loaders[cl].opt_data[opt_sol_index]["opt_cost"][0][0])
                opt_mult_aux_indeces.append(self.loaders[cl].opt_data[opt_sol_index]["multistart_index"][0][0])
            
            self._second_lev_opt_costs.append(opt_cost_aux_list)
            self._second_lev_man_measure.append(compute_man_index(opt_cost_aux_list, self._n_int))

            self._opt_mult_indeces.append(opt_mult_aux_indeces)

    def __compute_second_lev_true_costs(self):

        for cl in range(self._n_clust):
            
            min_second_lev_cl = np.min(np.array(self._second_lev_opt_costs[cl]))

            self._second_lev_true_costs[cl] = \
                min_second_lev_cl if (min_second_lev_cl<=self._l2_cl_cands_opt_cost[cl]) else self._l2_cl_cands_opt_cost[cl]

        self.second_lev_true_man = compute_man_index(self._second_lev_true_costs, self._n_int)

        did_cost_improve = np.array(self._l2_cl_cands_opt_cost != self._second_lev_true_costs)
        self.n_of_improved_costs = len(np.argwhere(did_cost_improve == True).flatten())

    def __compute_weighted_costs(self):

        for cl in range(self._n_clust):

            self._second_lev_weighted_costs[cl] = self._second_lev_true_costs[cl] / self._confidence_coeffs[cl]

    def __compute_rmse(self):
        
        for cl in range(self._n_clust):
            
            min_man_cl = self.second_lev_true_man[cl]
            sum_sqrd_man = 0.0
            sum_sqrd_cost = 0.0
            n_opt_ms_cl = len(self._second_lev_man_measure[cl])
            for ms_sample in range(n_opt_ms_cl):
                
                sum_sqrd_man = sum_sqrd_man + (self._second_lev_man_measure[cl][ms_sample] - min_man_cl)**2
                sum_sqrd_cost = sum_sqrd_cost + (self._second_lev_opt_costs[cl][ms_sample] - min_man_cl)**2

            self._rmse_man_meas[cl] = np.sqrt(sum_sqrd_man / n_opt_ms_cl)
            self._rmse_opt_cost[cl] = np.sqrt(sum_sqrd_cost / n_opt_ms_cl)

    def __compute_second_lev_best_sol(self, use_weighted = False):

        n_des_params = len(self._l2_cl_candidates) # number of design parameters

        best_second_lev_qcodes = np.zeros((n_des_params, 1)).flatten()

        if not use_weighted:
            
            best_second_lev_cost = min(self._second_lev_true_costs)
            best_second_lev_cl_index = \
                np.argwhere(np.array(self._second_lev_true_costs) == best_second_lev_cost)[0][0]

            best_second_lev_man_measure = compute_man_index([best_second_lev_cost], self._n_int)[0]

            for i in range(n_des_params): # iterating through design paramter dimension

                best_second_lev_qcodes[i] = self._l2_cl_candidates[i][best_second_lev_cl_index]

        else:   

            best_second_lev_cost = min(self._second_lev_weighted_costs)
            best_second_lev_cl_index = \
                np.argwhere(np.array(self._second_lev_weighted_costs) == best_second_lev_cost)[0][0]

            best_second_lev_man_measure = compute_man_index([best_second_lev_cost], self._n_int)[0]
            
            for i in range(n_des_params):

                best_second_lev_qcodes[i] = self._l2_cl_candidates[i][best_second_lev_cl_index]

        return best_second_lev_cost, best_second_lev_cl_index,\
                best_second_lev_man_measure, best_second_lev_qcodes

    def __compute_conf_coeff(self):
        
        opt_sols_n_aux_list = []
        for cl in range(self._n_clust):
            opt_sols_n_aux_list.append(len(self._second_lev_opt_costs[cl]))

        max_n_cl_opt_sol = self._ms_trgt

        self._confidence_coeffs = [opt_sols_n_aux_list[cl]/max_n_cl_opt_sol for cl in range(self._n_clust)]

    def __dump_results(self):

        if not os.path.isdir(self._dump_path):

            os.makedirs(self._dump_path)

        task_info_dumper = SolDumper()

        stuff = {"unique_id": self._unique_id,\
                    "n_clust": self._n_clust,
                    "n_int": self._n_int,
                    "_l2_cl_cand_inds": self._l2_cl_cand_inds,
                    "_l2_cl_best_candidates": self._l2_cl_candidates,
                    "_l2_cl_opt_costs": self._l2_cl_cands_opt_cost, 
                    "first_lev_man_measure": self._l2_cl_cands_man_measure,
                    "second_lev_opt_costs": np.array(self._second_lev_opt_costs, dtype=object),
                    "second_lev_man_measure": np.array(self._second_lev_man_measure, dtype=object),
                    "opt_mult_indeces": np.array(self._opt_mult_indeces, dtype=object),
                    "confidence_coeffs": self._confidence_coeffs,
                    "second_lev_true_costs": self._second_lev_true_costs,
                    "n_of_improved_costs": self.n_of_improved_costs,
                    "second_lev_weighted_costs": self._second_lev_weighted_costs,
                    "best_second_lev_cost": self.best_second_lev_cost,
                    "best_second_lev_cl_index": self.best_second_lev_cl_index,
                    "best_second_lev_man_measure": self.best_second_lev_man_measure,
                    "best_second_lev_qcodes": self.best_second_lev_qcodes,
                    "best_second_lev_weight_cost": self.best_second_lev_weight_cost,
                    "best_second_lev_weight_cl_index": self.best_second_lev_weight_cl_index,
                    "best_second_lev_weight_man_measure": self.best_second_lev_weight_man_measure,
                    "best_second_lev_weight_qcodes": self.best_second_lev_weight_qcodes,
                    "rmse_man_meas": self._rmse_man_meas, 
                    "rmse_opt_cost":self._rmse_opt_cost}

        task_info_dumper.add_storer(stuff, self._dump_path,\
                                "2nd_lev_postproc" + str(self._unique_id),\
                                False)        

        task_info_dumper.dump()

    def print_best_sol(self, weighted= False):

        if not weighted:

            print("Best sol. cost: " + str(self.best_second_lev_cost))
            print("Best sol. index : " + str(self.best_second_lev_cl_index))
            print("Best sol. man measure: " + str(self.best_second_lev_man_measure))
            print("Best q codes.: " + str(self.best_second_lev_qcodes))
            print("\n")

        else:
            
            print("Best weighted sol. cost: " + str(self.best_second_lev_weight_cost))
            print("Best weighted sol. index: " + str(self.best_second_lev_weight_cl_index))
            print("Best weighted sol. man measure: " + str(self.best_second_lev_weight_man_measure))
            print("Best weighted q codes.: " + str(self.best_second_lev_weight_qcodes))
            print("\n")