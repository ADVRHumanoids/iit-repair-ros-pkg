
from codesign_pyutils.dump_utils import SolDumper

from codesign_pyutils.load_utils import LoadSols

from codesign_pyutils.miscell_utils import compute_man_cost, compute_man_index

from codesign_pyutils.task_utils import gen_task_copies

from horizon.utils import mat_storer

import os

from termcolor import colored

import numpy as np

import rospkg
class PostProcL1:

    def __init__(self, load_path, 
                additional_info_pattern="info", 
                l1_dirname="first_level", 
                dump_dirname="l1_postproc", 
                opt_dirname="opt", 
                fail_dirname="failed"):

        self._load_path = load_path + "/" + l1_dirname + "/"
        self._dump_path = load_path + "/" + dump_dirname

        # useful paths
        rospackage = rospkg.RosPack() # Only for taking the path to the leg package
        urdfs_path = rospackage.get_path("repair_urdf") + "/urdf"
        urdf_name = "repair_full"
        self._urdf_full_path = urdfs_path + "/" + urdf_name + ".urdf"
        coll_yaml_name = "arm_coll.yaml"
        self._coll_yaml_path = rospackage.get_path("repair_urdf") + "/config/" + coll_yaml_name

        print(colored("\n--->Initializing first level postprocess object from folder \"" + 
                        self._load_path + "\n", 
                        "blue"))

        self._sol_loader = LoadSols(self._load_path,
                                opt_dir_name=opt_dirname, 
                                fail_dir_name=fail_dirname, 
                                additional_info_pattern=additional_info_pattern)
        
        self._opt_data = self._sol_loader.opt_data
        self._fail_data = self._sol_loader.fail_data

        self._prb_info_data = self._sol_loader.task_info_data

        self._dump_vars = {}
        
        # reading stuff from opt data
        self._coll_cnstrnt_data = [{}] * len(self._opt_data)
        self.__get_data_matching("coll", self._coll_cnstrnt_data,
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

        self._ms_index = [None] * len(self._opt_data)
        self.__get_data_matching("multistart_index", self._ms_index, 
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

        # reading stuff from add info data
        self._task_dt = self._prb_info_data["dt"][0][0]
        self._filling_nnodes = self._prb_info_data["filling_nodes"][0][0]
        self._integrator = self._prb_info_data["integrator"][0]
        self._ig_seed = self._prb_info_data["ig_seed"][0][0]
        self._max_retry_n = self._prb_info_data["max_retry_n"][0][0]
        self._ms_trgt = self._prb_info_data["n_msrt_trgt"][0][0]
        self._ny_sampl = self._prb_info_data["n_y_samples"][0][0]
        self._y_sampl_ub = self._prb_info_data["y_sampl_ub"][0][0]
        self._nodes_list = self._prb_info_data["nodes_list"]
        self._proc_sol_divs = self._prb_info_data["proc_sol_divs"]
        self._rot_error_epsi = self._prb_info_data["rot_error_epsi"][0][0]
        self._slvr_opts = self._prb_info_data["slvr_opts"]
        self._solver_type = self._prb_info_data["solver_type"][0]
        self._t_exec_task = self._prb_info_data["t_exec_task"][0][0]
        self._task_base_nnodes = self._prb_info_data["task_base_nnodes"]
        self._task_dict = self._prb_info_data["tasks_dict"]
        self._tasks_list = self._prb_info_data["tasks_list"]
        self._transcription_method = self._prb_info_data["transcription_method"][0]
        self._run_id = self._prb_info_data["unique_id"][0]
        self._is_class_man = self._prb_info_data["use_classical_man"][0][0]
        self._class_man_w_base = self._prb_info_data["w_clman_base"][0][0]
        self._class_man_w_a = self._prb_info_data["w_clman_actual"][0][0]
        self._man_w_base = self._prb_info_data["w_man_base"][0][0]
        self._man_w_a = self._prb_info_data["w_man_actual"][0][0]
        self._wrist_off = self._prb_info_data["sliding_wrist_offset"][0][0]
        self._is_sliding_wrist = bool(self._prb_info_data["is_sliding_wrist"][0][0])

        self._man_cost = self.__get_man_cost()
        self._man_index = self.__get_man_index(self._man_cost)

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

        print(colored("\nGenerating task copy...\n", "magenta"))
        
        self.task_copy = gen_task_copies(self._man_w_base, self._class_man_w_base,
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
    
    def print_sol_run_info(self):
        
        print(colored("\n######################################################\n", "blue"))

        print(colored("-->SOLUTION RUN " + self._run_id + " INFORMATION:\n", "blue"))

        print(colored(" TASK INFO:", "white"))

        print("\n")

        print(colored(" urdf_full_path:", "white"), self._urdf_full_path)

        print(colored(" coll_yaml_path:", "white"), self._coll_yaml_path)

        print(colored(" ms_trgt:", "white"), self._ms_trgt)

        print(colored(" filling_nnodes:", "white"), self._filling_nnodes)
        
        print(colored(" task_dt:", "white"), self._task_dt)
        
        print(colored(" ig_seed:", "white"), self._ig_seed)

        print(colored(" max_retry_n:", "white"), self._max_retry_n)

        print(colored(" ms_trgt:", "white"), self._ms_trgt)

        print(colored(" ny_sampl:", "white"), self._ny_sampl)

        print(colored(" y_sampl_ub:", "white"), self._y_sampl_ub)

        print(colored(" nodes_list:", "white"), self._nodes_list)

        print(colored(" t_exec_task:", "white"), self._t_exec_task)

        print(colored(" is_class_man:", "white"), self._is_class_man)

        print(colored(" is_sliding_wrist:", "white"), self._is_sliding_wrist)

        print(colored(" wrist_off:", "white"), self._wrist_off)

        print("\n")

        print(colored(" SOLVER INFO:", "white"))
        
        print("\n")

        print(colored(" integrator:", "white"), self._integrator)
        
        print(colored(" slvr_opts:", "white"), self._slvr_opts)

        print(colored(" solver_type:", "white"), self._solver_type)

        print(colored(" transcription_method:", "white"), self._transcription_method)

        print("\n")

        print(colored(" SOLUTION INFO:", "white"))

        print("\n")
        
        print(colored(" tot_sol_time:", "white"), np.round(self._tot_sol_time, 8))

        print(colored(" avrg_sol_time:", "white"), np.round(self._avrg_sol_time, 8))

        print(colored(" max_sol_time:", "white"), np.round(self._max_sol_time, 8))

        print(colored(" min_sol_time:", "white"), np.round(self._min_sol_time, 8))

        print(colored(" rmse_sol_time:", "white"), np.round(self._rmse_sol_time, 8))

        print("\n")

        print(colored(" tot_niters2sol:", "white"), np.round(self._tot_niters2sol, 8))

        print(colored(" avrg_niters2sol:", "white"), np.round(self._avrg_niters2sol, 8))

        print(colored(" max_niters2sol:", "white"), np.round(self._max_niters2sol, 8))

        print(colored(" min_niters2sol:", "white"), np.round(self._min_niters2sol, 8))

        print(colored(" rmse_niters2sol:", "white"), np.round(self._rmse_niters2sol, 8))

        print("\n")

        print(colored(" tot_trial_idxs:", "white"), np.round(self._tot_trial_idxs, 8))

        print(colored(" avrg_trial_idxs:", "white"), np.round(self._avrg_trial_idxs, 8))

        print(colored(" max_trial_idxs:", "white"), np.round(self._max_trial_idxs, 8))

        print(colored(" min_trial_idxs:", "white"), np.round(self._min_trial_idxs, 8))

        print(colored(" rmse_trial_idxs:", "white"), np.round(self._rmse_trial_idxs, 8))

        print("\n")

        print(colored(" avrg_man_cost:", "white"), np.round(self._avrg_man_cost, 8))

        print(colored(" max_man_cost:", "white"), np.round(self._max_man_cost, 8))

        print(colored(" min_man_cost:", "white"), np.round(self._min_man_cost, 8))

        print(colored(" rmse_man_cost:", "white"), np.round(self._rmse_man_cost, 8))

        print("\n")

        print(colored(" avrg_man_index:", "white"), np.round(self._avrg_man_index, 8))

        print(colored(" max_man_index:", "white"), np.round(self._max_man_index, 8))

        print(colored(" min_man_index:", "white"), np.round(self._min_man_index, 8))

        print(colored(" rmse_man_index:", "white"), np.round(self._rmse_man_index, 8))

        print("\n")

        print(colored(" avrg_opt_costs:", "white"), np.round(self._avrg_opt_costs, 8))

        print(colored(" max_opt_costs:", "white"), np.round(self._max_opt_costs, 8))

        print(colored(" min_opt_cost:", "white"), np.round(self._min_opt_costs, 8))

        print(colored(" rmse_opt_costs:", "white"), np.round(self._rmse_opt_costs, 8))

        print("\n")

        print(colored("\n######################################################\n", "blue"))

    def make_plots():

        return True

    def show_plots():

        return True

    def save_plots():

        return True

    def __dump_data2file(self):

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

        self.load_path = load_path + "/second_level/"
        self.clust_dir_basename = clust_dir_basename
        self.dump_path = load_path + "/" + dump_dirname

        # read sol paths from directory
        additional_info_name = additional_info_pattern
        additional_info_candidates = os.listdir(self.load_path)
        self.add_info_filename = ""

        add_info_filename_aux = []

        for i in range(len(additional_info_candidates)):

            if additional_info_name in additional_info_candidates[i]:

                add_info_filename_aux.append(additional_info_candidates[i]) 

        if len(add_info_filename_aux) == 0:

            raise Exception("PostProc2ndLev: didn't find any solution information file matching pattern \"" +
                            additional_info_name + "\"" + "in base directory " + self.load_path + ".\n" +
                            "Please provide a valid solution information file.")

        if len(add_info_filename_aux) > 1:

            raise Exception("PostProc2ndLev: too many solution information files provided.\n" + 
                            "Please make sure the loading directory only contains coherent data.")

        self.add_info_path = self.load_path + "/" + add_info_filename_aux[0]
        self.task_info_data = mat_storer.matStorer(self.add_info_path).load()

        self.n_clust = self.task_info_data["n_clust"][0][0] # double loaded as matrix --> [0][0] necessary
        self.first_lev_opt_costs = self.task_info_data["first_lev_opt_costs"][0]

        self.n_int = self.task_info_data["nodes_list"][-1][-1] # getting last node index (will be equal to n_nodes - 1)
        self.first_lev_cand_indeces = self.task_info_data["first_lev_cand_inds"][0]
        self.first_lev_best_qcodes_candidates = self.task_info_data["first_lev_best_candidates"]
        self.unique_id = self.task_info_data["unique_id"][0]
        self.first_lev_man_measure = compute_man_index(self.first_lev_opt_costs, self.n_int)
        # self.was_class_man_used = bool(self.task_info_data["use_classical_man"][0][0])

        # if (self.was_class_man_used):

        #     raise Exception("You still have to implement the case where the cl. man is used!!!")

        self.loaders = [] * self.n_clust # list of loaders (one for each cluster)
        self.second_lev_opt_costs = [] 
        self.second_lev_man_measure = []
        self.opt_mult_indeces = [] # index of the solution (w.r.t. the n multistarts per cluster)

        self.load_clust_sols()
        
        self.confidence_coeffs = [-1] * self.n_clust
        self.compute_conf_coeff()

        self.second_lev_true_costs = [-1] * self.n_clust
        self.second_lev_true_man = [-1] * self.n_clust
        self.n_of_improved_costs = 0
        self.compute_second_lev_true_costs()

        self.second_lev_weighted_costs = [-1] * self.n_clust
        self.compute_weighted_costs()
        
        self.rmse_man_meas = [-1] * self.n_clust
        self.rmse_opt_cost = [-1] * self.n_clust
        self.compute_rmse()

        self.best_second_lev_cost, self.best_second_lev_cl_index,\
            self.best_second_lev_man_measure, self.best_second_lev_qcodes = self.compute_second_lev_best_sol()

        self.best_second_lev_weight_cost, self.best_second_lev_weight_cl_index,\
            self.best_second_lev_weight_man_measure, self.best_second_lev_weight_qcodes =\
                self.compute_second_lev_best_sol(use_weighted=True)

        self.print_best_sol()
        self.print_best_sol(weighted=True)

        self.dump_results()

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

    def compute_weighted_costs(self):

        for cl in range(self.n_clust):

            self.second_lev_weighted_costs[cl] = self.second_lev_true_costs[cl] / self.confidence_coeffs[cl]

    def compute_rmse(self):
        
        for cl in range(self.n_clust):
            
            min_man_cl = self.second_lev_true_man[cl]
            sum_sqrd_man = 0.0
            sum_sqrd_cost = 0.0
            n_opt_ms_cl = len(self.second_lev_man_measure[cl])
            for ms_sample in range(n_opt_ms_cl):
                
                sum_sqrd_man = sum_sqrd_man + (self.second_lev_man_measure[cl][ms_sample] - min_man_cl)**2
                sum_sqrd_cost = sum_sqrd_cost + (self.second_lev_opt_costs[cl][ms_sample] - min_man_cl)**2

            self.rmse_man_meas[cl] = np.sqrt(sum_sqrd_man / n_opt_ms_cl)
            self.rmse_opt_cost[cl] = np.sqrt(sum_sqrd_cost / n_opt_ms_cl)

    def compute_second_lev_best_sol(self, use_weighted = False):

        best_second_lev_qcodes = np.zeros((len(self.first_lev_best_qcodes_candidates), 1)).flatten()

        if not use_weighted:
            
            best_second_lev_cost = min(self.second_lev_true_costs)
            best_second_lev_cl_index = \
                np.argwhere(np.array(self.second_lev_true_costs) == min(self.second_lev_true_costs))[0][0]

            best_second_lev_man_measure = compute_man_index([best_second_lev_cost], self.n_int)[0]

            for i in range(len(best_second_lev_qcodes)):

                best_second_lev_qcodes[i] = self.first_lev_best_qcodes_candidates[i][best_second_lev_cl_index]

        else:   

            best_second_lev_cost = min(self.second_lev_weighted_costs)
            best_second_lev_cl_index = \
                np.argwhere(np.array(self.second_lev_weighted_costs) == min(self.second_lev_weighted_costs))[0][0]

            best_second_lev_man_measure = compute_man_index([min(self.second_lev_weighted_costs)], self.n_int)[0]
            
            for i in range(len(best_second_lev_qcodes)):

                best_second_lev_qcodes[i] = self.first_lev_best_qcodes_candidates[i][best_second_lev_cl_index]

        return best_second_lev_cost, best_second_lev_cl_index,\
                best_second_lev_man_measure, best_second_lev_qcodes

    def compute_second_lev_true_costs(self):

        for cl in range(self.n_clust):
            
            min_second_lev_cl = np.min(np.array(self.second_lev_opt_costs[cl]))

            self.second_lev_true_costs[cl] = \
                min_second_lev_cl if (min_second_lev_cl< self.first_lev_opt_costs[cl]) else self.first_lev_opt_costs[cl]

        self.second_lev_true_man = compute_man_index(self.second_lev_true_costs, self.n_int)

        did_cost_improve = np.array(self.first_lev_opt_costs != self.second_lev_true_costs)
        self.n_of_improved_costs = len(np.argwhere(did_cost_improve == True).flatten())

    def compute_conf_coeff(self):
        
        opt_sols_n_aux_list = []
        for cl in range(self.n_clust):
            opt_sols_n_aux_list.append(len(self.second_lev_opt_costs[cl]))

        max_n_cl_opt_sol = np.max(np.array(opt_sols_n_aux_list))

        self.confidence_coeffs = [opt_sols_n_aux_list[cl]/max_n_cl_opt_sol for cl in range(self.n_clust)]

    def load_clust_sols(self):

        for cl in range(self.n_clust):
            
            self.loaders.append(LoadSols(self.load_path + self.clust_dir_basename + str(cl))) 
            #solutions are loaded mantaining cluster order

            opt_cost_aux_list = []
            opt_man_meas_aux_list = []
            opt_mult_aux_indeces = []

            for opt_sol_index in range(len(self.loaders[cl].opt_data)):

                opt_cost_aux_list.append(self.loaders[cl].opt_data[opt_sol_index]["opt_cost"][0][0])
                opt_mult_aux_indeces.append(self.loaders[cl].opt_data[opt_sol_index]["multistart_index"][0][0])
            
            self.second_lev_opt_costs.append(opt_cost_aux_list)
            self.second_lev_man_measure.append(compute_man_index(opt_cost_aux_list, self.n_int))

            self.opt_mult_indeces.append(opt_mult_aux_indeces)

    def dump_results(self):

        if not os.path.isdir(self.dump_path):

            os.makedirs(self.dump_path)

        task_info_dumper = SolDumper()

        stuff = {"unique_id": self.unique_id,\
                    "n_clust": self.n_clust,
                    "n_int": self.n_int,
                    "first_lev_cand_indeces": self.first_lev_cand_indeces,
                    "first_lev_best_qcodes_candidates": self.first_lev_best_qcodes_candidates,
                    "first_lev_opt_costs": self.first_lev_opt_costs, 
                    "first_lev_man_measure": self.first_lev_man_measure,
                    "second_lev_opt_costs": np.array(self.second_lev_opt_costs, dtype=object),
                    "second_lev_man_measure": np.array(self.second_lev_man_measure, dtype=object),
                    "opt_mult_indeces": np.array(self.opt_mult_indeces, dtype=object),
                    "confidence_coeffs": self.confidence_coeffs,
                    "second_lev_true_costs": self.second_lev_true_costs,
                    "n_of_improved_costs": self.n_of_improved_costs,
                    "second_lev_weighted_costs": self.second_lev_weighted_costs,
                    "best_second_lev_cost": self.best_second_lev_cost,
                    "best_second_lev_cl_index": self.best_second_lev_cl_index,
                    "best_second_lev_man_measure": self.best_second_lev_man_measure,
                    "best_second_lev_qcodes": self.best_second_lev_qcodes,
                    "best_second_lev_weight_cost": self.best_second_lev_weight_cost,
                    "best_second_lev_weight_cl_index": self.best_second_lev_weight_cl_index,
                    "best_second_lev_weight_man_measure": self.best_second_lev_weight_man_measure,
                    "best_second_lev_weight_qcodes": self.best_second_lev_weight_qcodes
                    }

        task_info_dumper.add_storer(stuff, self.dump_path,\
                                "2nd_lev_postproc" + str(self.unique_id),\
                                False)        

        task_info_dumper.dump()




