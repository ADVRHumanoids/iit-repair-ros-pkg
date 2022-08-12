import os

from horizon.utils import mat_storer

from codesign_pyutils.miscell_utils import compute_man_measure

import numpy as np

from codesign_pyutils.dump_utils import SolDumper

class LoadSols():

    def __init__(self, base_sol_path, 
                opt_dir_name = "opt", \
                fail_dir_name = "failed", \
                additional_info_pattern = "info"):

        self.base_sol_path = base_sol_path
        self.opt_path = base_sol_path + "/" + opt_dir_name
        self.fail_path = base_sol_path + "/" + fail_dir_name

        # read sol paths from directory
        additional_info_name = additional_info_pattern
        additional_info_candidates = os.listdir(base_sol_path)
        self.add_info_filename = ""

        add_info_filename_aux = []

        for i in range(len(additional_info_candidates)):

            if additional_info_name in additional_info_candidates[i]:

                add_info_filename_aux.append(additional_info_candidates[i]) 

        if len(add_info_filename_aux) == 0:

            raise Exception("LoadSols: didn't find any solution information file matching pattern \"" +
                            additional_info_name + "\"" + "in base directory " + self.base_sol_path + ".\n" +
                            "Please provide a valid solution information file.")

        if len(add_info_filename_aux) > 1:

            raise Exception("LoadSols: too many solution information files provided.\n" + 
                            "Please make sure the loading directory only contains coherent data.")

        self.add_info_path = self.base_sol_path + "/" + add_info_filename_aux[0]
        self.task_info_data = mat_storer.matStorer(self.add_info_path).load()

        opt_file_list = os.listdir(self.opt_path)
        fail_file_list = os.listdir(self.fail_path)
        self.opt_full_paths = [""] * len(opt_file_list)
        self.fail_full_paths = [""] * len(fail_file_list)
        for i in range(len(opt_file_list)):

            self.opt_full_paths[i] = self.opt_path + "/" + opt_file_list[i]

        for i in range(len(fail_file_list)):

            self.fail_full_paths[i] = self.fail_path + "/" + fail_file_list[i]

        self.opt_data = [None] * len(opt_file_list)
        self.fail_data = [None] * len(fail_file_list)

        for i in range(len(opt_file_list)):

            self.opt_data[i] = mat_storer.matStorer(self.opt_full_paths[i]).load()
        
        for i in range(len(fail_file_list)):

            self.fail_data[i] = mat_storer.matStorer(self.fail_full_paths[i]).load()

class PostProc2ndLev:

    def __init__(self, load_path, 
                clust_dir_basename = "clust", 
                additional_info_pattern = "info", 
                dump_dirname = "2nd_lev_postproc"):

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
        self.first_lev_man_measure = compute_man_measure(self.first_lev_opt_costs, self.n_int)
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
        self.n_of_improved_costs = 0
        self.compute_second_lev_true_costs()

        self.second_lev_weighted_costs = [-1] * self.n_clust
        self.compute_weighted_costs()

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
        
    def compute_second_lev_best_sol(self, use_weighted = False):

        best_second_lev_qcodes = np.zeros((len(self.first_lev_best_qcodes_candidates), 1)).flatten()

        if not use_weighted:
            
            best_second_lev_cost = min(self.second_lev_true_costs)
            best_second_lev_cl_index = \
                np.argwhere(np.array(self.second_lev_true_costs) == min(self.second_lev_true_costs))[0][0]

            best_second_lev_man_measure = compute_man_measure([best_second_lev_cost], self.n_int)[0]

            for i in range(len(best_second_lev_qcodes)):

                best_second_lev_qcodes[i] = self.first_lev_best_qcodes_candidates[i][best_second_lev_cl_index]

        else:   

            best_second_lev_cost = min(self.second_lev_weighted_costs)
            best_second_lev_cl_index = \
                np.argwhere(np.array(self.second_lev_weighted_costs) == min(self.second_lev_weighted_costs))[0][0]

            best_second_lev_man_measure = compute_man_measure([min(self.second_lev_true_costs)], self.n_int)[0]
            
            for i in range(len(best_second_lev_qcodes)):

                best_second_lev_qcodes[i] = self.first_lev_best_qcodes_candidates[i][best_second_lev_cl_index]

        return best_second_lev_cost, best_second_lev_cl_index,\
                best_second_lev_man_measure, best_second_lev_qcodes

    def compute_second_lev_true_costs(self):

        for cl in range(self.n_clust):
            
            min_second_lev_cl = np.min(np.array(self.second_lev_opt_costs[cl]))

            self.second_lev_true_costs[cl] = \
                min_second_lev_cl if (min_second_lev_cl< self.first_lev_opt_costs[cl]) else self.first_lev_opt_costs[cl]

        self.second_lev_true_man = compute_man_measure(self.second_lev_true_costs, self.n_int)

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
                opt_mult_aux_indeces.append(self.loaders[cl].opt_data[opt_sol_index]["solution_index"][0][0])
            
            self.second_lev_opt_costs.append(opt_cost_aux_list)
            self.second_lev_man_measure.append(compute_man_measure(opt_cost_aux_list, self.n_int))

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

