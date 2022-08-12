import os

from horizon.utils import mat_storer

from codesign_pyutils.miscell_utils import compute_man_measure
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

    def __init__(self, results_path, 
                clust_dir_basename = "clust", 
                additional_info_pattern = "info"):

        self.results_path = results_path + "/second_level/"
        self.clust_dir_basename = clust_dir_basename

        # read sol paths from directory
        additional_info_name = additional_info_pattern
        additional_info_candidates = os.listdir(self.results_path)
        self.add_info_filename = ""

        add_info_filename_aux = []

        for i in range(len(additional_info_candidates)):

            if additional_info_name in additional_info_candidates[i]:

                add_info_filename_aux.append(additional_info_candidates[i]) 

        if len(add_info_filename_aux) == 0:

            raise Exception("PostProc2ndLev: didn't find any solution information file matching pattern \"" +
                            additional_info_name + "\"" + "in base directory " + self.results_path + ".\n" +
                            "Please provide a valid solution information file.")

        if len(add_info_filename_aux) > 1:

            raise Exception("PostProc2ndLev: too many solution information files provided.\n" + 
                            "Please make sure the loading directory only contains coherent data.")

        self.add_info_path = self.results_path + "/" + add_info_filename_aux[0]
        self.task_info_data = mat_storer.matStorer(self.add_info_path).load()

        self.n_clust = self.task_info_data["n_clust"][0][0] # double loaded as matrix --> [0][0] necessary
        self.first_lev_opt_costs = self.task_info_data["first_lev_opt_costs"][0]
        self.n_int = self.task_info_data["nodes_list"][-1][-1] # getting last node index (will be equal to n_nodes - 1)
        self.first_lev_cand_indeces = self.task_info_data["first_lev_cand_inds"][0]
        self.first_lev_best_qcodes_candidates = self.task_info_data["first_lev_best_candidates"]
        self.unique_id = self.task_info_data["unique_id"]
        self.first_lev_man_measure = compute_man_measure(self.first_lev_opt_costs, self.n_int)
        
        self.loaders = [] * self.n_clust # list of loaders (one for each cluster)
        self.second_lev_opt_costs = [] 
        self.second_lev_man_measure = []
        self.opt_mult_indeces = [] # index of the solution (w.r.t. the n multistarts per cluster)

        self.load_clust_sols()
        
    def load_clust_sols(self):

        for cl in range(self.n_clust):
            
            self.loaders.append(LoadSols(self.results_path + self.clust_dir_basename + str(cl))) 
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


