import os

from horizon.utils import mat_storer

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

# class PostProc2ndLev:

#     def __init__(self, results_path, 
#                 clust_dir_basename = "clust"):

#         self.results_path = results_path + "/second_level/"
#         self.loaders = []


#     def load_2nd_l_sols(self):

#         self.loader.append()



        
