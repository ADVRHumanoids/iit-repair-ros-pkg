import numpy as np

import h5py

import matplotlib.pyplot as plt

class LogLoader:

    def __init__(self, mat_file_path):

        self.mat_file_h5py = h5py.File(mat_file_path,'r') # read the .mat file using the H5 .mat Python reader 
        
        self.data = {} # empty dictionary for storing data

        self.mat_file_h5py.visit(self.__h5py_visit_callable) # sliding through the loaded file and processing its fields

    ## Low-level methods for manipulating the .mat file ##

    def __read_str_from_ds(self, ds, index): 

        """

        Reads a string of a string cell array from a h5py database, given the index to be read. Only works with one-dimensional cell arrays of strings.
        
        Args:
            ds: dataset (see h5py docs)
            index: index to be read

        """

        read_str = ""

        try:
            ref = ds[0][index] # list of references to each object in the dataset
            st = self.mat_file_h5py[ref]

            read_str = ''.join(chr(i[0]) for i in st[:])

        except:
            print("ERROR: Could not extract a string from the provided h5py database. \n")

        return read_str

    def __h5py_visit_callable(self, name):

        """

        Callable function passed to the h5py visit() method. 
        Used to perform some useful initializations of the class attributes.

        Args:
            name: field names in the .mat file

        """

        if 'plugin_dt' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name])[0][0]   # add the pair key-value to the dictionary
        
        if 'plugin_time' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).flatten()   # add the pair key-value to the dictionary
        
        if 'q_p_cmd' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary
        
        if 'q_p_dot_cmd' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary
        
        if 'tau_cmd' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T  # add the pair key-value to the dictionary
        
        if 'q_p_meas' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary
        
        if 'q_p_dot_meas' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary

        if 'tau_meas' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary
        
        if 'traj_ref_time_vector' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).flatten()   # add the pair key-value to the dictionary

        if 'q_p_ref' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary

        if 'q_p_dot_ref' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary

        if 'tau_ref' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary

        if 'replay_damping' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).flatten()   # add the pair key-value to the dictionar

        if 'replay_stiffness' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).flatten()   # add the pair key-value to the dictionary

        if 'traj_dt_before_res' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).flatten()   # add the pair key-value to the dictionary
        
        if 'q_p_bf_res' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T  # add the pair key-value to the dictionary
        
        if 'q_p_dot_bf_res' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T   # add the pair key-value to the dictionary
        
        if 'tau_bf_res' in name: # assigns the aux names and their associated codes to a dictionary

            self.data[name] = np.array(self.mat_file_h5py[name]).T  # add the pair key-value to the dictionary

        return None # any other value != to None would block the execution of visit() method

class LogPlotter:

    def __init__(self, data: dict):
        
        self._n_arm_dofs = 7
        self._n_arms = 2

        self._data = data

        self.__extract_data()

    def __extract_data(self):

        self._plugin_dt = self._data["plugin_dt"]
        self._q_p_cmd = self._data["q_p_cmd"]
        self._q_p_dot_cmd = self._data["q_p_dot_cmd"]
        self._tau_cmd = self._data["tau_cmd"]
        self._q_p_meas = self._data["q_p_meas"]
        self._q_p_dot_meas = self._data["q_p_dot_meas"]
        self._tau_meas= self._data["tau_meas"]

        self._traj_ref_time = self._data["traj_ref_time_vector"]

        self._traj_dt_before_res = self._data["traj_dt_before_res"]
        self._q_p_bf_res = self._data["q_p_bf_res"]
        self._q_p_dot_bf_res = self._data["q_p_dot_bf_res"]
        self._tau_bf_res = self._data["tau_bf_res"]
        
        self._q_p_ref = self._data["q_p_ref"]
        self._q_p_dot_ref= self._data["q_p_dot_ref"]
        self._tau_ref = self._data["tau_ref"]

        self._replay_damping= self._data["replay_damping"]
        self._replay_stiffness = self._data["replay_stiffness"]

        self._n_jnts = len(self._q_p_cmd[:, 0])
        self._n_samples = len(self._q_p_cmd[0, :])
        self._n_samples_ref_traj = len(self._q_p_ref[0, :])

        self._plugin_time = self.__compute_plugin_time_vector(self._q_p_cmd)
        
        self._time_vect_bf_res = self.__compute_time_vect_before_res()

        self.__split_data()

    def __separate_data(self, data):

        x_data = data[0, :]
        arm2_data = data[1:self._n_arm_dofs + 1, :]
        arm1_data = data[self._n_arm_dofs + 1: (self._n_arms * self._n_arm_dofs + 1) , :]

        return x_data, arm1_data, arm2_data

    def __compute_time_vect_before_res(self):

        n_samples_bf_res = len(self._traj_dt_before_res) + 1

        time_vector = np.zeros((1, n_samples_bf_res)).flatten()

        for i in range(n_samples_bf_res - 1):

            time_vector[i + 1] =  time_vector[i] + self._traj_dt_before_res[i]
        
        return time_vector
    
    def __compute_plugin_time_vector(self, data):

        time_vector = np.zeros((1, len(data[0, :]))).flatten()
        
        for i in range(self._n_samples - 1):
            
            time_vector[i + 1]  = time_vector[i] + self._plugin_dt

        return time_vector

    def __split_data(self):

        self._joint_names = []
        self._joint_names1 = []
        self._joint_names2 = []

        self._joint_names.append("x_joint")
        for arm in range(2):

            for jnt in range(self._n_arm_dofs):

                self._joint_names.append("arm_" + str(arm + 1) + "_joint_" + str(jnt + 1))

                if arm == 0:
                    self._joint_names1.append("arm_" + str(arm + 1) + "_joint_" + str(jnt + 1))
                else:
                    self._joint_names2.append("arm_" + str(arm + 1) + "_joint_" + str(jnt + 1))


        self._q_p_cmd_x, self._q_p_cmd1, self._q_p_cmd2 = self.__separate_data(self._q_p_cmd)
        self._q_p_dot_cmd_x, self._q_p_dot_cmd1, self._q_p_dot_cmd2 = self.__separate_data(self._q_p_dot_cmd)
        self._tau_cmd_x, self._tau_cmd1, self._tau_cmd2 = self.__separate_data(self._tau_cmd)

        self._q_p_meas_x, self._q_p_meas1, self._q_p_meas2 = self.__separate_data(self._q_p_meas)
        self._q_p_dot_meas_x, self._q_p_dot_meas1, self._q_p_dot_meas2 = self.__separate_data(self._q_p_dot_meas)
        self._tau_meas_x, self._tau_meas1, self._tau_meas2 = self.__separate_data(self._tau_meas)
        
        self._q_p_ref_x, self._q_p_ref1, self._q_p_ref2 = self.__separate_data(self._q_p_ref)
        self._q_p_dot_ref_x, self._q_p_dot_ref1, self._q_p_dot_ref2 = self.__separate_data(self._q_p_dot_ref)
        self._tau_ref_x, self._tau_ref1, self._tau_ref2 = self.__separate_data(self._tau_ref)

        self._q_p_bf_res_x, self._q_p_bf_res1, self._q_p_bf_res2 = self.__separate_data(self._q_p_bf_res)
        self._q_p_dot_bf_res_x, self._q_p_dot_bf_res1, self._q_p_dot_bf_res2 = self.__separate_data(self._q_p_dot_bf_res)
        self._tau_bf_res_x, self._tau_bf_res1, self._tau_bf_res2 = self.__separate_data(self._tau_bf_res)

    def make_plots(self):
    
        self.__make_pos_plots()

        self.__make_interp_plot()

    def __make_interp_plot(self):
    
        # sol time
        leg_title = "Joint names:"
        _, ax_sol_t = plt.subplots(3)
        for i in range(self._n_arm_dofs):

            ax_sol_t[0].plot(self._time_vect_bf_res, self._q_p_bf_res1[i, :], label = self._joint_names1[i],\
                marker= "o", linestyle='', markersize=3)

        for i in range(self._n_arm_dofs):

            ax_sol_t[0].plot(self._traj_ref_time, self._q_p_ref1[i, :], label = self._joint_names1[i] + " resampled",\
                linestyle='-', linewidth=2, markersize=3)
        leg_t = ax_sol_t[0].legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        # ax_sol_t[0].set_xlabel(r"time [s]")
        ax_sol_t[0].set_ylabel(r"joint position [rad]")
        ax_sol_t[0].set_title(r"Arm 1 raw VS linear interpolated positions", fontdict=None, loc='center')
        ax_sol_t[0].grid()

        for i in range(self._n_arm_dofs):

            ax_sol_t[1].plot(self._time_vect_bf_res, self._q_p_bf_res2[i, :], label = self._joint_names1[i],\
                marker= "o", linestyle='', markersize=3)

        for i in range(self._n_arm_dofs):

            ax_sol_t[1].plot(self._traj_ref_time, self._q_p_ref2[i, :], label = self._joint_names1[i] + " resampled",\
                linestyle='-', linewidth=2, markersize=3)
        leg_t = ax_sol_t[1].legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        # ax_sol_t[1].set_xlabel(r"time [s]")
        ax_sol_t[1].set_ylabel(r"joint position [rad]")
        ax_sol_t[1].set_title(r"Arm 1 raw VS linear interpolated positions", fontdict=None, loc='center')
        ax_sol_t[1].grid()

        ax_sol_t[2].plot(self._time_vect_bf_res, self._q_p_bf_res_x, label = self._joint_names1[i],\
                marker= "o", linestyle='', markersize=3)
        ax_sol_t[2].plot(self._traj_ref_time, self._q_p_ref_x, label = self._joint_names1[i] + " resampled",\
                linestyle='-', linewidth=2, markersize=3)
        leg_t = ax_sol_t[2].legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        ax_sol_t[2].set_xlabel(r"time [s]")
        ax_sol_t[2].set_ylabel(r"joint position [rad]")
        ax_sol_t[2].set_title(r"Sliding guide raw VS linear interpolated positions", fontdict=None, loc='center')
        ax_sol_t[2].grid()
    

    def __make_pos_plots(self):

        # sol time
        leg_title = "Joint names:"
        _, ax_sol_t = plt.subplots(3)
        for i in range(self._n_arm_dofs):

            ax_sol_t[0].plot(self._plugin_time, self._q_p_meas1[i, :], label = self._joint_names1[i],\
                linewidth=2, markersize=12)

        for i in range(self._n_arm_dofs):

            ax_sol_t[0].plot(self._plugin_time, self._q_p_cmd1[i, :], label = self._joint_names1[i] + " reference",\
                linestyle='dashed', linewidth=2, markersize=12)
        leg_t = ax_sol_t[0].legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        # ax_sol_t[0].set_xlabel(r"time [s]")
        ax_sol_t[0].set_ylabel(r"joint position [rad]")
        ax_sol_t[0].set_title(r"Arm 1 joint positions VS references", fontdict=None, loc='center')
        ax_sol_t[0].grid()

        for i in range(self._n_arm_dofs):

            ax_sol_t[1].plot(self._plugin_time, self._q_p_meas2[i, :], label = self._joint_names2[i],\
                linewidth=2, markersize=12)

        for i in range(self._n_arm_dofs):

            ax_sol_t[1].plot(self._plugin_time, self._q_p_cmd2[i, :], label = self._joint_names2[i] + " reference",\
                linestyle='dashed', linewidth=2, markersize=12)
        leg_t = ax_sol_t[1].legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        # ax_sol_t[1].set_xlabel(r"time [s]")
        ax_sol_t[1].set_ylabel(r"joint position [rad]")
        ax_sol_t[1].set_title(r"Arm 2 joint positions VS references", fontdict=None, loc='center')
        ax_sol_t[1].grid()

        ax_sol_t[2].plot(self._plugin_time, self._q_p_meas_x, label = self._joint_names[0],\
            linewidth=2, markersize=12)
        ax_sol_t[2].plot(self._plugin_time, self._q_p_cmd_x, label = self._joint_names[0] + " reference",\
            linestyle='dashed', linewidth=2, markersize=12)
        leg_t = ax_sol_t[2].legend(loc="upper right", 
            title = leg_title)
        leg_t.set_draggable(True)
        ax_sol_t[2].set_xlabel(r"time [s]")
        ax_sol_t[2].set_ylabel(r"joint position [rad]")
        ax_sol_t[2].set_title(r"Sliding guide position VS reference", fontdict=None, loc='center')
        ax_sol_t[2].grid()

    def show_plots(self):

        plt.show()