
from horizon.utils import mat_storer

from datetime import datetime
from datetime import date


def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

def wait_for_confirmation(do_something = "proceed",\
                          or_do_something_else = "stop",\
                          on_confirmation = "Confirmation received!",\
                          on_denial = "Stop received!"):

  usr_input = input("\n \n Press Enter to " + do_something + \
                    " or type \"N/n\" to " + or_do_something_else + ". \n -> ")

  if usr_input == "":

      print("\n" + on_confirmation + "\n")
      go_on = True
  
  else:

      if (usr_input == "no" or usr_input == "No" or usr_input == "N" or usr_input == "n"):
        
        print("\n" +  on_denial + "\n")
        go_on = False
        
      else: # user typed something else

        print("\n Ops.. you did not type any of the options! \n")

        go_on = wait_for_confirmation()

  return go_on

def check_str_list(comp_list = ["x", "y", "z"], input = []):
  
  presence_array = [False] * len(comp_list)

  for i in range(len(comp_list)):

    for j in range(len(input)):

      if (input[j] == comp_list[i] or input[j] == comp_list[i].upper()) and presence_array[i] != True:

        presence_array[i] = True

  return presence_array

class SolDumper():

  def __init__(self, dump_path, backend_name = "matStorer"):

    self.sols = []
    self.storers = []

    self.sols_counter = 0

    self.unique_id = date.today().strftime("%d-%m-%Y") + "-" +\
                     datetime.now().strftime("_%H_%M_%S")

    self.backend_name = backend_name

    self.backend_list = ["matStorer"]

    self.storer_map = {}

  def add_storer(self, sol_dict, results_path = "/tmp",\
                 file_name = "SolDumper", add_unique_id = True):

    self.sols_counter = self.sols_counter + 1

    self.storer_map[file_name] = self.sols_counter - 1 # 0-based indexing

    self.sols.append(sol_dict)

    if self.backend_name == "matStorer":

      if add_unique_id:

        self.storers.append(mat_storer.matStorer(results_path + "/" + \
                            file_name + "-" + str(self.sols_counter) + \
                            "-" + self.unique_id + ".mat"))

      else:

        self.storers.append(mat_storer.matStorer(results_path + "/" + \
                           file_name + "-" + str(self.sols_counter) +  \
                           ".mat"))

    else:

      raise Exception("\n Sorry, the chosen backend" + \
                      " is not supported.\n Supported backends: " + \
                      self.backend_list) 

  def dump(self):
    
    for i in range(len(self.sols)):

      self.storers[i].store(self.sols[i])
    



    



