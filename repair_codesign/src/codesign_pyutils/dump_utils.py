from horizon.utils import mat_storer

from datetime import datetime
from datetime import date

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
  