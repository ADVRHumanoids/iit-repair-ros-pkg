import numpy as np

def str2bool(v):
  #susendberg's function
  return v.lower() in ("yes", "true", "t", "1")

# def PoseSampler():

#   def __init__(self, y_range, n_y_samples, z_pick):

#     self.y_range = y_range

#     self.n_y_samples = n_y_samples

#     if n_y_samples == 1:
      
#       self.intervals = - 1
    
#     else:

#       self.intervals = self.y_range / (n_y_samples - 1)

#     self.z_pick = z_pick

#     self.seed_pos = np.array([0.0, 0.0, 0.0])
#     self.seed_q = np.array([0.0, - 1.0, 0.0, 0.0])

#     self.Pick_pos = []
#     self.Pick_q = []

#     if n_y_samples % 2 != 0:

#       self.Pick_pos.append(self.seed_pos)
#       self.Pick_pos.append(self.seed_q)

#       for i in range((n_y_samples - 1) / 2):

#         self.Pick_pos.append(self.seed_pos + np.array([0,  ]))
#         self.Pick_pos.append(self.seed_q)

#     else:

#         self.Pick_pos.append(self.seed_pos)
#         self.Pick_rot.append()

    



