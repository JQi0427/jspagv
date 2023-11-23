import pandas as pd
import numpy as np


class Encode:
    def __init__(self, pt, ms, agv, J_num, M_num, A_num, population_size, num):
        self.pt = pt  # processing time
        self.ms = ms  # machine sequence
        self.agv = agv
        self.J_num = J_num
        self.M_num = M_num
        self.A_num = A_num
        self.population_size = population_size
        self.num = num

    def initMachineSequence(self):
        for i in range(self.population_size):
            nxm_random_num = list(
                np.random.permutation(self.num))  # generate a random permutation of 0 to num_job*num_mc-1
            self.ms.append(nxm_random_num)  # add to the machine_sequence
            for j in range(self.num):
                self.ms[i][j] = self.ms[i][
                                            j] % self.J_num  # convert to job number format, every job appears m times
        return self.ms

    def initAGVSequence(self):
        for i in range(self.population_size):
            nxm_random_num = list(
                np.random.permutation(self.num))  # generate a random permutation of 0 to num_job*num_mc-1
            self.agv.append(nxm_random_num)  # add to the machine_sequence
            for j in range(self.num):
                self.agv[i][j] = self.agv[i][
                                            j] % self.A_num  # convert to job number format, every job appears m times
        return agv


pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])
at_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="AGV Time", index_col=[0])

# dfshape = pt_tmp.shape
# num_mc = dfshape[1]  # number of machines
# num_job = dfshape[0]  # number of jobs

J_num = 10
M_num = 10
A_num = 3
population_size = 1
num = J_num * M_num
pt = [list(map(int, pt_tmp.iloc[i])) for i in range(J_num)]  # process time
ms = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # machine sequence
agv = [list(map(int, ms_tmp.iloc[i])) for i in range(J_num)]  # AGV sequence

JSPAGV = Encode(pt, ms, agv, J_num, M_num, A_num, population_size, num)

print(JSPAGV.initMachineSequence())
print(JSPAGV.initAGVSequence())
