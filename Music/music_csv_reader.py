__author__ = 'chhe'
import csv
import numpy as np

class CSV_Manager(object):

    def __init__(self,filename = 'alb1.csv'):
        self.csv_file = filename
        pass

    def process_file(self):
        f = open(self.csv_file)
        csv_f = csv.reader(f)
        raw_data_list = []
        for row in csv_f:
            row = [ i.strip() for i in row]
            event = row[2] #Note_on_c  #Control_c
            if event == 'Note_on_c' or event == 'Control_c':
                note = int(row[4])  # 0 to 100
                velocity = int(row[5])
                time = int(row[1])
                raw_data_tuple = (time,event,note,velocity)
                raw_data_list.append(raw_data_tuple)
                # raw data format (time,event,note,velocity)

        sorted_raw = sorted(raw_data_list, key=lambda tup: tup[0])

        init_time = 0

        for data in sorted_raw:
            self._translate_to_vec(data[0]-init_time,data[1],data[2],data[3])
            init_time = data[0]
            if init_time is not 0:
                break

        f.close()

    def _translate_to_vec(self,rel_time,event,note,vel):
        vec = np.zeros(14)
        # 0-13  time to last note
        # 14-15 event
        # 16-116 note
        # 116-243 vel
        time_bin = "{0:b}".format(rel_time)
        time_bin_reverse = time_bin[::-1]
        for index in range(len(time_bin_reverse)):
            value = int(time_bin_reverse[index])
            vec[13-index] = value
        print(rel_time,time_bin,time_bin_reverse,vec)
        





def main():
    cm = CSV_Manager()
    cm.process_file()



if __name__ == '__main__':
    main()