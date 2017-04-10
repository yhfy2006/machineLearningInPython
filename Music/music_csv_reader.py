__author__ = 'chhe'
import csv
import numpy as np
import keras.utils as util
from music_unit import Music_unit

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
                music_unit = Music_unit(time = time,event_str=event, note = note, velocity= velocity)
                raw_data_list.append(music_unit)

        sorted_raw = sorted(raw_data_list, key=lambda tup: music_unit.time)

        init_time = 0

        for music_unit in sorted_raw:
            vec = self._translate_to_vec(music_unit.time-init_time,music_unit.event,music_unit.note,music_unit.velocity)
            init_time = music_unit.time
            if init_time is not 0:
                print(music_unit.time-init_time,music_unit.event,music_unit.note,music_unit.velocity)
                self.vec_to_note(vec)
                break


        f.close()

    def _translate_to_vec(self,rel_time,event,note,vel):
        vec = np.zeros(14)
        # 0-13  time to last note
        # 14-15 event
        # 16-116 note
        # 117-244 vel
        time_bin = "{0:b}".format(rel_time)
        time_bin_reverse = time_bin[::-1]
        for index in range(len(time_bin_reverse)):
            value = int(time_bin_reverse[index])
            vec[13-index] = value

        # event
        event_digits = util.to_categorical(event,num_classes=Music_unit.music_events_len())
        vec = np.append(vec,event_digits)

        #notes
        notes_digits = util.to_categorical(note,num_classes=Music_unit.music_note_len())
        vec = np.append(vec,notes_digits)

        #vel
        vel_digits = util.to_categorical(vel,num_classes=Music_unit.music_vel_len())
        vec = np.append(vec, vel_digits)
        return vec

    def vec_to_note(self,vec):
        # 0-13  time to last note
        pos = 14
        time_bin = vec[0:pos]
        time_str = ''
        for i in time_bin:
            time_str += str(int(i))
        relative_time_int = int(time_str,2)

        event_int = np.argmax(vec[pos:pos+Music_unit.music_events_len()])
        pos += Music_unit.music_events_len()


        note_int = np.argmax(vec[pos:pos+Music_unit.music_note_len()])
        pos += Music_unit.music_note_len()

        vel_int = np.argmax(vec[pos:pos+Music_unit.music_vel_len()])

        return relative_time_int,event_int,note_int,vel_int










def main():
    cm = CSV_Manager()
    cm.process_file()

    print(util.to_categorical(1,num_classes=2))



if __name__ == '__main__':
    main()