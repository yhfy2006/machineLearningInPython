

class Music_unit:

    music_events = {'Control_c':0,
                    'Note_on_c':1}
    music_events_inv = dict((v, k) for k, v in music_events.items())

    @staticmethod
    def music_events_len():
        return len(Music_unit.music_events)

    @staticmethod
    def music_note_len():
        return 127

    @staticmethod
    def music_vel_len():
        return 128

    def __init__(self,time,event_str,note,velocity):
        self.time = time
        self.event = self.event_to_digit(event_str)
        self.event_name = event_str
        self.note = note
        self.velocity = velocity

    def event_to_digit(self,event_str):
        return self.music_events[event_str]