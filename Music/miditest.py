# import midi
# # Instantiate a MIDI Pattern (contains a list of tracks)
# pattern = midi.Pattern()
# # Instantiate a MIDI Track (contains a list of MIDI events)
# track = midi.Track()
# # Append the track to the pattern
# pattern.append(track)
# # Instantiate a MIDI note on event, append it to the track
# on = midi.NoteOnEvent(tick=0, velocity=200, pitch=midi.G_3)
# track.append(on)
# # Instantiate a MIDI note off event, append it to the track
# #off = midi.NoteOffEvent(tick=500, pitch=midi.G_4)
# #track.append(off)
# # Add the end of track event, append it to the track
# eot = midi.EndOfTrackEvent(tick=100)
# track.append(eot)
#
# midi.SetTempoEvent(tick=0, data=[8, 63, 125]),
# midi.SetTempoEvent(tick=240, data=[8, 26, 29]),
# # Print out the pattern
# print (pattern)
# # Save the pattern to disk
# midi.write_midifile("example.mid", pattern)

from midiutil.MidiFile import MIDIFile

degrees  = [60, 62, 64, 65, 67, 69, 71, 72] # MIDI note number
track    = 0
channel  = 0
time     = 0   # In beats
duration = 1   # In beats
tempo    = 60  # In BPM
volume   = 100 # 0-127, as per the MIDI standard

MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                     # automatically created)
MyMIDI.addTempo(track,time, tempo)

for pitch in degrees:
    MyMIDI.addNote(track, channel, pitch, time, duration, volume)
    time = time + 1

with open("major-scale.mid", "wb") as output_file:
    MyMIDI.writeFile(output_file)