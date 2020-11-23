import os
import numpy as np
import mido


def create_encoding(mid):

    # midi files often have multiple tracks but for now we just want the one thats the longest
    max_t = max(mid.tracks, key=len)

    data = []
    i = 0
    velocity_lst = []
    for msg in max_t:
        # these midi files only have types note_on and control_change message types
        # the note_off types that the paper spoke about are represented as note_on message types with velocity=0
        if msg.type == "note_on":

            # there are 128 total notes
            n_on = np.zeros(shape=128)
            n_off = np.zeros(shape=128)

            # the number of values for velocity varies with the midi file so I think we need to normalize it
            velocity = np.zeros(shape=32)

            # i cant figure out how to do the time shifting stuff the paper spoke of
            # the messages have a parameter "time" but im not sure how to interpret it
            time_shift = np.zeros(shape=125)

            if msg.velocity != 0:
                n_on[msg.note] = 1.0

            if msg.velocity == 0:
                n_off[msg.note] = 1.0

            velocity_lst.append(msg.velocity)

            data.append((n_on, n_off, velocity, time_shift))

            i += 1

    encoding = np.zeros(shape=(len(data), 413))

    # normalizing velocity values for a midi file and placing the concatenated one hot into the final array
    for i, (n_on, n_off, velocity, time_shift) in enumerate(data):
        normalized_velocity = int(np.floor(((velocity_lst[i] - min(velocity_lst)) / (max(velocity_lst) - min(velocity_lst))) * 31))
        velocity[normalized_velocity] = 1.0
        encoding[i] = np.concatenate((n_on, n_off, velocity, time_shift), axis=0)

    return encoding


def main():
    mid = mido.MidiFile('/Users/jofeinstein/PycharmProjects/maestro/maestro-v2.0.0/2013/ORIG-MIDI_01_7_6_13_Group__MID--AUDIO_01_R1_2013_wav--4.midi', clip=True)
    create_encoding(mid)


if __name__ == '__main__':
    main()