import mido
import numpy as np
import os
import csv
import argparse

"""

script to create midi encodings in parallel

"""


def getArgs():
    parser = argparse.ArgumentParser('python')
    parser.add_argument('-dirr',
                        default=None,
                        required=False,
                        help='Directory containing files')
    parser.add_argument('-func',
                        default=None,
                        required=False)
    parser.add_argument('-in_path',
                        default='',
                        required=False)
    parser.add_argument('-out_path',
                        default='',
                        required=False)
    return parser.parse_args()


def create_encoding(mid):
    '''
    Creates an encoding of one musical performance
    :param mid:
    :return:
    '''

    # midi files often have multiple tracks but for now we just want the one thats the longest
    max_t = max(mid.tracks, key=len)

    data = []
    velocity_lst = []

    zeroed_note = np.zeros(shape=128)
    zeroed_velocity = np.zeros(shape=32)
    zeroed_time = np.zeros(shape=125)
    for msg in max_t:
        # these midi files only have types note_on and control_change message types
        # the note_off types that the paper spoke about are represented as note_on message types with velocity=0
        if msg.type == "note_on":

            # there are 128 total notes
            n_on = np.zeros(shape=128)
            n_off = np.zeros(shape=128)

            # the number of values for velocity varies with the midi file so I think we need to normalize it
            velocity = np.zeros(shape=32)

            # these objects have a 'time' attribute, which represents the change in time
            # if the change in time is above 125, we create a second time event
            time_shifts = []
            time_shift = np.zeros(shape=125)
            delta_t = msg.time

            while delta_t >= 125:
                time_shift[-1] = 1.0
                time_shifts.append(time_shift)
                time_shift = np.zeros(shape=125)
                # not sure if this should be 125 or 124
                delta_t -= 125

            # we werent adding the leftover time to anything
            time_shift[delta_t] = 1.0
            time_shifts.append(time_shift)

            if msg.velocity != 0:
                n_on[msg.note] = 1.0

            if msg.velocity == 0:
                n_off[msg.note] = 1.0

            # tbh not sure how this should work, should double check that
            data.append((n_on, n_off, velocity, zeroed_time))
            velocity_lst.append(msg.velocity)

            for time_shift in time_shifts:
                """
                i made it so that if the time extends past 125, we create a new vector with zeros 
                for n_on, n_off, and velocity. If we just made another vector with the same values for 
                these, it would be like another key press.
                """
                data.append((zeroed_note, zeroed_note, zeroed_velocity, time_shift))

                velocity_lst.append(msg.velocity)

    encoding = np.zeros(shape=(len(data), 413))

    # normalizing velocity values for a midi file and placing the concatenated one hot into the final array
    for i, (n_on, n_off, velocity, time_shift) in enumerate(data):
        normalized_velocity = int(np.floor(((velocity_lst[i] - min(velocity_lst)) / (max(velocity_lst) - min(velocity_lst))) * 31))
        velocity[normalized_velocity] = 1.0
        encoding[i] = np.concatenate((n_on, n_off, velocity, time_shift), axis=0)

    return encoding


def split_array(array, max_len=2000):
    """
    splits an array into n arrays of len max_len
    :param array:
    :param max_len:
    :return:
    """
    new_array_lst = []
    for x in range(0, len(array), max_len):
        split = array[x: x + max_len]
        if len(split) >= max_len:
            new_array_lst.append(split)

    return new_array_lst


def midi_to_encoding(in_path, out_path):
    """
    converts a midi to an encoding, splits the encoding, and saves it
    :param in_path: path to midi
    :param out_path: path to save
    :return:
    """
    midi_file = mido.MidiFile(in_path, clip=True)
    encoding = create_encoding(midi_file)
    array_lst = split_array(encoding)

    for i, a in enumerate(array_lst):
        updated_out_path = out_path + '_' + str(i)
        np.save(updated_out_path, a)


def files_from_csv(csv_filename):
    '''
    Uses the CSV index file to find all the filenames
    :param csv_filename: name of
    :return: train, val, test
    '''
    train = []
    test = []
    val = []

    with open(csv_filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['split'] == "train":
                train.append(row['midi_filename'].split('/')[-1])
            elif row['split'] == "test":
                test.append(row['midi_filename'].split('/')[-1])
            elif row['split'] == "validation":
                val.append(row['midi_filename'].split('/')[-1])
    return train, val, test


def task_lst_gen(dirr, csv_path):
    """
    creates a task list for use with GNU-parallel
    :param dirr:
    :param csv_path:
    :return:
    """
    train_file_lst, val_file_lst, test_file_lst = files_from_csv(csv_path)

    task_dict = {}
    out_prefix = '/work/jfeins1/maestro/dataset-v3/'
    for subdirs, dirs, files in os.walk(dirr):
        for file in files:
            filepath = subdirs + os.sep + file

            if file in train_file_lst:
                uid = str(file).split('.')[0]
                out = out_prefix + 'train/' + uid
                task_dict[uid] = {'in': filepath, 'out': out}

            if file in test_file_lst:
                uid = str(file).split('.')[0]
                out = out_prefix + 'test/' + uid
                task_dict[uid] = {'in': filepath, 'out': out}

            if file in val_file_lst:
                uid = str(file).split('.')[0]
                out = out_prefix + 'val/' + uid
                task_dict[uid] = {'in': filepath, 'out': out}

    task_lst = open('/work/jfeins1/maestro/encoding_gen_task.lst', 'w')
    for uid, d in task_dict.items():
        print(d['in'], d['out'], file=task_lst)


if __name__ == "__main__":

    args = getArgs()

    dirr = args.dirr
    in_path = args.in_path
    out_path = args.out_path
    func = args.func

    if func == 'task_lst_gen':
        task_lst_gen(dirr, '/work/jfeins1/maestro/data/maestro-v3.0.0.csv')

    elif func == 'generate':
        midi_to_encoding(in_path, out_path)

