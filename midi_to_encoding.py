import os
import numpy as np
import mido
import csv
import os



def files_from_csv(csv_filename):
    '''
    Uses the CSV index file to find all the filenames
    :param csv_filename: name of
    :return:
    '''
    filenames = []
    with open(csv_filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filenames.append(row['midi_filename'])
    return filenames


def create_encoding(mid):
    '''
    Creates an encoding of one musical performance
    :param mid:
    :return:
    '''

    # midi files often have multiple tracks but for now we just want the one thats the longest
    max_t = max(mid.tracks, key=len)

    data = []
    i = 0
    velocity_lst = []
    last_time = max_t[0].time
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
            time_shift[delta_t] = 1.0

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


def get_data(filenames):
    '''
    Retrieves and encodes data from MIDI files based on list of filenaes
    :param filenames: list of filenames
    :return: train, val, test: training data, validation data, testing data in a 80/10/10 split
    '''
    data = []
    for filename in filenames:
        midi_file = mido.MidiFile(filename, clip=True)
        encoding = create_encoding(midi_file)
        data.append(encoding)

        print(filename)
    train_ind = int(len(data) * .8)
    val_ind = int(len(data) * .1) + train_ind

    #TODO: figure out how we actually want the data
    train = data[0:train_ind]
    val = data[train_ind:val_ind]
    test = data[val_ind:]
    return train, val, test

def list_from_npz(npz_file):
    '''
    Theoretically this is a helper if I ever figure out the caching
    :param npz_file:
    :return:
    '''
    arrays = []
    file = np.load(npz_file, allow_pickle=True)
    for array_name in file.files:
        arrays.append(file[array_name])
    return arrays


def main():
    filenames = files_from_csv("data/maestro-v2.0.0.csv")
    filenames = list(map(lambda x: 'data/' + x, filenames))
    train, val, test = get_data(filenames[0:3])

    np.savez("train", train)
    np.savez('val', val)
    np.savez('test', test)

    if "train.npz" in os.listdir() and 'test.npz' in os.listdir() and "val.npz" in os.listdir():
        train1 =  list_from_npz('train.npz')
        test1 = list_from_npz('test.npz')
        val1 = list_from_npz('val.npz')


    print(train[0])
    print(train1[0])
    print(np.array_equal(np.array(train[0]), train1[0]))
if __name__ == '__main__':
    main()