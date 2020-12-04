import numpy as np
import mido
import csv
import pickle
import progressbar



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
            if  row['split'] == "train":
                train.append(row['midi_filename'])
            elif  row['split'] == "test":
                test.append(row['midi_filename'])
            elif  row['split'] == "validation":
                val.append(row['midi_filename'])
    return train, val, test


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

            if msg.velocity != 0:
                n_on[msg.note] = 1.0

            if msg.velocity == 0:
                n_off[msg.note] = 1.0

            velocity_lst.append(msg.velocity)
            # tbh not sure how this should work, should double check that
            for time_shift in time_shifts:
                data.append((n_on, n_off, velocity, time_shift))

            i += 1

    encoding = np.zeros(shape=(len(data), 413))

    # normalizing velocity values for a midi file and placing the concatenated one hot into the final array
    for i, (n_on, n_off, velocity, time_shift) in enumerate(data):
        normalized_velocity = int(np.floor(((velocity_lst[i] - min(velocity_lst)) / (max(velocity_lst) - min(velocity_lst))) * 31))
        velocity[normalized_velocity] = 1.0
        encoding[i] = np.concatenate((n_on, n_off, velocity, time_shift), axis=0)

    return encoding


def get_raw_data(filenames):
    '''
    Retrieves and encodes data from MIDI files based on list of filenaes
    :param filenames: list of filenames
    :return: train, val, test: training data, validation data, testing data in a 80/10/10 split
    '''

    widgets = [
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.AdaptiveETA(), ') ',
    ]
    data = []

    for i in progressbar.progressbar(range(len(filenames)), widgets=widgets):
            midi_file = mido.MidiFile(filenames[i], clip=True)
            encoding = create_encoding(midi_file)
            data.append(encoding)

    return data

def get_data(train_file="data/train", val_file="data/val",test_file="data/test",
             csv_file="maestro-v2.0.0.csv", csv_prefix="data/", save_data=True):
    '''
    Retrives data from pickled files if exists, otherwise loads in from csv and pickles
    :return: train data, val data,  test data,
    '''

    try:
        train, val, test = get_pickled_data(train_file, val_file, test_file)
    except (FileNotFoundError, EOFError):
        train_filenames, val_filenames, test_filenames = files_from_csv(csv_prefix+csv_file)
        train_filenames = list(map(lambda x: csv_prefix + x, train_filenames))
        val_filenames = list(map(lambda x: csv_prefix + x, val_filenames))
        test_filenames = list(map(lambda x: csv_prefix + x, test_filenames))



        train = get_raw_data(train_filenames)
        val = get_raw_data(val_filenames)
        test = get_raw_data(test_filenames)
        with open(train_file, "wb") as trainfile:
            pickle.dump(train, trainfile)

        with open(test_file, "wb") as testfile:
            pickle.dump(test, testfile)

        with open(val_file, "wb") as valfile:
            pickle.dump(val, valfile)

    return train, val, test
def unpickle(file):
    with open(file, 'rb') as fo:
        array = pickle.load(fo, encoding='bytes')
    return array

def get_pickled_data(train_file, val_file, test_file):
    train = unpickle(train_file)
    test = unpickle(test_file)
    val = unpickle(val_file)
    return train, val, test

def main():
    train, test, val = get_data()
    print(len(train))
    print(len(test))
    print(len(val))
if __name__ == '__main__':
    main()