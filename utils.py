import numpy as np
import mido
import statistics
import csv
import sklearn as sk
from sklearn import model_selection


def print_prgress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd=""):
    """
    Call in a loop to create terminal progress bar
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def one_hot_to_int(one_hot_vec):
    """
    Convert a one hot vector back to an integer

    :param one_hot_vec: numpy one hot vector
    :return: integer
    """
    return np.argmax(one_hot_vec, axis=0)


def encoding_to_midi(encoding, out_path, max_velocity=108):
    """
    Convert an encoding back to a midi file. Check recreation_test for results

    :param encoding: array that encodes a midi file
    :param out_path: path to save new midi file
    :param max_velocity: maximum velocity of dataset. calculated using calculate_median_velocity()
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    data = []
    velocity_lst = []
    for vector in encoding:
        n_on = one_hot_to_int(vector[0:128])
        n_off = one_hot_to_int(vector[128:256])
        velocity = one_hot_to_int(vector[256:288])
        time_shift = one_hot_to_int(vector[288:413])

        # if time_shift value >= 125, we need to add the time value to the original msg and not create a new msg
        if n_on == 0 and n_off == 0 and velocity == 0:
            data[-1]['time_shift'] += time_shift
        else:
            data.append({'n_on': n_on, 'n_off': n_off, 'velocity': velocity, 'time_shift': time_shift})
            velocity_lst.append(velocity)

    for i, msg in enumerate(data):

        # We normalized the velocity bw [0,32) while encoding.
        # Here we are renormalizing the velocity to a value
        # that is more representative of the velocities in the original dataset.
        normalized_velocity = int(np.floor(((velocity_lst[i] - min(velocity_lst)) / (max(velocity_lst) - min(velocity_lst))) * max_velocity))
        track.append(mido.Message(type='note_on',
                                  note=max(msg['n_on'], msg['n_off']),
                                  velocity=normalized_velocity,
                                  time=msg['time_shift']))

    mid.save(out_path)



def kfold_cv(csv_filename, num_folds=5):
    '''
    Function for 5 fold cross-validation, not done yet
    '''
    filenames = []
    with open(csv_filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filenames.append('encodings/' + row['midi_filename'])

    skf = sk.model_selection.StratifiedKFold(n_splits=5)
    y = np.zeros(shape=(len(filenames)))
    X = np.asarray(filenames)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]

        print(i, X_train[0], X_test[0])


def calculate_median_velocity(csv_filename):
    """
    calculates some statistics about the original velocities. I think we can use these to normalize the
    outputted velocities to the original scale.

    Results:
    Median Velocity of All Midi Files: 66.0
    Mean Velocity of All Midi Files: 63.88
    Max Velocity: 126
    Mean Max Velocity: 108.72


    :param csv_filename:
    :return:
    """
    filenames = []
    with open(csv_filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filenames.append('data/' + row['midi_filename'])

    median_velocity_lst = []
    mean_velocity_lst = []
    max_velocity_lst = []
    for i, file in enumerate(filenames):
        midi_file = mido.MidiFile(file, clip=True)

        max_t = max(midi_file.tracks, key=len)

        velocity_lst = []

        for msg in max_t:
            if msg.type == "note_on":
                if msg.velocity != 0:
                    velocity_lst.append(msg.velocity)

        median_velocity_lst.append(statistics.median(velocity_lst))
        mean_velocity_lst.append(statistics.mean(velocity_lst))
        max_velocity_lst.append(max(velocity_lst))

        print(i)

    print('Median Velocity of All Midi Files: {}'.format(statistics.median(median_velocity_lst)))
    print('Mean Velocity of All Midi Files: {}'.format(statistics.mean(mean_velocity_lst)))
    print('Max Velocity: {}'.format(max(max_velocity_lst)))
    print('Mean Max Velocity: {}'.format(statistics.mean(max_velocity_lst)))
