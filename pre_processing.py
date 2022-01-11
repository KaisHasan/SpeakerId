import numpy as np
import os
from scipy.io import wavfile
from scipy import segnal
import librosa
import scipy
import joblib
from tqdm import tqdm
import random
import gc


def convert_to_16bits(dataset_directory: str,
                      dataset_converted: str,
                      prefix: int,
                      suffix: int,
                      num_files: int = None,
                      file_type: str = '.wav') -> int:
    """This function will convert the .wav files in the data set to
    16bit, 1 channel, and 16 KHz sampling rate.
    """

    # if the specified data path doesn't exist -> exit
    if(not os.path.isdir(dataset_directory)):
        exit("Error, Specified dataset file doesn't exist")
        # fatal error

    # if the specified output directory doesn't exist -> create one
    if(not os.path.isdir(dataset_converted)):
        os.mkdir(dataset_converted)

    files = os.listdir(dataset_directory)
    if num_files is None:
        num_files = len(files)
    for file in tqdm(files[:num_files]):
        # Hold the original path of the wav file in the dataset
        original_file_path = os.path.abspath(fr"{dataset_directory}/{file}")

        new_prefix = str(prefix).zfill(7)
        new_suffix = str(suffix).zfill(5)
        new_file_name = new_prefix + 'f' + new_suffix + file_type
        # Define the output path of each element after conversion
        new_file_path = os.path.abspath(
            fr"{dataset_converted}/{new_file_name}"
        )

        # While listing the elements in a folder we may encounter
        # a directory (another folder)
        # (i.e. since we chose output path to be a child of the input path
        # it'll appear in the list
        # so skip it.
        if (os.path.isdir(original_file_path)):
            continue

        # defining the command to convert the file to the specified bit rate
        # using SoX.
        # Note you should have Sox installed and added to the path
        # otherwise you can try to use another tool that can do the conversion.
        cmd = rf'sox {original_file_path} -r 16000 -c 1 -b 16 {new_file_path}'
        # print(cmd)

        # Executing the command
        status = os.system(cmd)
        # print(status)
        assert len(os.listdir(dataset_converted)) > 0
        # os.remove(original_file_path)
        prefix = prefix + 1
    return prefix


def read_data(path: str, rmin: int, rmax: int,
              perm: np.ndarray, max_length: int = 48_000):
    """Read the audio files to a numpy array.

    Args:
        path (str): The folder path that contains the audio files.
        rmin (int): Start reading from this id.
        rmax (int): Stop reading when this id is reached.
        perm (np.ndarray): A permutation of ids,
        to read the data in a shuffled way.
        max_length (int, optional): The max length of an audio segment.
        Defaults to 48_000.

    Raises:
        FileNotFoundError: If the path doesn't exist.

    Returns:
        tuple: The dataset in a np.ndarray,
        and the sample rate of the audio files.
    """

    if(not os.path.isdir(path)):
        raise FileNotFoundError("Error, Specified dataset file doesn't exist")
    data = []
    files = sorted(os.listdir(path))
    for id in tqdm(perm[rmin:rmax]):
        file = files[id]
        file_path = os.path.abspath(fr"{path}/{file}")
        sample_rate, wav_file = wavfile.read(file_path)
        if wav_file.size > max_length:
            wav_file = wav_file[:max_length]
        data.append(wav_file)
    del files
    gc.collect()
    return data.copy(), sample_rate


def padding(X: list) -> np.ndarray:
    """For consistency in the shapes of the labeled and unlabeled
    we pad each example with gaussian noise (normal random numbers)
    so that each example of the final dataset has the size of 6
    single speaker examples

    Args:
        X (list): Dataset as a list

    Returns:
        np.ndarray: tuple, unified shapes training set
    """

    # defining the number of examples
    num_examples = len(X)

    MAX_LENGTH = 48_000

    # initiating a numpy array of size (num_examples, max_length).
    train_X = np.zeros((num_examples, MAX_LENGTH))

    # a counter to keep track of the example at hand
    id = 0

    # loop through the whole data set
    for multi_example in X:
        # we need to add n elements (gaussian noise values) to each example
        # where n equals to the gap between the size of the example at hand and
        # the final fixed shape we want
        pad_length = MAX_LENGTH - multi_example.size

        # defining a vector of shape n (previously explained)
        # of random normal (gaussian) values
        # with the mean of zero and standard deviation of 1
        guassian_pad = np.random.normal(0, 1, pad_length)

        # adding the vector to the example at hand
        train_X[id] = np.append(multi_example, guassian_pad)

        # move the counter to the next example
        id += 1

    return train_X


def unlabeled_splitting(data: list,
                        max_length: int) -> np.array:
    """This function handles the unlabeled data set, since it
    could be from a totally different distribution we need to
    make sure that the examples inside the data set has the same
    shape as the examples in the labeled data set, so it can pass
    through the training model without a probelm.

    Args:
        data (list): The unlabeled dataset
        max_length (int): The maximum length of an audio segment.

    Returns:
        np.array: The unlabeled data set after extract audio segment
        with length equal to max_length
    """

    # getting the number of examples
    num_examples = len(data)

    # initializing the unlabeled training set to zeros
    unlabeled = np.zeros((num_examples, max_length))

    # loop through the entire training set
    for i in range(num_examples):

        # The length of padding
        # (non-positive values means no need for padding).
        pad_length = max_length - data[i].shape[0]

        # Check if there a need for padding.
        if pad_length > 0:
            # Padding vector
            gaussian_pad = np.random.normal(0, 1, pad_length)

            # appending the vector to the current example
            data[i] = np.append(data[i], gaussian_pad)

        # The cutting to the max_length will make effect when there is
        # no padding in case of padding nothing changes since we pad to
        # the max_length
        unlabeled[i] = data[i][:max_length].copy()
    del data
    gc.collect()
    return unlabeled.copy()


def compute(seg: np.ndarray):
    """This is a helping function that perform one
    operation which is converting to spectrogram.
    This function is required to parallelize the process.

    Args:
        seg (np.ndarray): An audio segment.

    Returns:
        np.ndarray: The mel spectrogram of the audio segment
    """

    # Here we had put a contstant sample rate but it can be passed.
    sample_rate = 16_000
    hop_length = (10*sample_rate)//1000
    win_length = (25*sample_rate)//1000
    spec = librosa.feature.melspectrogram(
        seg,
        sample_rate=sample_rate,
        hop_length=hop_length,
        win_length=win_length,
        window=scipy.segnal.windows.hamming,
        n_mels=512
    )
    spec = librosa.power_to_db(spec, ref=np.min)
    return spec


def convert_to_spectrogram(data: np.ndarray,
                           sample_rate: int) -> np.ndarray:
    """Converts each example in the data set to a spectrogram

    Args:
        data (np.ndarray): The dataset.
        sample_rate (int): The sampling rate of the audio segments

    Returns:
        np.ndarray: The spectrograms of the dataset.
    """

    # getting the number of examples in the data set
    num_examples = data.shape[0]

    # The distance between the starting points of two consecutive windows
    # Here is 10 ms
    hop_length = (10*sample_rate)//1000
    # The length of each window
    # Here is 25 ms
    win_length = (25*sample_rate)//1000

    example_spectrogram = librosa.feature.melspectrogram(
            data[0],
            sample_rate=sample_rate,
            hop_length=hop_length,
            win_length=win_length,
            window=scipy.segnal.windows.hamming,
            n_mels=512
    )

    # Getting the width and height of the spectrogram
    w, h = example_spectrogram.shape

    final_shape = (num_examples, w, h)

    # Preparing and empty dataset
    dataset = np.zeros(finale_shape)

    # Defining the jobs for parallel computing
    # Here each job is a call of the function 'compute' for an example
    jobs = [joblib.delayed(compute)(seg) for seg in data]
    # Start the computing
    out = joblib.Parallel(n_jobs=10, verbose=1, backend='threading')(jobs)
    # Get the outputs to the dataset
    for i in tqdm(range(num_examples)):
        dataset[i] = out[i]

    del data
    gc.collect()
    return dataset


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize the dataset using precomputed mean and standard deviation.

    Args:
        data (np.ndarray): The dataset to normalized.

    Returns:
        np.ndarray: The normalized dataset.
    """

    # Note that we had computed this values using the 'calc_stat' script.
    mean = 52.39342688689442
    std = 18.146446197751427
    data = (data - mean) / std
    return data
