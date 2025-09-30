import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import random
from time import perf_counter

# path_to_folder = 'D:\Py_Charm\CEITEC_research\data_folder\LIBS-2024'
# path_to_h5 = 'D:\Py_Charm\CEITEC_research\data_folder\spectra.h5'
# path_to_h5 = '/media/antonipa57/D/Py_Charm/CEITEC_research/data_folder/spectra.h5' #linux


class SpectraLoader():
    """
    Used for LIBS data acquired from different institutions in form of
    .txt files of spectra in folders of place of meteorite showers

    assumes meteorite spectra data structure:
        -path_to_folder:
            -place of meteorite materials 0
                -spectra1.txt
                    - wavelength;value
                      wavelength;value
                      ...
                -spectra2.txt

            -place of meteorite materials 1
                -spectra1.txt
                -spectra2.txt
            ...
            (only .txt files are considered)

    SpectraLoader is used to transform data from txt to hdf5 keeping the same general structure:
        -hdf5_file.h5:
            -group of meteorite materials 0:
                -dataset_from_spectra1: [[wavelengths, 'place', 'spectra'], [values, place, spectra type]]
                -dataset_from_spectra2
            ...

    additional features include:
        - hdf5 to pandas df
        - hdf5 to numpy array

    TODO:
    - in .load_to_hdf5() other file type processing if necessary
    - ask about labels
    """

    def __init__(self, path_to_folder : str = None):
        self.path_to_folder = path_to_folder
        self._data_pd = None
        
    def load_to_hdf5(self, path_out):
        if isinstance(self.path_to_folder, str) is False:
            raise ValueError(f'path_to_folder is {self.path_to_folder}, SpectraLoader requires path_to_folder to be initialized and used to .load_to_hdf5(path_to_folder)')

        place_files = os.listdir(self.path_to_folder) #place of meteorite materials
        hdf = h5py.File(path_out, 'w')

        for place in place_files:
            place_group = hdf.create_group(place)
            spectra_files = os.listdir(os.path.join(self.path_to_folder, place))
            spectra_files = [x for x in spectra_files if x.endswith('.txt')] #only txt files (for now at least)

            for i, spectra in enumerate(spectra_files):
                np_data = np.loadtxt(os.path.join(self.path_to_folder, place, spectra), delimiter=';').transpose()
                spectra_dataset = place_group.create_dataset(spectra, data = np_data)
                spectra_dataset.attrs['idx'] = i #add index opperation

        hdf.close()

    def _hdf5_to_list_of_np_arrays(self, h5_path :str):
        """
        :return: list(np.array([wavelengths, 'place', 'spectra'], [values, place, spectra type]))
        """
        if isinstance(h5_path, str) is False:
            raise ValueError(f'h5_path is {h5_path}, has to be string type')

        np_list = []
        with h5py.File(h5_path, 'r') as hdf:
            place_keys = list(hdf.keys())

            for place in place_keys:
                place_group = hdf.get(place)
                spectra_keys = list(place_group.keys())

                for spectra in spectra_keys:
                    sp = np.array(place_group.get(spectra))
                    label = np.array([['place', 'spectra'], [place, spectra[:-4]]]) #[:-4] -> without .txt
                    sp = np.append(sp, label, axis=1)
                    np_list.append(sp)

        return np_list

    def _hdf5_to_pandas(self, h5_path : str):
        """
        concatenates all np arrays from hdf5 to pandas df
        :return: pd.DataFrame([spectra values, place, spectra type])
        """
        if isinstance(h5_path, str) is False:
            raise ValueError(f'h5_path is {h5_path}, has to be string type')

        np_list = self._hdf5_to_list_of_np_arrays(h5_path)
        pd_list = []

        def map_func(i):
            columns = i[0]
            data = i[1]
            df = pd.DataFrame([data], columns=columns)
            return df

        # for i in np_list:
        #     data = i[1]
        #     columns = i[0]
        #     df = pd.DataFrame([data], columns=columns)
        #     pd_list.append(df)

        pd_list = list(map(map_func, np_list)) #pretty much the same speed but more memory efficient

        concatenated_df = pd.concat(pd_list, axis=0)

        columns = (float(i) for i in concatenated_df.columns if i != 'place' and i != 'spectra')
        sorted_columns = [str(i) for i in sorted(columns)] + ['place', 'spectra'] #to make sure columns as in ascending order
        sorted_df = concatenated_df[sorted_columns]
        sorted_df = sorted_df.reset_index(drop=True)
        return sorted_df

    def _hdf5_to_numpy(self, h5_path : str):
        if h5_path is None:
            raise ValueError(f'h5_path is {h5_path}, has to be string type')

        pd_df = self._hdf5_to_pandas(h5_path)
        return pd_df.to_numpy()

    def load_h5(self, path_to_h5: str):
        if isinstance(path_to_h5, str) is False:
            raise ValueError(f'path_to_h5 is {path_to_h5}, has to be string type')

        hdf_pd = self._hdf5_to_pandas(path_to_h5)
        self._data_pd = hdf_pd
        return self

    def get_data_as_numpy(self):
        return np.array(self._data_pd[self._data_pd.columns[:-2]]) #without place and spectra

    def get_labels_as_numpy(self):
        return np.array(self._data_pd['place']) # for now place is 'label'

    def get_wavelengths_as_numpy(self):
        return np.array(self._data_pd.columns[:-2])

#EXAMPLE USE
# start_time = perf_counter()
# spectra_loader = SpectraLoader(path_to_folder)
# spectra_loader.load_to_hdf5(path_to_h5)
# spectra_loader = SpectraLoader()
# pd_df = spectra_loader._hdf5_to_pandas(path_to_h5)
# a = spectra_loader.load_h5(path_to_h5)
# data = spectra_loader.get_data_as_numpy()
# labels = spectra_loader.get_labels_as_numpy()
# wavelengths = spectra_loader.get_wavelengths_as_numpy()
# print(data, labels, wavelengths)
# print(perf_counter() - start_time)

class SpectraViewer():
    """
    allows for a quick visual inspection of the data structure and spectra

    features:
        - show structure
        - show spectra as 2D plot

    TODO:
    - fix: .show_struct("data from Jakub B\\MP17.h5") OSError: Can't synchronously read data (wrong B-tree signature) (probably dependency error)
    """
    def __init__(self, path_to_h5: str = None):
        self.path_to_h5 = path_to_h5

    def show_struct(self, path: str):
        if path is None:
            raise ValueError("Invalid path to HDF5 file provided.")

        with h5py.File(path, 'r') as hdf:
            self._print_structure(hdf)

    def _print_structure(self, h5, indent=0):
        if isinstance(h5, h5py.Group):
            for key in h5.keys():
                item = h5[key]
                print('      ' * indent + key)
                self._print_structure(item, indent + 1)
        elif isinstance(h5, h5py.Dataset):
            print('      ' * indent + f"Dataset: shape={h5.shape}, dtype={h5.dtype}, name={h5.name}\n"+ '      ' * indent + f"data sample={h5[:3]}\n")

    def show_spectra(self, path : str, type : str = 'map'):

        if isinstance(path, str) is False:
            raise ValueError("Invalid path to HDF5 file provided.")

        with h5py.File(path, 'r') as hdf:
            measure_path = list(hdf.get('/measurements').keys())[0] #this gets the key that is named after current date and hour
            spectra_df = hdf.get('/measurements/'+ measure_path +'/libs/data') #spectra dataset
            calibration_df = hdf.get('/measurements/'+ measure_path +'/libs/calibration') #calibration (wavelength) dataset
            x = np.array(calibration_df)
            plt.figure(figsize=(10, 6))

            if type == 'map' or type == 'line' or type == 'multi point':
                for i in range(len(spectra_df)):
                    y = np.array(spectra_df[i])
                    plt.plot(x, y, label=f'Sample_{i + 1}')

            elif type == 'single point':
                y = np.array(spectra_df)
                plt.plot(x, y, label=f'Sample_0')

            file_name = '\\'.join([measure_path] + path.split("\\")[-2:])[:-3] #only show filename without .h5
            plt.title(f'Spectra Plot for: {file_name}')
            plt.xlabel('wavelength (nm)')
            plt.ylabel('Intensity [a.u.]')
            plt.grid(True)
            plt.show()


#EXAMPLE USE

# path = r"D:\Py_Charm\CEITEC_research\data_folder\first_spectra\DR-1.h5"
# path = r"D:\Py_Charm\CEITEC_research\data_folder\first_spectra\MAG-1_01.h5"
# path = r"D:\Py_Charm\CEITEC_research\data_folder\first_spectra\MP17.h5"

# spv = SpectraViewer()
# spv.show_struct(path)
# spv.show_spectra(path, type = 'map')

class LIBSSpectraLoader():
    """
    Used for measurements carried out at Brno University of Technology/CEITEC
    with file structure:

    -date of measurement:
        -minerals:
            -matrix_01.h5
            -matrix_02.h5
            -figure.jpg (will be ignored)
             ...

    (samples may be reffered to as minerals/meteorites when it comes to variable names)

    """
    def __init__(self, path_to_folder : str = None):
        self.path_to_folder = path_to_folder
        self._data_h5 = None
        self._data_h5_spectra = None
        self.random_idx = None

    def _access_spectra(self, hdf5):
        """
        :param: hdf5
        :return: given datasets still in hdf5
        """
        measurements = hdf5.get('/measurements')
        m_keys = list(measurements.keys()) #the date and hour of measurement that is always provided in .h5
        m_a = measurements[m_keys[0]]

        libs_spectra = m_a.get('libs/data')
        calibration_data = m_a.get('libs/calibration')
        libs_metadata = m_a.get('libs/metadata')
        global_metadata = m_a.get('global_metadata')

        return libs_spectra, calibration_data, libs_metadata, global_metadata

    def _access_hdf5(self, path):
        """
        assumes data structure:
        path/only_folders_named_as_dates_of_measurements/minerals/measurements.h5

        :return:  paths to list of spectra files and list of mineral types for each spectrum
        """
        dates = os.listdir(path)
        mineral_types = []
        spectra_path = []

        for d in dates:
            try:
                minerals = os.listdir(os.path.join(path, d))
            except:
                raise ValueError('Improper folder structure, expected: path/only_folders_with_dates_of_measurements/minerals/measurements')

            for m in minerals:
                try:
                    spectra = os.listdir(os.path.join(path, d, m))
                    spectra = [s for s in spectra if s.endswith('.h5')]
                except:
                    raise ValueError('Improper folder structure, expected: path/only_folders_with_dates_of_measurements/minerals/measurements')

                for s in spectra:
                    try:
                        spectra_path.append(os.path.join(path, d, m, s))
                        mineral_types.append(m.lower())
                    except:
                        raise ValueError('Improper folder structure, expected: path/only_folders_with_dates_of_measurements/minerals/measurements')

        unique_minerals = set(x.split('_')[0] for x in mineral_types)
        print(f'Number of spectra files found: {len(spectra_path)}'
              f'\nNumber of uniqe materials found: {len(unique_minerals)}'
              f'\nMaterials in total: {len(set(mineral_types))} \nSpectra files found in total: {len(mineral_types)}\n'
              f'Unique materials: {sorted(list(unique_minerals))}')
        return spectra_path, mineral_types

    def _h5_iterate(self, h5):
        """
        iterates over .h5 file structure (and)
        :return: all datasets in a list
        """
        results = []
        def iterate(group):
            for key in group:
                item = group[key]
                if isinstance(item, h5py.Group):
                    iterate(item)
                elif isinstance(item, h5py.Dataset):
                    results.append((item))

        if isinstance(h5, h5py.Group):
            iterate(h5)
        elif isinstance(h5, h5py.Dataset):
            results.append((h5))

        return results

    def measurements_to_h5(self, path, output_path, only_0_underscore = False):
        """
        only_0_underscore: if name of sample contains underscore it will split it and use only the first part
        e.g meteorite_the_big_one -> meteorite

        TODO:
        - find better name for only_0_underscore
        """
        paths, materials = self._access_hdf5(path)
        # lists of datasets:
        libs_spectra = []
        calibration_data = []
        libs_metadata = []
        global_metadata = []

        if only_0_underscore:
            materials = [m.split('_')[0] for m in materials]

        # append all datasets to lists
        for p in paths:
            hdf5 = h5py.File(p, 'r')
            x = (self._access_spectra(hdf5))
            libs_spectra.append(x[0])
            calibration_data.append(x[1])
            libs_metadata.append(x[2])
            global_metadata.append(x[3])

        #DO NOT hdf5.close()

        max_len = 0 # len of libs_metadata; libs_metadata.shape[0] = libs_spectra.shape[0] ; always: global.shape= (1,), calibration.shape = (14905,)

        # find max length for each list
        for j in libs_metadata:
            a = self._h5_iterate(j)[0].shape[0]
            if a > max_len:
                max_len = a

        # return list of lists of datasets for metadata e.g [[angle_data], [x_pos_data], ...]
        libs_meta = [self._h5_iterate(x) for x in libs_metadata]
        global_meta = [self._h5_iterate(x) for x in global_metadata]

        # stack into np array while maintainting the shape of the biggest array by padding with nan
        np_libs_meta = np.stack([np.concatenate((np.array(x), np.full(max_len - x.shape[0], np.nan))).tolist() for y in libs_meta for x in y])

        # returns list of metadata names e.g. [angle, x_pos, ...]
        np_libs_meta_names = [x.name.split("/")[-1] for y in libs_meta for x in y]

        pd_libs = pd.DataFrame(data = np_libs_meta.transpose().tolist(), columns=np_libs_meta_names) #dataframe with column names as meta_names and rows as metadata
        s_libs = set(pd_libs.columns) #unique meta_names
        prop_libs = [pd_libs[x] for x in s_libs] #list of dataframes for each meta_name
        np_libs = np.stack([np.array(x) for x in prop_libs]).transpose(0, 2, 1) #transpose from (meta_name, meta_data, sample) into (meta_name, sample, meta_data) 3D array

        #same for global metadata
        np_global_meta = np.stack([np.array(x).tolist() for y in global_meta for x in y])
        np_global_meta_names = [x.name.split("/")[-1] for y in global_meta for x in y]
        pd_global = pd.DataFrame(data = np_global_meta.transpose().tolist(), columns=np_global_meta_names)
        s_global = set(pd_global.columns)
        prop_global = [pd_global[x] for x in s_global]
        np_global = np.stack([np.array(x) for x in prop_global]).transpose(0, 2, 1)

        libs_spectra = [np.array(x) for x in libs_spectra]
        libs_spectra_pad= []
        for i in libs_spectra:
            a = np.concatenate((i, np.full((max_len - i.shape[0], i.shape[1]), np.nan)), axis=0)
            libs_spectra_pad.append(a)

        with h5py.File(output_path, 'w') as hdf:
            hdf.create_dataset('libs/data', data=libs_spectra_pad)
            hdf.create_dataset('libs/calibration', data=np.array(calibration_data))
            hdf.create_dataset('libs/minerals', data=materials)

            #decode data from libs metadata and global metadata
            libs_metadata_group = hdf.create_group('libs/metadata')
            global_metadata_group = hdf.create_group('global_metadata')

            for i, s in enumerate(s_libs):
                libs_metadata_group.create_dataset(s, data=np_libs[i])

            for i, s in enumerate(s_global):
                global_metadata_group.create_dataset(s, data=np_global[i])

    def load_h5(self, path_to_h5: str):
        hdf = h5py.File(path_to_h5, 'r')
        # a = hdf['libs/data'][()]
        self._data_h5 = hdf
        return self

    def get_data_as_numpy(self, divide_spectra = None):
        if divide_spectra is not None:
            return np.array(self._data_h5['libs/data'][:, ::int(divide_spectra), :][()])
        else:
            return np.array(self._data_h5['libs/data'][()]) # [()] required for scalar datasets


    def get_labels_as_numpy(self):
            return np.array(self._data_h5['libs/minerals'])

    def get_wavelengths_as_numpy(self):
        return np.array(self._data_h5['libs/calibration'][0]) # assuming the wavelengths are the same


class LIBSDataset():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.extracted_data = None
        self.extracted_labels = None

    def extract(self):
        #if data is 3d not 2d
        is_3d = len(self.data.shape) == 3
        if is_3d:
            sample, measurements, spectra = self.data.shape
            data_flatten = np.reshape(self.data, (-1, spectra))
            labels_flatten = np.repeat(self.labels, measurements).reshape(-1, 1)
        else:
            data_flatten = self.data
            labels_flatten = self.labels.reshape(-1, 1)

        del self.data, self.labels #for less memory usage
        data_df = pd.concat([pd.DataFrame(data_flatten), pd.DataFrame(labels_flatten)], axis=1)
        del data_flatten, labels_flatten #del for storage usage

        if is_3d: #3d data has missing values to keep dimentions uniform
            data_df = data_df[data_df.iloc[:, 0].notna()] #delete nan rows that are created during data handling
        data_df = data_df.reset_index(drop=True)

        self.extracted_data = data_df.iloc[:, :-1].to_numpy()
        self.extracted_labels = data_df[data_df.iloc[:, -1].apply(lambda x: isinstance(x, bytes))].iloc[:, -1].to_numpy().reshape(-1, 1) #only bite values, so no 0.0

    def __getitem__(self, idx):
        if self.extracted_data is None:
            self.extract()
        return self.extracted_data[idx], self.extracted_labels[idx]

    def __len__(self):
        if self.extracted_data is None:
            self.extract()
        return len(self.extracted_data)

        # return data_flatten, labels_flatten


# path = r'D:\Py_Charm\CEITEC_research\data_folder\test folder\test spectra'
# lsl = LIBSSpectraLoader()
# lsl._access_hdf5(path)
# lsl.measurements_to_h5(path, 'D:\Py_Charm\CEITEC_research\data_folder\\test folder\\test_spectra.h5')
# print('----------------------------------')
# spv.show_struct('D:\Py_Charm\CEITEC_research\data_folder\\test folder\\test_spectra.h5')
#git test

# lbs = LIBSDataset(data, labels)
# lbs.extract()
# d = lbs.extracted_data
# l = lbs.extracted_labels
# print(d, l)