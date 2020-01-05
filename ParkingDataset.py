from os.path import join

import numpy as np
import jsonpickle


class ParkingDataset():

    def __init__(self):
        """"Initialize the default path"""
        self.super_path = "insert directory here"
        self.path = join(self.super_path, "parking_lot_ds")

    def get_all_data(self):
        """Returns all the loaded data as -- x , y"""

        saved_file = open(self.path, mode="r")
        json_data = saved_file.read()
        json_data = jsonpickle.decode(json_data)
        saved_file.close()

        # variables from file
        blocks_ahead = json_data['blocks_ahead']

        # warning: condensing time image
        time_image = json_data['time_image']
        time_image_condensed = np.zeros(shape=(3, len(time_image[0])))

        index_te = 0
        index_tp = 1
        index_ex = 2
        while index_te < len(time_image):
            te = time_image[index_te]
            tp = time_image[index_tp]
            ex = time_image[index_ex]

            time_image_condensed[0] = np.sum([time_image_condensed[0], te], axis=0)
            time_image_condensed[1] = np.sum([time_image_condensed[1], tp], axis=0)
            time_image_condensed[2] = np.sum([time_image_condensed[2], ex], axis=0)

            index_te += 3
            index_tp += 3
            index_ex += 3

        # joining data to inputs
        index_input = 0
        total_size_inputs = len(json_data['inputs'])
        while index_input < total_size_inputs:
            index_from_time_image = json_data['inputs_indexes'][index_input]

            values_te = time_image_condensed[0][index_from_time_image:index_from_time_image + blocks_ahead]
            values_tp = time_image_condensed[1][index_from_time_image:index_from_time_image + blocks_ahead]
            values_ex = time_image_condensed[2][index_from_time_image:index_from_time_image + blocks_ahead]

            # this represents the time image information
            json_data['inputs'][index_input] = np.insert(
                json_data['inputs'][index_input],
                len(json_data['inputs'][index_input]),
                values_te)
            json_data['inputs'][index_input] = np.insert(
                json_data['inputs'][index_input],
                len(json_data['inputs'][index_input]),
                values_tp)
            json_data['inputs'][index_input] = np.insert(
                json_data['inputs'][index_input],
                len(json_data['inputs'][index_input]),
                values_ex)

            index_input += 1

        x_values = np.array(json_data['inputs'])
        y_values = np.array(json_data['labels'])

        return x_values, y_values
