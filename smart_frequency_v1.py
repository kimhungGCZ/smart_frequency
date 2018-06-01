import json
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ploting as ploting

def mean_absolute_error(a,b):
    result = []
    for index, value in enumerate(a):
        result.append(np.abs(a[index] - b[index]))
    return np.mean(result)

def effective_factor(a,b):
    result = (2*a*b)/(a+b)
    return result

def normalize_data(x):
    return (x - min(x)) / (max(x) - min(x))

def change_after_k_seconds_with_abs(data, k=1):
    data1 = data[0:len(data) -k]
    data2 = data[k:]
    return list(map(lambda x: abs(x[1] - x[0]), zip(data1, data2)))

############################# READING JSON DARA ################################
# with open('./data/data.json') as f:
#     data = json.load(f)
#
# raw_value = [i['agg_avg_temperature_air_hdc1000']['value'] for i in data][1000:]

############################## READING DO DATA ######################################
pd_data = pd.read_csv('./data/DO.csv')
#pd_data = pd.read_csv('./data/turbinity.csv')
raw_value = list(pd_data['Value'].values)

############################# READING JSON DARA ################################
# with open('./data/data_202409.json') as f:
#     data = json.load(f)
#
# raw_value = [i[1] for i in data]


############################# PROCESSING ###############################################

def processing_frequency(windown_size = 20, n = 4):
    start_frequency = 1;
    output_array = raw_value[0:windown_size]
    exam_point_index = 0

    frequency_array = []
    instructed_array = []
    diff_array = []
    while exam_point_index < len(raw_value):
        instructed_array.append(exam_point_index)
        if exam_point_index >= windown_size:
            new_value = np.interp(exam_point_index, range(len(raw_value)), raw_value)
            mean_difff = np.mean(
                change_after_k_seconds_with_abs(output_array[-windown_size:]))

            D_value = ( np.abs((new_value - output_array[-1])) -  (n + 1)/2*mean_difff )/ mean_difff


            frequency_change = n + (1 - n) / (1 + (np.power(np.e, (-n*D_value))))
            #frequency_change = n + (1 - n) / (1 + (np.power(np.e, (-n*D_value))))

            new_frequecy = 0 + frequency_change
            frequency_array.append(new_frequecy)
            exam_point_index = exam_point_index + new_frequecy
            output_array.append(new_value)
            diff_array.append(( np.abs((new_value - output_array[-1])) - mean_difff ))
            start_frequency = new_frequecy
        else:
            exam_point_index = exam_point_index + 1
            frequency_array.append(start_frequency)

    # from scipy import signal
    # f = signal.resample(output_array, len(raw_value))
    # from sklearn.metrics import mean_absolute_error
    f = interpolate.interp1d(instructed_array, output_array)
    try:
        reconstructed_data = f(range(len(raw_value)))
    except:
        instructed_array.append(len(raw_value))
        output_array.append(raw_value[-1])
        f = interpolate.interp1d(instructed_array, output_array)
        reconstructed_data = f(range(len(raw_value)))
        frequency_array.append(start_frequency)

    #print("size: {}, MAE: {}".format(len(output_array), mean_absolute_error(raw_value, reconstructed_data)))
    power_saving = 1 - len(output_array)/len(raw_value)
    MSE = mean_absolute_error(normalize_data(np.array(raw_value)), normalize_data(reconstructed_data))

    ############################################### DRAWING #########################################################################

    # plt.subplot(3, 1, 1)
    # plt.plot(raw_value, 'g.')
    # plt.title("Power Saving " + str(len(output_array)) + ", MAE: " + str(MSE))
    print(len(output_array))
    print(str(n) +  ": Radio Size " + str(power_saving) + ", MAE: " + str(MSE) +", EF: " + str(effective_factor(power_saving,MSE)))
    # #
    # plt.subplot(3, 1, 2)
    # plt.plot(instructed_array, output_array, 'r.')
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(instructed_array, frequency_array, 'b.')
    #
    # plt.show()
    #
    return [n, windown_size, len(output_array), 100*MSE]

if __name__ == "__main__":
    #n_array = [2,4,6,8,10]
    #windown_size_array = [5, 10, 20, 50, 100]
    windown_size_array = [50]
    n_array = np.arange(2,20,2)
    n_array = np.concatenate(([1], n_array))
    ploting_result = {}
    for ws in windown_size_array:
        result = []
        for n in n_array:
            result.append(processing_frequency(windown_size = ws, n=n))
        ploting_result[ws] = result

    ploting.plot_resutl(ploting_result)



    #     cmfunc.plot_data_all(file_path_chart,
    #                          [[list(range(0, len(raw_dta.value))), raw_dta.value],
    #                           [list([index for index, value in enumerate(raw_dta.anomaly_point.values) if value == 1]),
    #                            raw_dta.value[list(
    #                                [index for index, value in enumerate(raw_dta.anomaly_point.values) if value == 1])]],
    #                           [list([index for index, value in enumerate(raw_dta.change_point.values) if value == 1]),
    #                            raw_dta.value[list(
    #                                [index for index, value in enumerate(raw_dta.change_point.values) if value == 1])]],
    #                           [list(
    #                               [index for index, value in enumerate(raw_dta.anomaly_pattern.values) if value == 1]),
    #                            raw_dta.value[list(
    #                                [index for index, value in enumerate(raw_dta.anomaly_pattern.values) if
    #                                 value == 1])]]],
    #                          ['lines', 'markers', 'markers', 'markers'],
    #                          [None, 'x', 'circle', 'x'],
    #                          ['Raw data', "Detected Anomaly Point", "Detected Change Point",
    #                           "Detected Anomaly Patterm"]
    #                          )

    ############ COMPARASION ####################

    array_result = np.array(ploting_result[50])

    f = interpolate.interp1d(array_result[:,2], array_result[:,3])
    #new_value = f([1064,548,421,297,146])
    new_value = f([1064,548,421,297])
    DDASA = [1.62, 5.52, 8.43, 9.99]
    #DDASA = [1.62, 5.52, 8.43, 9.99, 11.7]
    print(new_value - DDASA)

