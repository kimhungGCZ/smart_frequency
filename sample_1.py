import json
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ploting as ploting

def normalize_data(x):
    return (x - min(x)) / (max(x) - min(x))
def mean_absolute_error(a,b):
    result = []
    for index, value in enumerate(a):
        result.append(np.abs(a[index] - b[index]))
    return np.mean(result)

############################# READING JSON DARA ################################
# with open('./data/data.json') as f:
#     data = json.load(f)
#
# raw_value = [i['agg_avg_temperature_air_hdc1000']['value'] for i in data][1000:]

# ############################## READING DO DATA ######################################
# pd_data = pd.read_csv('./data/DO.csv')
# #pd_data = pd.read_csv('./data/turbinity.csv')
# raw_value = list(pd_data['Value'].values)

############################# READING JSON DARA ################################
with open('./data/data_202409.json') as f:
    data = json.load(f)

raw_value = [i[1] for i in data]

############################# PROCESSING ###############################################

start_frequency = 1;
windown_size = 50
output_array = raw_value[0:windown_size]
exam_point_index = 0
instructed_array = []
frequency_array = []
t = 0.1
n= 2
#n=8
while exam_point_index < len(raw_value):
    instructed_array.append(exam_point_index)
    if exam_point_index >= windown_size:
        new_value = np.interp(exam_point_index, range(len(raw_value)), raw_value)
        mean_windown_size = np.mean(output_array[-windown_size:])



        D_value = (np.abs(np.abs(new_value - output_array[-1])))/mean_windown_size

        frequency_change = n / (1 + np.power(np.e,-(D_value-t)))
        new_frequecy = start_frequency * frequency_change
        frequency_array.append(new_frequecy)
        exam_point_index = exam_point_index + 1/new_frequecy
        output_array.append(new_value)
        start_frequency = new_frequecy
    else:
        exam_point_index = exam_point_index + 1


# from scipy import signal
# f = signal.resample(output_array, len(raw_value))
#from sklearn.metrics import mean_absolute_error
f = interpolate.interp1d(instructed_array, output_array)
try:
    reconstructed_data = f(range(len(raw_value)))
except:
    instructed_array.append(len(raw_value))
    output_array.append(raw_value[-1])
    f = interpolate.interp1d(instructed_array, output_array)
    reconstructed_data = f(range(len(raw_value)))

output_size = len(output_array)
MSE = mean_absolute_error(normalize_data(np.array(raw_value)), normalize_data(reconstructed_data))

print("size: {}, MAE: {}".format(output_size,MSE))


############################################### DRAWING #########################################################################


plt.subplot(3, 1, 1)
plt.plot(raw_value, 'g.')
plt.title("Size " + str(len(output_array)) + ", MAE: " + str(MSE))

plt.subplot(3, 1, 2)
plt.plot(instructed_array,output_array, 'r.')

plt.subplot(3, 1, 3)
plt.plot(reconstructed_data, 'b.')


plt.show()

ploting.plot_data_all("Smart Frequency.html",
                          [
                              [list(range(0, len(raw_value))), raw_value],
                              [list(range(0, len(raw_value))), raw_value],
                              [list(instructed_array), output_array],
                              [list(range(0, len(raw_value))), list(reconstructed_data)]
                            ],
                            ['lines', 'markers', 'markers', 'lines'],
                            [None,'circle', 'circle', None, 'x'],
                            ['Origin data', "Origin point", "Sampled data", "Reconstructed Data"]
                          )