import json
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

with open('./data/data.json') as f:
    data = json.load(f)

raw_value = [i['agg_avg_temperature_air_hdc1000']['value'] for i in data]

plt.plot(raw_value)
plt.ylabel('some numbers')
plt.show()

start_frequency = 1;
windown_size = 50
output_array = raw_value[0:windown_size]
exam_point_index = 0
frequency_array = []
t = 0.01
while exam_point_index < len(raw_value):
    if exam_point_index >= windown_size:
        new_value = np.interp(exam_point_index, range(len(raw_value)), raw_value)
        mean_windown_size = np.mean(output_array[-windown_size:])
        D_value = np.abs((new_value - output_array[-1]))/ mean_windown_size
        frequency_change = 4 / (1 + np.power(np.e,-(D_value - t)))
        new_frequecy = start_frequency * frequency_change
        frequency_array.append(new_frequecy)
        exam_point_index = exam_point_index + new_frequecy
        output_array.append(new_value)
    else:
        exam_point_index = exam_point_index + 1

plt.plot(output_array)
plt.show()