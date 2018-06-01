import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly

def plot_data_all(charName,data, mode,symbol,name):
    fig = []
    for index,in_data in enumerate(data):
        trace1 = go.Scatter(x=in_data[0], y=in_data[1], name = name[index], mode  = 'lines' if mode[index] == None else mode[index], marker = dict(
        size = 7, symbol = symbol[index] if symbol[index] != None else "circle") if mode[index] == 'markers' else dict())
        fig.append(trace1)

    layout = dict(title=charName
                  )

    # Working online
    #py.plot(fig, filename=charName)
    #return py.plot(dict(data=fig,layout=layout), filename=charName)
    # Working Offline
    plotly.offline.plot(dict(data=fig,layout=layout), filename=charName, auto_open=True)

def plot_resutl(ploting_result):
    #result = np.array(result)
    marker_list = ['o','v','s','*','^',]
    marker_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    fig = plt.figure()

    fig, (ax1, ax) = plt.subplots(1, 2, sharex=True, figsize=(15, 5))
    marker_index = 0
    for ws in ploting_result:
        result = np.array(ploting_result[ws])
        X = result[:, 0]
        Y_1 = result[:, 2]
        ax1.plot(X, Y_1, marker=marker_list[marker_index], markerfacecolor=marker_color[marker_index], markersize=8, color='black', linewidth=1, label=str(ws)+ " WS")
        marker_index = marker_index + 1
    ax1.set_xticks(X)
    marker_index = 0
    for ws in ploting_result:
        result = np.array(ploting_result[ws])
        X = result[:, 0]
        Y_2 = result[:, 3]
        ax.plot(X, Y_2, marker=marker_list[marker_index], markerfacecolor=marker_color[marker_index], markersize=8, color='black', linewidth=1, label=str(ws)+ " WS")
        marker_index = marker_index + 1

    ax.set_ylabel('MSE')
    ax.set_xlabel('% N')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xlabel('% N')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.2, 0.92), ncol=5)
    # plt.savefig('new_changeALpercentage.pdf', bbox_inches='tight')
    plt.show()
