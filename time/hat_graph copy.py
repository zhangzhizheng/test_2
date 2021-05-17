"""
=========
Hat graph
=========
This example shows how to create a `hat graph`_ and how to annotate it with
labels.

.. _hat graph: https://doi.org/10.1186/s41235-019-0182-3
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt



category_names = ['convolution_1', 'convolution_2', 'convolution_3', 'convolution_4', 'convolution_5', 'convolution_6', 'convolution_7', 'convolution_8']
# c = 1.564+0.124+0.137+0.105+0.305+0.126+0.473+0.068+0.089+0.074+0.136+0.045+0.078+0.045+0.118+0.046+0.216+0.064+0.214+0.065+0.215+0.066+0.214+0.047+1.337+2.459
# results = {
#     'MobileNet': [1.564/c,0.124/c+0.137/c+0.105/c+0.305/c+0.126/c,0.473/c,0.068/c+0.089/c+0.074/c+0.136/c+0.045/c+0.078/c+0.045/c+0.118/c+0.046/c+0.216/c+0.064/c+0.214/c+0.065/c,0.215/c,0.066/c+0.214/c+0.047/c,1.337/c,2.459/c]
# }
c = 93+66.509+21.456+3.0+3.0+3.0+3.0+3.0
results = {
    'ResNet18': [93/c,66.509/c,21.456/c,3.0/c,3.0/c,3.0/c,3.0/c,3.0/c]
}
# print(np.array(list(results.values())))
# data_cum = np.array(list(results.values())).cumsum()
# print(data_cum)
def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 27))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()