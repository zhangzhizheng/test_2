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
a = 0.195+ 0.179+ 0.142+0.183+ 0.221+ 0.357+0.312+0.311
b = 0.228+0.178+0.166+0.24+0.245+0.359+0.330+0.329
c = 0.3+0.6+0.2+0.3+0.24+0.38+0.37+0.37
d = 0.529+1.196+0.381+0.545+0.373+0.597+0.571+0.575
e = 0.96+1.106+1.7+1.03+0.631+1.001+0.953+0.954
results = {
    'Batch size 32': [0.195/a, 0.179/a, 0.142/a, 0.183/a, 0.221/a, 0.357/a,0.312/a,0.311/a],
    'Batch size 64': [0.228/b,0.178/b,0.166/b,0.24/b,0.245/b,0.359/b,0.330/b,0.329/b],
    'Batch size 128': [0.3/c,0.6/c,0.2/c,0.3/c,0.24/c,0.38/c,0.37/c,0.37/c],
    'Batch size 256': [0.529/d,1.196/d,0.381/d,0.545/d,0.373/d,0.597/d,0.571/d,0.575/d],
    'Batch size 512': [0.96/e,1.106/e,1.7/e,1.03/e,0.631/e,1.001/e,0.953/e,0.954/e]
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

    fig, ax = plt.subplots(figsize=(9.2, 8))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        # print(widths)
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