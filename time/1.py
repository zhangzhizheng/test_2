# # a = np.array([100000000,0])
# # a = set(a)
# # print(a)
# b = [20,120,220,320,420,520,620,720,820,920,1030,1140,1240,1340]
# # print(b)
# # b = set(b)
# # print(b)
# # #b = a[:,[7,0,2]]   # 从索引 2 开始到索引 7 停止，间隔为 2
# # c = list(b - a)
# # print(c)

# dict_users = {i: np.array([]) for i in range(5)}
# dict_users[0] = np.concatenate((dict_users[0], b[1*2:(1+1)*2]), axis=0)
# dict_users[0] = np.concatenate((dict_users[0], b[1*2:(1+1)*2]), axis=0)
# dict_users[1] = np.concatenate((dict_users[1], b[1*2:(1+1)*2]), axis=0)
# print(dict_users)

#PLOTTING (optional)
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

    def label_bars(heights, rects):
        """Attach a text label on top of each bar."""
        for height, rect in zip(heights, rects):
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # 4 points vertical offset.
                        textcoords='offset points',
                        ha='center', va='bottom')

    values = np.asarray(values)
    x = np.arange(values.shape[1])
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    spacing = 0.3  # spacing between hat groups
    width = (1 - spacing) / values.shape[0]
    heights0 = values[0]
    for i, (heights, group_label) in enumerate(zip(values, group_labels)):
        style = {'fill': False} if i == 0 else {'edgecolor': 'black'}
        rects = ax.bar(x - spacing/2 + i * width, heights - heights0,
                       width, bottom=heights0, label=group_label, **style)
        label_bars(heights, rects)

# Saving the objects train_loss and train_accuracy:
file_name_1 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_107.87097311019897_batch_32.pkl'
file_name_2 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_83.20832443237305_batch_64.pkl'
file_name_3 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_71.0647406578064_batch_128.pkl'
file_name_4 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_65.34578490257263_batch_256.pkl'
file_name_5 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_62.09153151512146_batch_512.pkl'


with open(file_name_1, 'rb') as f1:
    [time_1] = pickle.load(f1)

with open(file_name_2, 'rb') as f2:
    [time_2] = pickle.load(f2)
with open(file_name_3, 'rb') as f3:
    [time_3] = pickle.load(f3)
with open(file_name_4, 'rb') as f4:
    [time_4] = pickle.load(f4)
with open(file_name_5, 'rb') as f5:
    [time_5] = pickle.load(f5)

# initialise labels and a numpy array make sure you have
# N labels of N number of values in the array
xlabels = [32, 64, 128, 256, 512]
# min_b = np.array([min(time_1), min(time_2), min(time_3), min(time_4), min(time_5)])
# max_b = np.array([max(time_1), max(time_2), max(time_3), max(time_4), max(time_5)])
playerA = np.array([min(time_1), min(time_2), min(time_3), min(time_4), min(time_5)])
playerB = np.array([max(time_1), max(time_2), max(time_3), max(time_4), max(time_5)])

fig, ax = plt.subplots()
hat_graph(ax, xlabels, [playerA, playerB], ['min', 'max'])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('batch')
ax.set_ylabel('s')
ax.set_ylim(12, 25)
ax.set_title('time of one epoch')
ax.legend()

fig.tight_layout()
plt.show()


# Saving the objects train_loss and train_accuracy:
file_name_1 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_107.87097311019897_batch_32.pkl'
file_name_2 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_83.20832443237305_batch_64.pkl'
file_name_3 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_71.0647406578064_batch_128.pkl'
file_name_4 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_65.34578490257263_batch_256.pkl'
file_name_5 = 'H:\\paper\\P-idea-1\\test_2\\time\\time_62.09153151512146_batch_512.pkl'


with open(file_name_1, 'rb') as f1:
    [time_1] = pickle.load(f1)

with open(file_name_2, 'rb') as f2:
    [time_2] = pickle.load(f2)
with open(file_name_3, 'rb') as f3:
    [time_3] = pickle.load(f3)
with open(file_name_4, 'rb') as f4:
    [time_4] = pickle.load(f4)
with open(file_name_5, 'rb') as f5:
    [time_5] = pickle.load(f5)
# with open(file_name_2, 'rb') as f2:
#     [list_acc_1, list_loss_1, list_acc_2, list_loss_2] = pickle.load(f2)
# with open(file_name_3, 'rb') as f3:
#     # [list_acc, list_loss] = pickle.load(f1)
#     [list_acc_3_2, list_loss_3_2, list_acc_4_2, list_loss_4_2] = pickle.load(f3)
#     # print(list_acc_2_2, list_loss_2_2)
# with open(file_name_4, 'rb') as f4:
#     [list_acc_3, list_loss_3, list_acc_4, list_loss_4] = pickle.load(f4)


def hat_graph(ax, xlabels, values, group_labels):
    """
    Create a hat graph.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes to plot into.
    xlabels : list of str
        The category names to be displayed on the x-axis.
    values : (M, N) array-like
        The data values.
        Rows are the groups (len(group_labels) == M).
        Columns are the categories (len(xlabels) == N).
    group_labels : list of str
        The group labels displayed in the legend.
    """

xlabels = [32, 64, 128, 256, 512]
# min_b = np.array([min(time_1), min(time_2), min(time_3), min(time_4), min(time_5)])
# max_b = np.array([max(time_1), max(time_2), max(time_3), max(time_4), max(time_5)])
min_b = np.array([5, 15, 22, 20, 25])
max_b = np.array([25, 32, 34, 30, 27])

print(min_b, max_b)
#plt.figure()

# plt.title('Training Loss vs Communication rounds')
# plt.plot(range(len(time_1)), time_1, "-", label = "batch_32")
# plt.plot(range(len(time_2)), time_2, "-", label = "batch_64")
# plt.plot(range(len(time_3)), time_3, "-", label = "batch_128")
# plt.plot(range(len(time_4)), time_4, "-", label = "batch_256")
# plt.plot(range(len(time_5)), time_5, "-", label = "batch_512")
# plt.legend()
# plt.ylabel('Training loss')
# plt.xlabel('Communication Rounds')


# plt.figure()

fig, ax = plt.subplots()
hat_graph(ax, xlabels, [min_b, max_b],['min','max'])
ax.set_xlabel('batch')
ax.set_ylabel('s')
ax.set_ylim(0, 40)
ax.set_title('time per epoch')

ax.legend()
fig.tight_layout()

fig.savefig('H:\\paper\\P-idea-1\\test_2\\time\\time.png')
        