import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def xy_plot(x_values, y_values, labels=None, size=(12, 9), xy_labels=('x', 'y'), title=None, min_max_xlim=None, markers=None, show = True, save_to=False):
    """
    :param x_values: array of x values, has to be list of lists e.g. [[values]]
    :param y_values: array of y values, has to be list of lists e.g. [[values]]
    :param labels: array of labels, just as list
    :param size: size of plot
    :param xy_labels: labels of x, y axis
    :param title: title of plot
    :param min_max_xlim: (min_value_of_x_axis, max_value_of_x_axis)
    :param markers: ([x1, x2, ...], [y1, y2, ...])
    :return: None, (plot)
    """
    plt.figure(figsize=size)
    if labels is None and markers is None:
        for x, y in zip(x_values, y_values):
            plt.plot(x, y)

    elif markers is not None:
        for x, y, m_x, m_y in zip(x_values, y_values, markers[0], markers[1]):
            plt.plot(x, y)
            plt.plot(m_x, m_y, 'ro')

    else:
        for x, y, l in zip(x_values, y_values, labels):
            plt.plot(x, y, label=l)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

    plt.xlabel(xy_labels[0])
    plt.ylabel(xy_labels[1])
    if min_max_xlim is not None:
        plt.xlim(xmin=min_max_xlim[0], xmax=min_max_xlim[1])
    if title is not None:
        plt.title(title)
    plt.grid(True)
    if save_to:
        plt.savefig(save_to)
    if show:
        plt.show()


def scatter_2d_plot(data, labels, size=(12, 9), xy_labels=('d1', 'd2'), title=None):
    pd_df = pd.DataFrame(data, columns=['d1', 'd2'])
    pd_df['label'] = labels

    cmap = plt.cm.tab20
    colors = [cmap(i) for i in range(len(pd_df['label'].unique()))]
    color_dict = {label: color for label, color in zip(pd_df['label'].unique(), colors)}

    fig, ax = plt.subplots(figsize=size)
    scatter = ax.scatter(pd_df['d1'], pd_df['d2'], c=[color_dict[label] for label in pd_df['label']])

    try:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label, 'utf-8'), markerfacecolor=cmap(i), markersize=10) for i, label in enumerate(color_dict.keys())]
    except:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=cmap(i), markersize=10) for i, label in enumerate(color_dict.keys())]
    ax.legend(handles=legend_handles, title='Sample Labels', bbox_to_anchor=(1.07, 1), loc='upper left')

    plt.subplots_adjust(right=0.75)
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.5)
    plt.show()

def scatter_3d_plot(data, labels, size=(12, 9), xyz_labels=('d1', 'd2', 'd3'), title=None):
    pd_df = pd.DataFrame(data, columns=['d1', 'd2', 'd3'])
    pd_df['label'] = labels
    cmap = plt.cm.tab10

    colors = [cmap(i) for i in range(len(pd_df['label'].unique()))]
    color_dict = {label: color for label, color in zip(pd_df['label'].unique(), colors)}
    color = pd_df['label'].apply(lambda x: color_dict[x])

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pd_df['d1'], pd_df['d2'], pd_df['d3'], c=color)

    # legend with lables
    try:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=str(label, 'utf-8'), markerfacecolor=cmap(i), markersize=10) for i, label in enumerate(color_dict.keys())]
    except:
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=cmap(i), markersize=10) for i, label in enumerate(color_dict.keys())]
    ax.legend(handles=legend_handles, title='Sample Labels', bbox_to_anchor=(1.07, 1), loc='upper left')

    ax.set_xlabel(xyz_labels[0])
    ax.set_ylabel(xyz_labels[1])
    ax.set_zlabel(xyz_labels[2])
    if title is not None:
        ax.set_title('t-SNE embedding visualization')

    plt.subplots_adjust(right=0.75)
    plt.show()

def list_to_3d(lst, ydim=None, title=None, class_dict=None, cmap_name='tab20'):
    """
    Turns a list into a spectral 'image; based on the y size of it
    :param:
    - lst: list of values
    - ydim: y dimension of the image
    - class_dict = {value0 : 'class0', value1 : 'class1', ...}

    :return: None (create plot)
    """
    # Divide the list into sublists of length ydim
    divided_list = [lst[i:i + ydim] for i in range(0, len(lst), ydim)]
    # Flip every other sublist for the "snake" arrangement
    flipped_list = [row[::-1] if i % 2 != 0 else row for i, row in enumerate(divided_list)]
    # Convert the list of lists to a numpy array and transpose
    arr = np.array(flipped_list).transpose()

    fig, ax = plt.subplots()
    cmap = plt.get_cmap(cmap_name)

    # Generate a list of unique colors for each unique element in lst
    unique_elements = sorted(set(lst))
    cmap_list = [cmap(i / len(unique_elements)) for i in range(len(unique_elements))]

    if class_dict is not None:
        # Ensure that the class_dict and unique_elements lengths match
        class_labels = [class_dict[element] for element in unique_elements if element in class_dict]
        if len(class_labels) != len(unique_elements):
            raise ValueError("class_dict must contain labels for all unique elements in lst")
        color_dict = {label: cmap_list[i] for i, label in enumerate(class_labels)}
    else:
        color_dict = {element: color for element, color in zip(unique_elements, cmap_list)}

    # Create legend handles
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10) for
                      label, color in color_dict.items()]

    ax.legend(handles=legend_handles, title='Clusters', bbox_to_anchor=(1.07, 1), loc='upper left')

    # Create a color-mapped version of the array using color_dict
    color_mapped_arr = np.empty(arr.shape + (3,))
    for value in unique_elements:
        if class_dict is not None:
            label = class_dict[value]
        else:
            label = value
        color = color_dict[label]
        color_mapped_arr[arr == value] = color[:3]  # only use RGB, discard alpha

    ax.imshow(color_mapped_arr)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.subplots_adjust(right=0.75)
    plt.show()

def confusion_matrix_plot(pred_idx, actual_labels, class_names, title='Confusion Matrix', cmap='Blues'):
    """
    Plot a confusion matrix.

    Parameters:
    - pred_idx: list of int, predicted indices
    - actual_labels: list of int, actual labels
    - class_names: list of strings, class names
    - title: string, title of the plot
    - cmap: string, colormap

    :return: None, (plot)
    """
    # Calculate the confusion matrix
    conf_mat = confusion_matrix(actual_labels, pred_idx)

    # Normalize the confusion matrix
    with np.errstate(all='ignore'):
        conf_mat_norm = np.divide(conf_mat, conf_mat.sum(axis=1)[:, np.newaxis], where=conf_mat.sum(axis=1)[:, np.newaxis]!=0)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Create a heatmap
    im = ax.imshow(conf_mat_norm, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    cbar = fig.colorbar(im)
    cbar.set_label('Normalized frequency')

    ax.set_xticks(np.arange(conf_mat.shape[1]))
    ax.set_yticks(np.arange(conf_mat.shape[0]))

    class_names = [a for i, a in enumerate(class_names) if i in actual_labels  or i in pred_idx]
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the xtick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            text = ax.text(j, i, f"{conf_mat_norm[i, j]:.2f}" if not np.isnan(conf_mat_norm[i, j]) else "",
                           ha="center", va="center", color="w")

    plt.show()