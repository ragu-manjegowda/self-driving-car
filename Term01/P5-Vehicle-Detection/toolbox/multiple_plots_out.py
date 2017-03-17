import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import pandas as pd


def plot_cluster(img_list=[], img_text=False, new_img_shape=None, cluster_shape=None, fontsize=18):

    if cluster_shape == None:
        val_col = int(np.ceil(np.sqrt(len(img_list))))
        val_row = int(np.ceil(len(img_list) / val_col))
        cluster_shape = (val_row, val_col)

    if new_img_shape == None:
        if isinstance(img_list[0], np.ndarray):
            if len(img_list[0].shape) >= 2:
                new_img_shape = tuple([img_list[0].shape[i] / 100 for i in range(len(img_list[0].shape))])
            else:
                new_img_shape = (7.2, 12.8, 0.03)
        else:
            new_img_shape = (7.2, 12.8, 0.03)

        new_img_shape = tuple(reversed(
            [int(new_img_shape[i] * cluster_shape[i] / max(cluster_shape)) for i in range(len(cluster_shape))]))
    # Todo: Calculate New Image Shape if One Value is given

    size_of_cluster = (cluster_shape[0] * cluster_shape[1])
    if size_of_cluster < len(img_list):
        logging.warning('Cluster to small for all Images')

    # Plot the result
    f, plot_cluster = plt.subplots(cluster_shape[0], cluster_shape[1], figsize=new_img_shape, squeeze=False)
    #f.tight_layout()

    plot_index = 0
    for row in range(cluster_shape[0]):
        for col in range(cluster_shape[1]):

            if isinstance(img_list[plot_index], np.ndarray):
                plot_cluster[row][col].imshow(img_list[plot_index], cmap='gray')
            elif isinstance(img_list[plot_index], pd.Series):
                img_list[plot_index].plot(ax=plot_cluster[row][col], stacked=False)
            elif isinstance(img_list[plot_index], pd.DataFrame):
                img_list[plot_index].plot(ax=plot_cluster[row][col], stacked=True)

            if img_text:
                if plot_index < len(img_text):
                    if img_text[plot_index] != None:
                        plot_cluster[row][col].set_title(img_text[plot_index], fontsize=fontsize)
            plot_index += 1
            if (plot_index >= len(img_list)):
                break
        if (plot_index >= len(img_list)):
            break

    plt.subplots_adjust(left=.05, right=.95, top=0.9, bottom=0.1)

    plt.show()


if __name__ == "__main__":
    # Read in the image

    img1 = cv2.imread('../test_images/straight_lines1.jpg')
    img2 = cv2.imread('../test_images/straight_lines2.jpg')

    img3 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    new_image = plot_cluster([img1, img2, img3, img2, img3], img_text=['Image 1', 'Image 2', 'Image 3', None, 'Image 3'], new_img_shape=None, cluster_shape=None)


    # pd.DataFrame(histogram).plot(ax=ax2)
    # lanes