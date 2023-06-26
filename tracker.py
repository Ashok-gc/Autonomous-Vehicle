# Define a class for tracking lane line centroids
import numpy as np


class Tracker:
    def __init__(self, Mywindow_width, Mywindow_height, Mymargin, My_ym, My_xm, Mysmooth_factor):
        # Set parameters
        self.window_width = Mywindow_width
        self.window_height = Mywindow_height
        self.margin = Mymargin
        self.ym_per_pix = My_ym
        self.xm_per_pix = My_xm
        self.smooth_factor = Mysmooth_factor

        # Store previous centroid values for smoothing
        self.recent_centers = []

    def find_window_centroids(self, warped):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin

        window_centroids = []  # Store (left, right) window centroid positions per level
        window = np.ones(window_width)  # Create window template for convolution

        # Find the starting points for the left and right lanes
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

        # Add the first layer centroid positions to the list
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, int(warped.shape[0] / window_height)):
            # Mask the image within the window area
            image_layer = np.sum(
                warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
                :], axis=0)
            conv_signal = np.convolve(window, image_layer)

            # Offset the window centroid by the previous layer's position
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

            # Add the current layer centroid positions to the list
            window_centroids.append((l_center, r_center))

        # Update recent_centers with the current window_centroids
        self.recent_centers.append(window_centroids)

        # Smooth the centroid values over the last n iterations
        if len(self.recent_centers) > self.smooth_factor:
            self.recent_centers = self.recent_centers[-self.smooth_factor:]

        # Calculate the average centroid positions from the recent iterations
        center_vals = np.mean(self.recent_centers, axis=0)

        return center_vals
