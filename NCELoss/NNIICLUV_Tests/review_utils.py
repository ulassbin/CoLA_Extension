import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import image
import torch
import json

def json_to_gt(vid_name, json_data, num_segments, num_classes, classname_to_idx, sec_to_index=1):
    gt = np.zeros((num_segments, num_classes), dtype=np.float32)
    # database = json_data['database']
    if vid_name not in json_data['database']:
        print(f"Video {vid_name} not found in JSON data.")
        return gt
    annotations = json_data['database'][vid_name]['annotations']
    for annotation in annotations:
        label = annotation['label']
        if label not in classname_to_idx:
            print(f"Label {label} not found in class names.")
            continue
        class_idx = classname_to_idx[label]
        segment = annotation['segment']
        start_sec = segment[0]
        end_sec = segment[1]
        start_idx = int(start_sec / sec_to_index)
        end_idx = int(end_sec / sec_to_index)
        gt[start_idx:end_idx + 1, class_idx] = 1.0
    return gt


def visualize_cas(cas, gt):
    """
    Visualizes the CAS (Cumulative Accuracy Score) and ground truth values.

    Parameters:
    cas (list or np.ndarray): The CAS values to visualize.
    gt (list or np.ndarray): The ground truth values to compare against.

    Returns:
    None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(cas, label='CAS', color='blue')
    plt.plot(gt, label='Ground Truth', color='red', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('CAS vs Ground Truth')
    plt.legend()
    plt.grid()
    plt.show()
    return plt.gca()


def visualize_cas_on_raw_image(cas, gt, frames, index_to_frame_id = 16, frame_of_interest=None):
    if frame_of_interest is None:
        frame_of_interest = np.argmax(np.mean(cas.mean(axis=1)))
        frame_of_interest = index_to_frame_id * frame_of_interest + index_to_frame_id // 2
    index_of_interest = frame_of_interest // index_to_frame_id
    frame = frames[frame_of_interest]

    data = frame.copy()

    plt.subplot(2,2,1)
    plt.imshow(data)
    plt.title(f'Frame {frame_of_interest}')
    plt.subplot(2,2,2)
    plt.plot(cas, label='Cas')
    max_cas = np.max(cas)
    min_cas = np.min(cas)
    # mark the region from min_cas to max_cas and at the index of interest
    plt.gca().add_patch(mpatches.Rectangle((index_of_interest, min_cas), 1, max_cas-min_cas, color='yellow', alpha=0.5))
    plt.title('CAS Visualization')
    plt.subplot(2,2,4)
    plt.plot(gt, label='Ground Truth', color='red')
    plt.gca().add_patch(mpatches.Rectangle((index_of_interest, 0), 1, 1, color='green', alpha=0.5))
    plt.title('Ground Truth Visualization')
    plt.title('Ground Truth')

def test_visualize_cas_on_raw_image():
    # Example usage
    cas = np.random.rand(10, 10)  # Replace with actual CAS data
    gt = np.random.rand(10, 10)   # Replace with actual ground truth data
    frames = [np.random.rand(100, 100, 3) for _ in range(100)]  # Replace with actual frames
    visualize_cas_on_raw_image(cas, gt, frames, 10, frame_of_interest=50)

if __name__ == "__main__":
    test_visualize_cas_on_raw_image()
    plt.show()