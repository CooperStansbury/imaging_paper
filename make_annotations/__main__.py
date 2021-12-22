import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
import shutil
from importlib import reload
import skimage.io
import matplotlib 
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import json
import argparse


def load_config(config_path):
    """Load json with user params """
    with open(config_path) as json_file:
        data = json.load(json_file)
    return data


def load_trackmate_file(trackmate_path):
    """Read a single trackmate output file (json) and perform basic is operations """
    base = os.path.basename(trackmate_path)
    filename, _ = os.path.splitext(base)
    
    with open(trackmate_path, "r") as read_file:
        data = json.load(read_file)
    df = pd.DataFrame(data)
    
    # alwasy drop NA Ids
    df = df[df['TRACK_ID'].notna()]
    
    # add file_specific IDS
    df['TRACK_ID'] = f"{filename}_" + df['TRACK_ID'].astype(int).astype(str) 
    df['FILE'] = filename
    df['FILE_PATH'] = trackmate_path
    return df


def load_trackmate_dir(trackmate_path):
    """Load a directory of trackmake files, non-robust. assumes directory 
    only contains trackmate files """
    df_list = []
    for f in os.listdir(trackmate_path):
        if f.endswith(".csv"):
            f_path = f"{trackmate_path}{f}"
            tmp = load_trackmate_file(f_path)
            df_list.append(tmp)
    df = pd.concat(df_list, ignore_index=True)
    return df
    
    
def load_trackmate(trackmate_path):
    """A function to manage single vs. directory inputs for trackmate paths """
    if not os.path.isdir(trackmate_path):
        df = load_trackmate_file(trackmate_path)
    else:
        df = load_trackmate_dir(trackmate_path)
    return df


def load_tif_file(tif_path):
    """Function to return a single image and basename """
    base = os.path.basename(tif_path)
    filename, _ = os.path.splitext(base)
    img = skimage.io.imread(tif_path)
    return filename, img


def load_tif_dir(tif_path):
    """ function to return image basename pairs for images
    in a directory """
    files = {}
    for f in os.listdir(tif_path):
        if f.endswith(".tif"):
            f_path = f"{tif_path}{f}"
            filename, img = load_tif_file(f_path)
            files[filename] = img         
    return files


def load_tif(tif_path):
    """A function to manage single vs. directory inputs for image paths
    returns a dict
    """
    if not os.path.isdir(tif_path):
        filename, img = load_tif_file(tif_path)
        files = {}
        files[filename] = img
    else:
        files = load_tif_dir(tif_path)
    return files


def add_step_vars(df):
    """A function to add a few columns for filtering """
    df = df.sort_values(by=['TRACK_ID', 'FRAME']) 
    
    df['STEP'] =  df.groupby('TRACK_ID').transform('cumcount')
    df['MAX_STEP'] =  df.groupby('TRACK_ID')['STEP'].transform('max')
    return df


def filter_frame(df):
    """A function to filter the dataframe """
    ####### perform all filtering on a temporary dataframe
    tmp = df.copy()

    # filter by step length 
    filtered = df.groupby('TRACK_ID')['STEP'].max().reset_index()
    filtered = filtered[filtered['STEP'] > MINIMUM_TRACK_LEN]
    tmp = tmp[tmp['TRACK_ID'].isin(filtered['TRACK_ID'])]

    # filter by size
    filtered = tmp.groupby('TRACK_ID')['RADIUS'].mean().reset_index()
    filtered = filtered[filtered['RADIUS'] > MINIMIM_AVERAGE_SIZE]
    tmp = tmp[tmp['TRACK_ID'].isin(filtered['TRACK_ID'])]

    df = tmp.copy()
    df = df.sort_values(by=['TRACK_ID', 'FRAME', "FILE"])
    return df


def get_images(tiff, frame, xpos, ypos, xwin=50, ywin=50):
    """A function to return a small frame of from a position
    at a specific time from a movie
    
    expects input shape (time, y, x, 3) for RGB image
    """
    f, m, n, c = tiff.shape
    
    xmin = int(xpos - xwin)
    xmax = int(xpos + xwin)
    ymin = int(ypos - ywin)
    ymax = int(ypos + ywin)
    
    spot_x, spot_y = xwin, ywin
    
    # handle boundaries
    if xmin < 0:
        spot_x = spot_x + xmin
        xmin = 0 
    
    if xmax > n:
        xmax = n
        
    if ymin < 0:
        spot_y = spot_y + ymin
        ymin = 0

    if ymax > m:
        ymax = m

    img = tiff[frame, ymin:ymax, xmin:xmax, :]
    return img, (spot_x, spot_y)


def get_samples(df):
    """A function to sample tracks from the dataframe"""
    n = df['TRACK_ID'].nunique()
    sample_size = int(n * SAMPLE_PROPORTION)
    tracks = np.random.choice(df['TRACK_ID'].unique(), sample_size)
    sample_tracks = df[df['TRACK_ID'].isin(tracks)]
    return sample_tracks


def make_annotation_output_dir():
    # will overwrite the dir each time
    if os.path.exists(ANNOTATION_OUTPUT):
        shutil.rmtree(ANNOTATION_OUTPUT)
    os.makedirs(ANNOTATION_OUTPUT)
    return ANNOTATION_OUTPUT
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-config", nargs='?', default='make_annotations/config.json', 
                        help="The path to a processed alignment table.")
    args = parser.parse_args()
    
    # argument parsing 
    config_path = args.config
    config = load_config(config_path)
    # load all config params as variables
    # these are all UPPERCASE
    locals().update(config)
    
    df = load_trackmate(TRACKMATE_PATH)
    
    print(f"pre-filter {df['TRACK_ID'].nunique()=}")
    
    files = load_tif(TIF_PATH)
    
    df = add_step_vars(df)
    df = filter_frame(df)
    
    print(f"post-filter {df['TRACK_ID'].nunique()=}")
    df.to_csv(FILTERED_OUTPUT, index=False)
    print(f"saved: {FILTERED_OUTPUT}")
    
    # generate samples
    sample_tracks = get_samples(df)
    print(f"sample size: {sample_tracks['TRACK_ID'].nunique()}")
    print()
    
    output_dir = make_annotation_output_dir()
    
    
    """
    make annotation files
    """
    xwin = 100
    ywin = 100
    nucleus_size = 30
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 6
    plt.rcParams['figure.figsize'] = 4,4
    
    
    last_track = None
    for idx, row in sample_tracks.iterrows():
        xpos = row['POSITION_X'] * PIXEL_SCALING
        ypos = row['POSITION_Y'] * PIXEL_SCALING

        frame = int(row['FRAME'])
        track = row['TRACK_ID']
        
        if not track == last_track:
            print(f"saved: {track}")
        last_track = track

        # make sure we always have a left and right image
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        tiff = files[row["FILE"]]

        # image under question 
        img, spot = get_images(tiff, frame, xpos, ypos, xwin, ywin)

        if len(img) == 0:
            continue

        ax1.imshow(img)
        ax1.axis('off')
        spot_blob = plt.Circle(spot, nucleus_size, color='r', fill=False)
        ax1.add_patch(spot_blob)

        ax2.imshow(img[:, :, 1], cmap='Greens')
        ax2.axis('off')
        spot_blob = plt.Circle(spot, nucleus_size, color='r', fill=False)
        ax2.add_patch(spot_blob)

        ax3.imshow(img[:, :, 0], cmap='Reds')
        ax3.axis('off')
        spot_blob = plt.Circle(spot, nucleus_size, color='r', fill=False)
        ax3.add_patch(spot_blob)

        plt.title(f"{track=} {frame=}")

        frame_fname = str(frame).zfill(4)

        filename = f"{track}_frame_{frame_fname}.png"
        save_path = f"{output_dir}{filename}"

        plt.savefig(save_path,  bbox_inches='tight')
        plt.close(fig)
        
