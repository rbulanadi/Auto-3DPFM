#v2025.12.10
#Last Editted on Vero South Computer by Ralph Bulanadi
#For public use

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tifffile
from scipy import ndimage as ndi
import re
import time
import cv2
from skimage import registration
import json
from igor2 import binarywave
import pyNSID
from skimage import io
import ast
from skimage.registration import phase_cross_correlation
import SciFiReaders
import h5py


#AUTOFOCUS SECTION

# ---------- existing helpers (unchanged) ----------

def to_uint8_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    arr = np.asarray(arr)
    p1, p99 = np.percentile(arr, [1, 99])
    if p99 <= p1:
        p1, p99 = float(arr.min()), float(arr.max())
    arr = np.clip((arr.astype(np.float32) - p1) / max(p99 - p1, 1e-6), 0, 1)
    return (arr * 255.0 + 0.5).astype(np.uint8)

def downsample_for_speed(img8: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    h, w = img8.shape[:2]
    step = int(np.ceil(max(h, w) / max_dim))
    return img8[::step, ::step] if step > 1 else img8

def variance_of_laplacian(img8: np.ndarray) -> float:
    lap = ndi.laplace(img8.astype(np.float32), mode='reflect')
    return float(lap.var())

def tenengrad(img8: np.ndarray) -> float:
    fx = ndi.sobel(img8.astype(np.float32), axis=1, mode='reflect')
    fy = ndi.sobel(img8.astype(np.float32), axis=0, mode='reflect')
    return float((fx * fx + fy * fy).mean())

def iter_tiff_pages(path: Path):
    with tifffile.TiffFile(str(path)) as tf:
        if len(tf.pages) == 0:
            raise ValueError("No pages found in TIFF.")
        for i, page in enumerate(tf.pages):
            yield i, page.asarray()

# ---------- new ROI + visualization helpers ----------

def _normalize_roi(roi, shape, relative: bool):
    """
    roi: (x, y, w, h). If relative=True, values are 0..1 fractions of width/height.
    Returns integer (x, y, w, h) clamped to image bounds.
    """
    if roi is None:
        return None
    if len(roi) != 4:
        raise ValueError("roi must be a 4-tuple (x, y, w, h)")
    H, W = shape[:2]
    x, y, w, h = roi
    if relative:
        x = int(round(x * W))
        y = int(round(y * H))
        w = int(round(w * W))
        h = int(round(h * H))
    # Clamp
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return (x, y, w, h)

def _maybe_show(img8_full, img8_used, roi, title=""):
    """
    Show either the full frame with ROI overlay (if roi given),
    or the image used for scoring when no ROI (which might be downsampled).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if roi is None:
        plt.figure()
        plt.imshow(img8_used, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
        return

    # Show full frame (downsampled for speed) with ROI overlay, plus cropped patch
    plt.figure()
    plt.imshow(img8_full, cmap='gray')
    x, y, w, h = roi
    rect = Rectangle((x, y), w, h, fill=False, linewidth=2)
    plt.gca().add_patch(rect)
    plt.title(f"{title} — ROI overlay")
    plt.axis('off')

    plt.figure()
    plt.imshow(img8_used, cmap='gray')
    plt.title(f"{title} — ROI crop used")
    plt.axis('off')
    plt.show()

def _should_show_page(i, show_pages):
    """
    show_pages: 'first' | 'all' | Iterable[int]
    """
    if show_pages == 'all':
        return True
    if show_pages == 'first':
        return i == 0
    try:
        return int(i) in set(show_pages)  # list/tuple of indices
    except Exception:
        return False

# ---------- updated scoring: ROI + optional visualization ----------

def score_image_focus(
    arr: np.ndarray,
    method: str = "both",
    max_dim: int = 2048,
    roi=None,
    roi_relative: bool = False,
    show: bool = False,
    title: str = "",
):
    """
    Compute focus score(s) for a single frame, optionally on a ROI and optionally show.
    roi: (x, y, w, h). If roi_relative=True, interpret as fractions.
    show: if True, displays the full frame with ROI overlay and the crop used.
    """
    # Prepare 8-bit grayscale full-res for ROI selection
    img8_full = to_uint8_grayscale(arr)

    # Resolve/crop ROI BEFORE downsampling (pixel-accurate ROI)
    roi_int = _normalize_roi(roi, img8_full.shape, roi_relative) if roi is not None else None
    if roi_int is not None:
        x, y, w, h = roi_int
        img8_used = img8_full[y:y+h, x:x+w]
    else:
        img8_used = img8_full

    # Downsample for speed
    img8_used = downsample_for_speed(img8_used, max_dim=max_dim)

    # Show if requested
    if show:
        # Also provide a downsampled full frame for fast display
        img8_full_ds = downsample_for_speed(img8_full, max_dim=max_dim)
        _maybe_show(img8_full_ds, img8_used, roi_int, title=title)

    # Compute metrics
    scores = {}
    if method in ("lapvar", "both"):
        scores["lapvar"] = variance_of_laplacian(img8_used)
    if method in ("tenengrad", "both"):
        scores["tenengrad"] = tenengrad(img8_used)
    return scores

def focus_score_from_tiff(
    tif_path: str,
    method: str = "both",
    max_dim: int = 2048,
    aggregate: str = "max",
    return_per_page: bool = False,
    roi=None,
    roi_relative: bool = False,
    show: bool = False,
    show_pages = 'first',  # 'first' | 'all' | [indices]
):
    """
    Compute focus score(s) for a TIFF (single or multi-page) with optional ROI and visualization.

    Args:
        tif_path: path to .tif/.tiff
        method: 'lapvar' | 'tenengrad' | 'both'
        max_dim: downsample largest dimension for speed
        aggregate: combine multi-page when return_per_page=False ('max' | 'mean' | 'median')
        return_per_page: return list of per-page dicts instead of aggregate
        roi: (x, y, w, h) in pixels OR 0..1 fractions if roi_relative=True
        roi_relative: interpret roi as fractions of (W,H)
        show: if True, render preview(s) with ROI overlay/crop
        show_pages: which pages to show when show=True: 'first' (default), 'all', or list of indices

    Returns:
        If return_per_page=True: List[Dict[str, float]] per page.
        Else:
            - method == 'both' -> Dict[str, float]
            - single method -> float
    """
    if method not in ("lapvar", "tenengrad", "both"):
        raise ValueError('method must be "lapvar", "tenengrad", or "both"')
    if aggregate not in ("max", "mean", "median"):
        raise ValueError('aggregate must be "max", "mean", or "median"')

    per_page = []
    for i, frame in iter_tiff_pages(Path(tif_path)):
        scores = score_image_focus(
            frame,
            method=method,
            max_dim=max_dim,
            roi=roi,
            roi_relative=roi_relative,
            show=(show and _should_show_page(i, show_pages)),
            title=f"page {i}",
        )
        per_page.append(scores)

    if return_per_page or len(per_page) == 1:
        if not return_per_page and len(per_page) == 1:
            return per_page[0] if method == "both" else list(per_page[0].values())[0]
        return per_page

    # Aggregate across pages
    def agg_func(vals):
        if aggregate == "max":
            return float(np.max(vals))
        if aggregate == "mean":
            return float(np.mean(vals))
        return float(np.median(vals))

    if method == "both":
        keys = per_page[0].keys()
        return {k: agg_func([d[k] for d in per_page]) for k in keys}
    else:
        k = "lapvar" if method == "lapvar" else "tenengrad"
        return agg_func([d[k] for d in per_page])

# ---------- convenience wrapper ----------

def focus_score(
    tif_path: str,
    roi=None,
    roi_relative: bool = False,
    show: bool = False,
    method = "lapvar"
):
    """
    Convenience: Laplacian-variance focus score with 'max' aggregation.
    Returns a single float.
    """
    return float(
        focus_score_from_tiff(
            tif_path,
            method=method,
            aggregate="max",
            roi=roi,
            roi_relative=roi_relative,
            show=show,
            show_pages='first',
        )
    )

def get_next_filename(base_name, directory, filetype):
    """
    Generates the next available filename with a zero-padded numeric suffix,
    based on all files starting with the base name, regardless of extension.
    
    Parameters:
        base_name (str): The base filename (e.g., "Test").
        directory (str): The directory to search in.
        filetype (str): The file extension to return (e.g., ".ibw").
    
    Returns:
        str: The next available filename (e.g., "Test0002.ibw").
    """
    pattern = re.compile(rf"^{re.escape(base_name)}(\d{{4}})\..+$")
    max_index = -1

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index

    next_index = max_index + 1
    return f"{base_name}{next_index:04d}{filetype}"

def windows2igor(file_loc_windows):
    file_loc = file_loc_windows.split('\\',1)
    file_loc = file_loc[0]+file_loc[1]
    file_loc = file_loc.replace('\\', ':')
    if file_loc[-1]!= ':':
        file_loc+=':'
    return file_loc
    
    
def igor2windows(file_loc_igor):
    file_loc = file_loc_igor.split(':', 1)
    file_loc[1] = file_loc[1].replace(':', '\\')
    file_loc = file_loc[0]+':\\'+file_loc[1]
    return file_loc

def auto_focus_calc (igor,
                     file_loc = r'D:\User data',
                     base_filename = 'Image',
                     log_filename = 'LogFilename',
                     save_plot = True,
                     save_plot_filename = 'Image',
                     auto_calc_centre = True,
                     box = [300, 450, 150, 100],
                     step_size = 5,
                     test=False
                    ):

    set_folder(igor, file_loc)
    os.chdir(file_loc)

    igor.Execute('arpxl_WriteValue("zoom",2)')
    if auto_calc_centre:
        igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "O'+base_filename+'"')
        igor.Execute('PV("BaseSuffix", 0000)')
        igor.Execute('ARCheckSuffix()')
        fname = get_next_filename('O'+base_filename, file_loc, '.tif')
        igor.Execute('ARVideoButtonFunc("ARVCapture")')
        img = cv2.imread(fname)
        laser_x, laser_y = lsrCentre(img)
        box = [int(laser_x-40),int(laser_y-55), 120, 110]
    
    scores_tip = []
    scores_sample = []

    focus_positions = []
    igor.Execute('td_wv("Red Laser", 0)')
    if auto_calc_centre:
        input('Ensure Laser Off')
    
    igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "'+base_filename+'"')
    igor.Execute('MoveToTipFocusPosition()')
    time.sleep(0.1)
    
    for i in range(5):
        igor.Execute('MoveMotorRelativeCounts("Head", '+str(step_size*3)+')')
        time.sleep(0.05)
    for i in range(50):
        igor.Execute('MoveMotorRelativeCounts("Head", -'+str(step_size)+')')
        time.sleep(0.05)
        igor.Execute('PV("BaseSuffix", 0000)')
        igor.Execute('ARCheckSuffix()')
        filename = get_next_filename(base_filename, file_loc, '.tif')
        igor.Execute('ARVideoButtonFunc("ARVCapture")')
        igor.Execute("save_head_position_in_igor()")
        focus_positions.append(igor.DataFolder(r"root").Wave("focus_pos").GetNumericWavePointValue(1))
        if test:
            s_tip = focus_score(filename, roi=box, method = "tenengrad", show = True)
            break
        s_tip = focus_score(filename, roi=box, method = "tenengrad")
        s_sample = focus_score(filename, method = "tenengrad")
        scores_tip.append(s_tip)
        scores_sample.append(s_sample)
    igor.Execute('td_wv("Red Laser", 1)')

    with open(log_filename, 'a') as f:
        for h in focus_positions:
            f.write(str(h)+'\n')

    plt.plot(np.array(focus_positions)*1000, scores_tip, color = 'tab:blue', label = 'Tip Focus')
    plt.plot(np.array(focus_positions)*1000, scores_sample, color = 'tab:orange', label = 'Sample Focus')
    plt.xlabel('Focus Height (mm)')
    plt.ylabel('Focus Score (Arb.)')
    plt.legend()
    if save_plot:
        plt.savefig(save_plot_filename)
    plt.show()
    plt.close()
    tip_focus = focus_positions[np.argmax(scores_tip)]
    sample_focus = focus_positions[np.argmax(scores_sample)]
    return tip_focus, sample_focus

def auto_focus_save(igor, tip_focus, sample_focus):
    curr_tip_diff = 1
    curr_sample_diff = 1
    last_tip_diff = 1
    last_sample_diff = 1
    for i in range(100):
        igor.Execute('MoveMotorRelativeCounts("Head", 5)')
        time.sleep(0.05)
        igor.Execute("save_head_position_in_igor()")
        curr_focus_position = (igor.DataFolder(r"root").Wave("focus_pos").GetNumericWavePointValue(1))
        if abs(curr_focus_position - tip_focus) <= curr_tip_diff:
            igor.Execute('SetTipFocus()')
            last_tip_diff = curr_tip_diff
            curr_tip_diff = abs(curr_focus_position - tip_focus)
        if abs(curr_focus_position - sample_focus) <= curr_sample_diff:
            igor.Execute('SetSampleFocus()')
            last_sample_diff = curr_sample_diff
            curr_sample_diff = abs(curr_focus_position - sample_focus)
        if (curr_tip_diff > last_tip_diff) and (curr_sample_diff > last_sample_diff):
            break
    igor.Execute('MoveToSampleFocusPosition()')
    time.sleep(0.1)
    igor.Execute('ARVideoButtonFunc("ARVCapture")')
    time.sleep(0.1)
    igor.Execute('MoveToTipFocusPosition()')
    time.sleep(0.1)
    igor.Execute('ARVideoButtonFunc("ARVCapture")')
    
    
def set_base_filename(igor, base_filename):
    igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "'+base_filename+'"')
    igor.Execute('PV("BaseSuffix", 0000)')
    igor.Execute('ARCheckSuffix()')
    

def ex(igor, variable = "", panel = "", val = 0, string = "", verbose=False):
    execution_line = ""
    if verbose:
        execution_line += "print "
    execution_line += 'ARExecuteControl("'
    execution_line += variable
    execution_line += '", "'
    execution_line += panel
    execution_line += '", '
    execution_line += str(val)
    execution_line += ', "'
    execution_line += string
    execution_line += '")'
    igor.Execute(execution_line)
    return execution_line


def lsrCentre(img, plot=0, tresh=253, offset=0):
    if len(np.shape(img)) == 3:
        img_flat = np.mean(img, axis = 2)
    else:
        img_flat = img
    laser = img_flat > tresh
    centroid = find_largest_blob_centroid(laser)
    return(centroid)


def find_largest_blob_centroid(boolean_array):
    # Convert boolean to uint8 (0 or 255) for OpenCV
    binary_image = boolean_array.astype(np.uint8) * 255

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)

    if num_labels <= 1:
        return None, 0  # No blobs (only background)

    # Ignore background (label 0), find label with max area
    # stats[:, 4] is the area
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    centroid = tuple(centroids[largest_label])  # (x, y) as float
    size = stats[largest_label, cv2.CC_STAT_AREA]

    return centroid


def rot_trans_orb_then_phasecorr2(img_ref, img_mov, nfeatures=3000, upsample=20,
    return_matrix=True,):
    """
    Return (angle_deg, tx, ty) such that:
      warp img_mov by rotate(angle_deg) about center, then translate by (tx, ty) -> img_ref
    """
    if img_ref.ndim == 3: img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if img_mov.ndim == 3: img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    a = img_ref.astype(np.uint8); b = img_mov.astype(np.uint8)
    h, w = a.shape
    #center = (w/2, h/2)

    # --- rotation (and coarse translation) from ORB + RANSAC partial affine
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7, edgeThreshold=15)
    kpa, desa = orb.detectAndCompute(a, None)
    kpb, desb = orb.detectAndCompute(b, None)
    if desa is None or desb is None: raise RuntimeError("Not enough features.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desb, desa, k=2)
    good = [m for m,n in matches if n is not None and m.distance < 0.75*n.distance]
    if len(good) < 6: raise RuntimeError("Not enough good matches.")
    pts_b = np.float32([kpb[m.queryIdx].pt for m in good])
    pts_a = np.float32([kpa[m.trainIdx].pt for m in good])

    A, _ = cv2.estimateAffinePartial2D(
        pts_b, pts_a, method=cv2.RANSAC,
        ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999
    )
    if A is None: raise RuntimeError("estimateAffinePartial2D failed")

    a00, a01, tx0 = A[0,0], A[0,1], A[0,2]
    a10, a11, ty0 = A[1,0], A[1,1], A[1,2]
    angle_deg = float(np.degrees(np.arctan2(a10, a00)))
    

    center = lsrCentre(img_ref, tresh = 250)
    # --- refine pure translation after removing rotation
    Mrot = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    b_rot = cv2.warpAffine(b, Mrot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return angle_deg, b_rot

def save_scan_location(igor,
                       file_loc = r'D:\User data',
                       base_filename = 'Image',
                       set_strength = 15):
    ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", 0.1)
    
    igor.Execute('arpxl_WriteValue("zoom",1)')
    set_folder(igor, file_loc)
    igor.Execute('MoveToSampleFocusPosition()')
    axis1 = "output.x"
    sign = 1
    axis2 = 0
    command_list = ['PTS("CimPhase","'+str(axis1)+'")',
                    'PV("CimPhase",1)',
                    'PVU("CimPhase",'+str(sign)+')',
                    'PVMU("cimPhase",'+str(axis2)+')',
                   ]
    for command in command_list:
        igor.Execute(command)
    igor.Execute('CimCallback()')
    
    set_base_filename(igor, base_filename)
    target_filename = get_next_filename(base_filename, file_loc, '.tif')
    time.sleep(0.1)
    
    igor.Execute('ARVideoButtonFunc("ARVCapture")')
    time.sleep(0.1)
    
    igor.Execute('MoveToTipFocusPosition()')
    
    if set_strength:
        ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", set_strength)
    
    return target_filename

def approach_location(igor,
                      target_filename,
                      file_loc = r'D:\User data',
                      base_filename = 'Image',
                      verbose = True,
                      use_mask = True,
                      set_strength = 15,
                      init_strength = 15
                     ):
    igor.Execute('arpxl_WriteValue("zoom",1)')
    set_folder(igor, file_loc)
    os.chdir(file_loc)
    
    target = cv2.imread(target_filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:,200:]
    set_base_filename(igor, base_filename)
    igor.Execute('MoveToSampleFocusPosition()')
    
    strength = init_strength

    ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", strength)
    last_corr = np.array([0,  0])
    ready_to_scan = False
    runs_on_current_strength = 0
    mask = np.ones_like(target)
    mask[1000:1300, 0:1000] = 0
    mask[:,:100]=0

    for i in range(15):
        igor.Execute('PV("BaseSuffix", 0000)')
        igor.Execute('ARCheckSuffix()')
        filename = get_next_filename(base_filename, file_loc, '.tif')
        igor.Execute('ARVideoButtonFunc("ARVCapture")')
        time.sleep(0.1)

        now = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:,200:]
        #corr = phase_cross_correlation(img_o, img, reference_mask = mask, moving_mask = mask)
        angle_deg, b_rot = rot_trans_orb_then_phasecorr2(target, now)
        if use_mask:
            corr = registration.phase_cross_correlation(target, b_rot, reference_mask = mask)
        else:
            corr = registration.phase_cross_correlation(target, b_rot)[0]
        ang = -np.radians(angle_deg)
        matrix = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        new_corr = np.dot(corr, matrix)
        un = np.roll(np.roll(b_rot, int(corr[0]), axis=0), int(corr[1]), axis=1)
        if verbose:
            plt.figure(figsize = (10,10))
            plt.subplot(1,2,1)
            plt.title('Overlaid')
            plt.imshow(target-b_rot)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,2,2)
            plt.title('Adjusted')
            plt.imshow(target-un)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            plt.close()

        
        if max(abs(last_corr-new_corr)) > max(abs(new_corr)) and runs_on_current_strength >= 2:
            if strength == 1:
                break
            strength = int(np.ceil(strength*1/2))
            ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", strength)
        else:
            runs_on_current_strength += 1

        if corr[0]**2+corr[1]**2 <=4:
            ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", strength)
            ready_to_scan = False
            break

        if abs(new_corr[0]) > abs(new_corr[1]):
            #move up down
            axis1 = "output.y"
            axis2 = 1
            sign = -np.sign(new_corr[0]) #1 for right or down; -1 for left or up
            if sign == 1:
                direction = 'Down'
            else:
                direction = 'Up'
        else:
            #move left right
            axis1 = "output.x"
            axis2 = 0
            sign = -np.sign(new_corr[1]) #1 for right or down; -1 for left or up
            if sign == 1:
                direction = 'Right'
            else:
                direction = 'Left'

        if sign == 0:
            break
        else:
            command_list = ['PTS("CimPhase","'+str(axis1)+'")',
                            'PV("CimPhase",1)',
                            'PVU("CimPhase",'+str(sign)+')',
                            'PVMU("cimPhase",'+str(axis2)+')',
                           ]
            for command in command_list:
                igor.Execute(command)
            igor.Execute('CimCallback()')
        last_corr = new_corr

        time.sleep(0.1)
        if verbose:
            print('Moving: '+direction)
            print('Rotation Angle: '+str(angle_deg))
            print('Correction: ' +str(corr))
            print('Strength: ' +str(strength))
            print('---')
        
    ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", 15)    
    igor.Execute('MoveToTipFocusPosition()')
    
    if set_strength:
        ex(igor, "CimStrenSetVar_2", "MasterMotorPanel", set_strength)
        
def DoLDMove(igor, micronsX, micronsY):
    igor.Execute('DoLDMove('+str(micronsX)+', '+str(micronsY)+')')
    
    
def saveprint(text, f, verbose=True):
    if verbose:
        print(text)
    if f:
        f.write(text+'\n')
        
        
        
def RelLsrPos(fname, plot=True, LsrThresh = 253, rounding=True, length = None, width = None, box=[0,0,384,512]):
    img = cv2.imread(fname)
    img = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    if img is None:
        raise ValueError("Image not found")
    sqImg, angD = angRemover(img)
    # Convert to grayscale and blur
    blue = sqImg[:, :, 0].astype(float)
    green = sqImg[:, :, 1].astype(float)

    # Compute grayscale manually without the red channel
    # Typical luminance weights (adjusted without red): 0.587 for green, 0.114 for blue
    gray_custom = (0.587 * green + 0.114 * blue).astype(np.uint8)
    gray = cv2.cvtColor(sqImg, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_custom, (11, 11), 0)
    
    # Otsu threshold (standard, no inversion)
    #thresh = cv2.adaptiveThreshold(
    #    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 2
    )
    edges = ~thresh
    blur_edges = cv2.GaussianBlur(edges, (11,11), 0)
    edges = (blur_edges > 195).astype('uint8')
    kernel_size = 3
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    kernel_size = 1
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges = cv2.dilate(edges, kernel_1, iterations=1)
    edges = cv2.erode(edges, kernel_2, iterations=1)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=90,
        maxLineGap=2
    )
    
    # Prepare to store horizontal lines
    horizontal_lines = []
    
    # Loop through lines and filter for roughly horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if the line is roughly horizontal
            if abs(y2 - y1) < 20:  # Tolerance for horizontal lines
                if x1 < img.shape[1] * 0.15:  # Ensure starting from the left border
                    horizontal_lines.append((x1, y1, x2, y2))


    y_positions = [((y1 + y2) / 2) for (x1, y1, x2, y2) in horizontal_lines]
    # Determine the gap to divide into two groups
    min_y = min(y_positions)
    max_y = max(y_positions)
    gap = (max_y - min_y) / 2  # You can adjust this if needed
    
    # Group lines
    horizontal_lines_1 = []
    horizontal_lines_2 = []
    for line in horizontal_lines:
        mid_y = (line[1] + line[3]) / 2
        if mid_y < (min_y + gap):
            horizontal_lines_1.append(line)
        else:
            horizontal_lines_2.append(line)
            
    approx_x_position = np.mean([np.mean(np.sort(np.array(horizontal_lines_1)[:,2])[-2:]),
                                 np.mean(np.sort(np.array(horizontal_lines_2)[:,2])[-2:])])
    approx_y1_position = np.mean(np.array(horizontal_lines_1)[:,3])
    approx_y2_position = np.mean(np.array(horizontal_lines_2)[:,3])

    
    lines_short = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=25,
        minLineLength=35,
        maxLineGap=5
    )
    
    # Prepare to store horizontal lines
    diagonal_lines_1 = []
    diagonal_lines_2 = []

    tolerance = 15
    x_tolerance = 15
    extra_tolerance = 15
    # Loop through lines and filter for roughly horizontal lines
    all_short_lines = []
    if lines_short is not None:
        for line in lines_short:
            x1, y1, x2, y2 = line[0]
            # Check if the line is roughly horizontal
            if abs(x1-approx_x_position) < tolerance+x_tolerance:  # Tolerance for horizontal lines
                if abs(y1-approx_y1_position) < tolerance:  # Tolerance for horizontal lines
                    if abs(x2-(approx_x_position+gap)) < tolerance+extra_tolerance+x_tolerance:  # Tolerance for horizontal lines
                        if abs(y2-(approx_y1_position+gap)) < tolerance+extra_tolerance:  # Tolerance for horizontal lines
                            if abs(np.arctan2(y2 - y1, x2 - x1) - (np.pi / 4)) < (np.pi / 4):
                                diagonal_lines_1.append((x1, y1, x2, y2))
            if abs(x1-approx_x_position) < tolerance+x_tolerance:  # Tolerance for horizontal lines
                if abs(y1-approx_y2_position) < tolerance:  # Tolerance for horizontal lines
                    if abs(x2-(approx_x_position+gap)) < tolerance+extra_tolerance+x_tolerance:  # Tolerance for horizontal lines
                        if abs(y2-(approx_y1_position+gap)) < tolerance+extra_tolerance:  # Tolerance for horizontal lines
                            if abs(np.arctan2(y2 - y1, x2 - x1) - (-np.pi / 4)) < (np.pi / 4):
                                diagonal_lines_2.append((x1, y1, x2, y2))
            all_short_lines.append((x1,y1,x2,y2))

    x_1_test = approx_x_position
    y_1_test = approx_y1_position
    x_2_test = approx_x_position+gap
    y_2_test = approx_y1_position+gap
    x_3_test = approx_x_position
    y_3_test = approx_y2_position
    
    if plot:
        image_with_lines = sqImg.copy()
        
        # Draw the detected horizontal lines on the image
        
        line_colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (255,0,255)]
        for line_type_i, line_type in enumerate([horizontal_lines_1, horizontal_lines_2, diagonal_lines_1, diagonal_lines_2]):#, all_short_lines]):
            #for line_type_i, line_type in enumerate([horizontal_lines_1, horizontal_lines_2, diagonal_lines_1, diagonal_lines_2, all_short_lines]):
            for x1, y1, x2, y2 in line_type:
                cv2.line(image_with_lines, (x1, y1), (x2, y2), line_colors[line_type_i], 2)
        
        # Convert the image from BGR to RGB for plotting
        image_with_lines_rgb = cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB)
        
        # Plot the original image and the image with detected lines side by side
        plt.figure(figsize=(5, 2))
        plt.subplot(1, 2, 1)
        plt.title('Detected Contours')
        plt.imshow(image_with_lines_rgb)
        plt.axis('off')
        #plt.show()
        

    flat_horz_lines_1 = safe_mean_line(horizontal_lines_1)
    flat_horz_lines_2 = safe_mean_line(horizontal_lines_2)
    flat_diag_lines_1 = safe_mean_line(diagonal_lines_1)
    flat_diag_lines_2 = safe_mean_line(diagonal_lines_2)
    
    
    topmost = np.nanmean([flat_horz_lines_1[3],flat_diag_lines_1[1]])
    botmost = np.nanmean([flat_horz_lines_2[3],flat_diag_lines_2[1]])
    midmost = np.nanmean([topmost, botmost])
    if not width:
        width = topmost - botmost
    else:
        topmost = midmost+width/2
        botmost = midmost-width/2
        
    rightmost = np.nanmean([flat_diag_lines_1[2],flat_diag_lines_2[2]])
    if not length:
        centremost = np.nanmean([flat_diag_lines_1[0],flat_diag_lines_2[0], flat_horz_lines_1[2],flat_horz_lines_2[2]])
        #centremost = np.nanmean([flat_diag_lines_1[0],flat_diag_lines_2[0]])
        leftmost = 2*centremost - rightmost
        length = rightmost-leftmost
    else:
        centremost = rightmost-(length/2)
        leftmost = rightmost-(length)

    top = [topmost, centremost]
    bot = [botmost, centremost]
    right = [midmost, rightmost]
    left = [midmost, leftmost]

    minTipx = leftmost
    maxTipx = rightmost
    minTipy = topmost
    maxTipy = botmost

    centre = lsrCentre(gray)
    #the final calcul
    RelPosx = lsrPurcent(centre[0], minTipx, maxTipx)
    RelPosy = lsrPurcent(centre[1], minTipy, maxTipy)
    if rounding :
        RelPosRx = round(RelPosx,2)
        RelPosRy = round(RelPosy,2)
        

    if plot:
        plt.subplot(1, 2, 2)
        plt.title('Diamond')
        plt.imshow(cv2.cvtColor(sqImg, cv2.COLOR_BGR2RGB))


        plt.axvline(maxTipx, color='blue')
        plt.axhline(minTipy, color='blue')
        plt.axhline(np.mean([minTipy,maxTipy]), color='blue')
        plt.axhline(maxTipy, color='blue')
        plt.scatter(centre[0],centre[1], marker="o", color="red", s=40)

        for point in [top, bot, right, left]:
            plt.scatter(point[1],point[0], marker="o", color="green", s=40)
        plt.plot([top[1], bot[1]], [top[0], bot[0]], color='green', linestyle = 'dashed')
        plt.plot([left[1], right[1]], [left[0], right[0]], color='green', linestyle = 'dashed')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    centre_converted = convert_coordinates(top, bot, left, right, [centre[1], centre[0]])
    return centre_converted, angD, length, width


def lsrPurcent(centre, minTip, maxTip):
    lenght = maxTip-minTip
    p = (centre-minTip)/lenght
        
    return(p)

def convert_coordinates(top, bot, left, right, centre):
    #print(centre)
    range_i = bot[0]-top[0]
    new_i = +1-((centre[0]-top[0])*2/range_i)
    
    range_j = right[1]-left[1]
    new_j = -1+((centre[1]-left[1])*2/range_j)
    
    return [new_j, new_i]

def do_scan(igor, wait=False, mode = None, file_loc = None, base_filename = None):
    if file_loc:
        set_folder(igor, file_loc)
    else:
        file_loc = check_folder(igor)
    if base_filename:
        set_base_filename(igor, base_filename)
    else:
        base_filename = check_base_filename(igor)
    if mode:
        ex(igor, 'LastScanPopup_0', "MasterPanel", 0, mode)
    filename = get_next_filename(base_filename, file_loc, '.ibw')
    
    ex(igor, 'DownScan_0','MasterPanel')
    time.sleep(10)
    if wait:
        while scanning(igor):
            time.sleep(1)
    if file_loc[-1] != '\\':
        file_loc+= '\\'
    full_path = file_loc+filename
    return full_path

def scanning(igor):
    data = igor.DataFolder(r"root:packages:MFP3D").Wave("OutWaves")
    outputting = bool(data.GetTextWavePointValue(0,0))
    return outputting

def angRemover(img):
    blue = img[:, :, 0].astype(float)
    green = img[:, :, 1].astype(float)

    # Compute grayscale manually without the red channel
    # Typical luminance weights (adjusted without red): 0.587 for green, 0.114 for blue
    gray_custom = (0.587 * green + 0.114 * blue).astype(np.uint8)
    blur = cv2.GaussianBlur(gray_custom, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray_custom, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 15, 2
    )

    edges = ~thresh
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=130,
        minLineLength=80,
        maxLineGap=5
    )
    
    # Prepare to store horizontal lines
    horizontal_lines = []
    
    # Loop through lines and filter for roughly horizontal lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if the line is roughly horizontal
            if abs(y2 - y1) < 20:  # Tolerance for horizontal lines
                if x1 < img.shape[1] * 0.1:  # Ensure starting from the left border
                    horizontal_lines.append((x1, y1, x2, y2))


    y_positions = [((y1 + y2) / 2) for (x1, y1, x2, y2) in horizontal_lines]
    # Determine the gap to divide into two groups
    min_y = min(y_positions)
    max_y = max(y_positions)
    gap = (max_y - min_y) / 2  # You can adjust this if needed
    
    # Group lines
    horizontal_lines_1 = []
    horizontal_lines_2 = []
    for line in horizontal_lines:
        mid_y = (line[1] + line[3]) / 2
        if mid_y < (min_y + gap):
            horizontal_lines_1.append(line)
        else:
            horizontal_lines_2.append(line)

    flattened_horz_lines_1 = np.mean(np.array(horizontal_lines_1),axis = 0)
    angle_1 = np.degrees(np.arctan2(flattened_horz_lines_1[2]-flattened_horz_lines_1[0], flattened_horz_lines_1[3]-flattened_horz_lines_1[1]))
    flattened_horz_lines_2 = np.mean(np.array(horizontal_lines_2),axis = 0)
    angle_2 = np.degrees(np.arctan2(flattened_horz_lines_2[2]-flattened_horz_lines_2[0], flattened_horz_lines_2[3]-flattened_horz_lines_2[1]))

    angD = np.mean([angle_2, angle_1])-90

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Define the rotation matrix: rotate by 1 degree
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, -angD, scale)
    
    # Perform the rotation
    rotated = cv2.warpAffine(img, M, (w, h))

    sqImg = rotated[20:-20, 20:-20]
    return sqImg, angD


def safe_mean_line(lines):
    return np.mean(np.array(lines), axis=0) if len(lines) > 0 else [np.nan, np.nan, np.nan, np.nan]

def compute_affine_transform(lab_coords, asylum_coords):
    # Convert to numpy arrays
    lab = np.array(lab_coords)
    asylum = np.array(asylum_coords)

    # Add ones for affine transform (to handle translation)
    A = np.hstack([lab, np.ones((lab.shape[0], 1))])  # Shape: (n_points, 3)

    # Solve for affine transform matrix (2x3) using least squares
    T, _, _, _ = np.linalg.lstsq(A, asylum, rcond=None)  # asylum = A @ T

    return T  # Shape: (3, 2)


def apply_affine_transform(T, point):
    # Append 1 to the point to make it [x, y, 1]
    point_augmented = np.append(point, 1)
    transformed = point_augmented @ T  # Shape: (2,)
    return transformed


def approach_coordinate(igor, T, new_lab_coord, limit = 70, reps = 1):
    for i in range(reps):
        mapped_asylum_coord = apply_affine_transform(T, new_lab_coord)
        target_x = mapped_asylum_coord[0]
        target_y = mapped_asylum_coord[1]

        igor.Execute("save_laser_position_in_igor()")
        LDX_x = igor.DataFolder(r"root").Wave("LDX_pos").GetNumericWavePointValue(0)
        LDX_y = igor.DataFolder(r"root").Wave("LDX_pos").GetNumericWavePointValue(1)

        x_move = (-LDX_x+target_x)/10
        y_move = (-LDX_y+target_y)/10

        if abs(x_move) > limit:
            raise('Error; x_move above limit')
        if abs(y_move) > limit:
            raise('Error; y_move above limit')

        DoLDMove(igor, x_move, y_move)
        time.sleep(0.5)
        DoLDMove(igor, x_move, y_move)
        time.sleep(0.5)

def find_curr_position_on_diamond_optically(igor,
                                            file_loc,
                                            base_filename,
                                            log_filename,
                                            verbose = True):
    orig_filename = check_base_filename(igor)
    json_filename = base_filename
    with open(json_filename+'.json', "r") as f:
        content = f.read()
    
    # Extract each [...] block
    blocks = content[1:-1].split('][')
    
    # Merge all dicts into one
    final_dict = {}
    for block in blocks:
        obj = json.loads(block)
        final_dict.update(obj)
    
    with open (log_filename+'.txt', 'a') as f: 
        set_base_filename(igor, 'O'+base_filename)
        fname = get_next_filename('O'+base_filename, file_loc, '.tif')
        igor.Execute('ARVideoButtonFunc("ARVCapture")')
        first_run = False
        time.sleep(0.2)
        img = cv2.imread(fname)
        laser_y = lsrCentre(img)[1]
        box = [0,int(laser_y-128), 384, 256]

        pos, angD, length, width = RelLsrPos(fname,plot=verbose,length=float(final_dict['diamond_length']), width = float(final_dict['diamond_width']), box=box)
        saveprint('Calculated Position:', f, verbose)
        saveprint(str(pos), f, verbose)
    set_base_filename(igor, 'O'+orig_filename)
    return pos

def approach_coordinate_open(igor, file_loc, base_filename, log_filename, new_lab_coord, limit = 70, reps = 5):
     
    igor.Execute('arpxl_WriteValue("zoom",4)')   
    for i in range(reps):
        curr_lab_coord = find_curr_position_on_diamond_optically(igor,
                                                file_loc = file_loc,
                                                base_filename = base_filename,
                                                log_filename = log_filename,
                                                verbose = False)
        factor = 1/(2**i)
        json_filename = base_filename
        with open(json_filename+'.json', "r") as f:
            content = f.read()
        
        # Extract each [...] block
        blocks = content[1:-1].split('][')
        
        # Merge all dicts into one
        final_dict = {}
        for block in blocks:
            obj = json.loads(block)
            final_dict.update(obj)
        W = float(final_dict['tip_width'])
    
        x_diff = new_lab_coord[0]-curr_lab_coord[0]
        y_diff = new_lab_coord[1]-curr_lab_coord[1]
    
        x_move = x_diff*(W/2)*1e6*factor
        y_move = y_diff*(W/2)*1e6*factor
    
        if abs(x_move) > limit:
            raise('Error; x_move above limit')
        if abs(y_move) > limit:
            raise('Error; y_move above limit')
    
        DoLDMove(igor, x_move, y_move)
        time.sleep(1)

    
    curr_lab_coordinate = find_curr_position_on_diamond_optically(igor,
                                            file_loc = file_loc,
                                            base_filename = base_filename,
                                            log_filename = log_filename,
                                            verbose = True)

def compute_piezoresponse_c(amplitude_data, phase_data):
    """
    Compute the piezoresponse using the formula: Piezoresponse = Amplitude * cos(Phase)
    
    Args:
        amplitude_data (np.array): The amplitude data.
        phase_data (np.array): The phase data (in degrees).
        
    Returns:
        np.array: The calculated piezoresponse.
    """
    # Convert phase from degrees to radians
    phase_radians = np.radians(phase_data)
    
    # Compute piezoresponse
    piezoresponse_data = (amplitude_data * np.exp(1j*phase_radians))
    
    return piezoresponse_data


def check_lab_coordinates_are_reasonable(curr_coordinates, last_coordinates, last_direction):
    if np.isnan(curr_coordinates[0]) or np.isnan(curr_coordinates[1]):
        return False
    if (abs(curr_coordinates[0]) > 1.5) or (abs(curr_coordinates[1]) > 1.5):
        return False
    if np.sign(curr_coordinates[0]-last_coordinates[0]) * np.sign(last_direction[0]) < 0:
        return False
    if np.sign(curr_coordinates[1]-last_coordinates[1]) * np.sign(last_direction[1]) < 0:
        return False
    return True


def calculate_asylum_coordinate_to_lab_coordinate_transform(igor,
                                                             file_loc = r'D:\User data',
                                                             base_filename = 'Image',
                                                             log_filename = 'LogFilename',
                                                             auto_calc_centre = True,
                                                             box=[0,128,384,256],
                                                             verbose = True):
    igor.Execute('MoveToTipFocusPosition()')      
    igor.Execute('arpxl_WriteValue("zoom",4)')          
    
    
    set_folder(igor, file_loc)
    os.chdir(file_loc)
    asylum_coordinates = []
    lab_coordinates = []
    angD = None
    length = None
    width = None
    
    last_lab_coordinates = [0,0]
    last_motion = [0,0]
    first_run = True
    reasonable_count = 0
    unreasonable_count = 0
    
    with open (log_filename+'.txt', 'a') as f: 
        igor.Execute("LoadTestProcedures()")
        time.sleep(1)
        for position in [[0,0], [-3,-3], [3,-3], [3,3], [3,3], [-3,3],[-3,3],[-3,-3], [-3,-3],[6,0]]:
            
            DoLDMove(igor, position[0], position[1])
            time.sleep(1)
            igor.Execute("save_laser_position_in_igor()")
            LDX_x = igor.DataFolder(r"root").Wave("LDX_pos").GetNumericWavePointValue(0)
            LDX_y = igor.DataFolder(r"root").Wave("LDX_pos").GetNumericWavePointValue(1)
            
            saveprint('---', f)
            igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "O'+base_filename+'"')
            igor.Execute('PV("BaseSuffix", 0000)')
            igor.Execute('ARCheckSuffix()')
            fname = get_next_filename('O'+base_filename, file_loc, '.tif')
            igor.Execute('ARVideoButtonFunc("ARVCapture")')
            saveprint('Capturing Image:', f)
            saveprint('   ' + fname, f)
            
            if auto_calc_centre and first_run:
                
                img = cv2.imread(fname)
                laser_y = lsrCentre(img)[1]
                box = [0,int(laser_y-128), 384, 256]
            
            if (verbose) and first_run:
                plt.figure(figsize = (3,4))
                img = cv2.imread(fname)

                rect=plt.Rectangle((box[0],box[1]), box[2],box[3], color='red', fill=False)
                #print(np.shape(img))
                plt.imshow(img)
                plt.gca().add_patch(rect)
                plt.xticks([])
                plt.yticks([])
                plt.show()
                plt.close()
            first_run = False
            pos, angD, length, width = RelLsrPos(fname,plot=verbose,length=length, width = width, box=box)
            saveprint('Calculated Position:', f)
            saveprint(str(pos), f)
            curr_lab_coordinates = [pos[0],pos[1]]
            last_motion[0] += position[0]
            last_motion[1] += position[1]
            if check_lab_coordinates_are_reasonable(curr_lab_coordinates, last_lab_coordinates, position):
                lab_coordinates.append([pos[0], pos[1]])
                asylum_coordinates.append([LDX_x, LDX_y])
                last_lab_coordinates = curr_lab_coordinates
                last_motion = [0,0]
                saveprint('Reasonable', f)
                reasonable_count += 1
            else:
                saveprint('Unreasonable', f)
                unreasonable_count += 1
        T = compute_affine_transform(lab_coordinates, asylum_coordinates)
        saveprint('Affine Transform:', f)
        saveprint(str(T), f)
        
        saveprint('Valid Diamonds:', f)
        saveprint(str(reasonable_count)+" out of "+str(reasonable_count+unreasonable_count), f)
    approach_coordinate(igor, T, [0,0])
    
    thermal_data = igor.DataFolder(r"root:packages:MFP3D:Main:Variables").wave(r"ThermalVariablesWave")
    thermal_dim = thermal_data.GetDimensions()[1]
    tmv = {}
    for i in range(thermal_dim):
        tmv[thermal_data.DimensionLabel(0,i,0)] = thermal_data.GetNumericWavePointValue(i)
    L = tmv['CustomLeverLength']
    W = tmv['CustomLeverWidth']
    
    with open(base_filename+".json", "a") as f:
        json.dump([{"diamond_length": length,
                    "diamond_width": width,
                    "tip_length": L,    #FIX
                    "tip_width": W,      #FIX
                    "affine_transform": T.tolist()
                   }], f)
    return T


def measure_tip_height(igor, 
                       T,
                       file_loc = r'D:\User data',
                       base_filename = 'Image',
                       log_filename = 'LogFilename',
                       closed_LD = True
                       ):
    set_folder(igor, file_loc)
    os.chdir(file_loc)
    with open (log_filename+'.txt', 'a') as f: 
        saveprint('---', f)
        if closed_LD:
            approach_coordinate(igor, T, [0,0])
        else:
            approach_coordinate_open(igor, file_loc, base_filename, log_filename, [0,0])
        
        fname = get_next_filename('O'+base_filename, file_loc, '.tif')
        igor.Execute('ARVideoButtonFunc("ARVCapture")')
        saveprint('Capturing Image:', f)
        saveprint('   ' + fname, f)
        igor.Execute("save_focus_positions()")
        sample_pos = igor.DataFolder(r"root").Wave("sample_pos").GetNumericWavePointValue(0)
        tip_pos = igor.DataFolder(r"root").Wave("tip_pos").GetNumericWavePointValue(1)
        note = 'F'
        data = igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Wave("MasterVariablesWave")
        dim = data.GetDimensions()[1]
        gmv = {}
        for i in range(dim):
            gmv[data.DimensionLabel(0,i,0)] = data.GetNumericWavePointValue(i)
        ex(igor, "TriggerChannelPopup_1", "MasterPanel", 0, "DeflVolts")
        ex(igor, "TriggerPointSetVar_1", "MasterPanel", gmv["DeflectionSetpointVolts"])
        igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "'+note+base_filename+'"')
        igor.Execute('PV("BaseSuffix", 0000)')
        igor.Execute('ARCheckSuffix()')
        fname = get_next_filename(note+base_filename, file_loc, '.ibw')
        igor.Execute("AutoWedge()")
        time.sleep(3)
        saveprint('Doing Force Curve:', f)
        saveprint('   ' + fname, f)
        ex(igor, "SingleForce_1", "MasterPanel", 1)
        time.sleep(5)
        file = binarywave.load(fname)
        z = file['wave']['wData'][:,0]
        defl = file['wave']['wData'][:,1]
        engage_i = np.argmin(defl[:np.argmax(defl)])
        engage_height = z[engage_i]-z[0]
        h = (tip_pos-sample_pos-engage_height-1e-6)/np.cos(np.radians(11))
        saveprint('Tip Focus:', f)
        saveprint('   ' + str(tip_pos), f)
        saveprint('Sample Focus:', f)
        saveprint('   ' + str(sample_pos), f)
        saveprint('Engage Height:', f)
        saveprint('   ' + str(engage_height), f)
        saveprint('Calculated h:', f)
        saveprint('   ' + str(h), f)
        saveprint('---', f)

    with open(base_filename+".json", "a") as f:
        json.dump([{"h": str(h),
                    "tip_pos": str(tip_pos),
                    "sample_pos": str(sample_pos),
                    "engage_height": str(engage_height),
                    "force_curve_filename": fname,
                   }], f)

    return h 

def run_tip_location_scans_at_points(igor,
                                     T,
                                     sequence= ['C', 'N', 'S', 'E', 'W', 'C'],
                                     file_loc = r'D:\User data',
                                     base_filename = 'Image',
                                     log_filename = 'LogFilename',
                                     dist_proportion = 0.6,
                                     dist_proportion_y = None,
                                     dist_proportion_x = None,
                                     closed_LD = True):
    
    set_folder(igor, file_loc)
    os.chdir(file_loc)
    ex(igor, 'LastScanPopup_0', "MasterPanel", 0, "One Frame Mode")
    if not dist_proportion_y:
        dist_proportion_y = dist_proportion
    if not dist_proportion_x:
        dist_proportion_x = dist_proportion
    with open (log_filename+'.txt', 'a') as f: 
        filelist= []
        original_invols = None
        for target_i in range(len(sequence)):
            if sequence[target_i] == 'C':
                main_target = [0,0]
            elif sequence[target_i] == 'N':
                main_target = [0,dist_proportion_y]
            elif sequence[target_i] == 'S':
                main_target = [0,-dist_proportion_y]
            elif sequence[target_i] == 'E':
                main_target = [dist_proportion_x,0]
            elif sequence[target_i] == 'W':
                main_target = [-dist_proportion_x,0]
            elif sequence[target_i] == 'A':
                main_target = [0,dist_proportion_y]
            elif sequence[target_i] == 'X':
                main_target = [-dist_proportion_x,dist_proportion_y]
            elif sequence[target_i] == 'Y':
                main_target = [0,-dist_proportion_y]
            elif sequence[target_i] == 'B':
                with open(base_filename+'.json', "r") as f2:
                    content = f2.read()
                blocks = content[1:-1].split('][')
                final_dict = {}
                for block in blocks:
                    obj = json.loads(block)
                    final_dict.update(obj)
                x_b = float(final_dict['x_b'])
                main_target = [x_b,0]
            
            note = sequence[target_i]            
            if closed_LD:
                approach_coordinate(igor, T, main_target)
            else:
                approach_coordinate_open(igor, file_loc, base_filename, log_filename, main_target)
            
            time.sleep(5)
            if original_invols is None:
                original_invols = measure_invols(igor, file_loc, 'F'+base_filename, log_filename)
                gmv = get_gmv(igor)
                original_setpoint = gmv['DeflectionSetpointVolts']
            else:
                curr_invols = measure_invols(igor, file_loc, 'F'+base_filename, log_filename)
                curr_setpoint = original_setpoint * original_invols/curr_invols
                if abs(curr_setpoint) < 10:
                    igor.Execute('root:Packages:MFP3D:Main:Variables:MasterVariablesWave[20][0] = '+str(curr_setpoint))
                else:
                    print('NOMINAL SET POINT IS TOO HIGH:' + str(curr_setpoint))
            
            igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "'+note+base_filename+'"')
        
            fname = get_next_filename('O'+base_filename, file_loc, '.tif')
            igor.Execute('ARVideoButtonFunc("ARVCapture")')
            saveprint('Capturing Image:', f)
            saveprint('   ' + fname, f)
            optical_fname = fname
            
            igor.Execute('PV("BaseSuffix", 0000)')
            igor.Execute('ARCheckSuffix()')
            fname = get_next_filename(note+base_filename, file_loc, '.ibw')
            time.sleep(3)
            igor.Execute("AutoWedge()")
            time.sleep(3)
            ex(igor, "MotorEngageButton_0", "MasterMotorPanel")
            time.sleep(1)
            saveprint('Doing Scan:', f)
            saveprint('   ' + fname, f)
            do_scan(igor, wait = True)
            filelist.append(fname)
            with open(base_filename+".json", "a") as f2:
                json.dump([{note+"_Filename": fname,
                            note+"_Location_x": main_target[0],
                            note+"_Location_y": main_target[1],
                            note+"_Optical_Filename": optical_fname,
                           }], f2)
                curr_lab_coord = find_curr_position_on_diamond_optically(igor,
                                        file_loc = file_loc,
                                        base_filename = base_filename,
                                        log_filename = log_filename,
                                        verbose = False)
                json.dump([{note+"_RealLocation_x": curr_lab_coord[0],
                            note+"_RealLocation_y": curr_lab_coord[1],
                           }], f2)
    set_base_filename(igor, base_filename)
    return filelist
    
def check_base_filename(igor):
    return igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Variable("BaseName").GetStringValue(0)

def set_folder(igor, file_loc):
    file_loc_igor = windows2igor(file_loc)
    igor.Execute('root:Packages:MFP3D:Main:Strings:GlobalStrings[20] = "'+file_loc_igor+'"')
    igor.Execute('InsertNewPathInHistory("'+file_loc_igor+'")')
    igor.Execute('root:Packages:MFP3D:Main:Strings:GlobalStrings[18] = "'+file_loc_igor+'"')
    igor.Execute('root:Packages:MFP3D:Main:Strings:GlobalStrings[19] = "'+file_loc_igor+'"')
    igor.Execute('root:Packages:MFP3D:Main:Strings:GlobalStrings[21] = "'+file_loc_igor+'"')
    igor.Execute('BuildFileFolder("SaveForce","'+file_loc_igor+'")')
    igor.Execute('NewPath/O SaveImage "'+file_loc_igor+'"' )
    igor.Execute('NewPath/O SaveForce "'+file_loc_igor+'"' )
    igor.Execute('PV("BaseSuffix", 0000)')
    igor.Execute('ARCheckSuffix()')
    igor.Execute('UpdateStatusText()')
    igor.Execute('UpdateHDDStrength()')
    
def check_folder(igor):
    file_loc_igor = igor.DataFolder(r"root:Packages:MFP3D:Main:Strings").Wave("GlobalStrings").GetTextWavePointValue(18, 0)
    return igor2windows(file_loc_igor)

def quick_load_ibw(full_path):
    return np.fliplr((binarywave.load(full_path)['wave']['wData']).T)


def measure_invols(igor, 
                  file_loc = r'D:\User data',
                  base_filename = 'Image',
                  log_filename = 'LogFilename',):
    set_folder(igor, file_loc)
    os.chdir(file_loc)
    
    with open (log_filename+'.txt', 'a') as f: 
        note = 'F'
        data = igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Wave("MasterVariablesWave")
        dim = data.GetDimensions()[1]
        gmv = {}
        for i in range(dim):
            gmv[data.DimensionLabel(0,i,0)] = data.GetNumericWavePointValue(i)
        ex(igor, "TriggerChannelPopup_1", "MasterPanel", 0, "DeflVolts")
        ex(igor, "TriggerPointSetVar_1", "MasterPanel", gmv["DeflectionSetpointVolts"])
        igor.Execute('root:Packages:MFP3D:Main:Variables:BaseName = "'+note+base_filename+'"')
        igor.Execute('PV("BaseSuffix", 0000)')
        igor.Execute('ARCheckSuffix()')
        fname = get_next_filename(note+base_filename, file_loc, '.ibw')
        saveprint('Doing Force Curve:', f)
        saveprint('   ' + fname, f)
        ex(igor, "SingleForce_1", "MasterPanel", 1)
        time.sleep(5)
        file = binarywave.load(fname)
        z = file['wave']['wData'][:,4]
        defl = file['wave']['wData'][:,1]
        max_i = np.argmax(defl)
        engage_i = np.argmin(defl[:max_i])
        max_i = np.argmax(defl)
        
        curr_invols = float(str(binarywave.load(fname)['wave']['note']).split('rInvOLS: ')[1].split('\\')[0])
        deflV = defl / curr_invols
        
        diff_i = max_i - engage_i
        i_1 = int(engage_i + 0.25*diff_i)
        i_2 = int(max_i - 0.25*diff_i)
        
        invols = (z[i_2]-z[i_1])/(deflV[i_2]-deflV[i_1])
        
        saveprint('Nominal InvOLS:', f)
        saveprint('   ' + str(invols), f)
        if (invols < 10e-7) and (invols > 4e-7):
            ex(igor, 'InvOLSSetVar_1', 'MasterPanel', invols)
    return invols
            
def line_fit(line, order=1, box=[0]):
    """
    Do a nth order polynomial line flattening

    Parameters
    ----------
    line : 1d array-like
    order : integer

    Returns
    -------
    result : 1d array-like
        same shape as data
    """
    if order < 0:
        raise ValueError('expected deg >= 0')
    newline = line
    if len(box) == 2:
        newline = line[box[0]:box[1]]
    x = np.arange(len(newline))
    k = np.isfinite((newline))
    if not np.isfinite(newline).any():
        return line
    coefficients = np.polyfit(x[k], newline[k], order)
    return line - np.polyval(coefficients, np.arange(len(line)))


def line_flatten_image(data, order=1, axis=0, box=[0]):
    """
    Do a line flattening

    Parameters
    ----------
    data : 2d array
    order : integer
    axis : integer
        axis perpendicular to lines

    Returns
    -------
    result : array-like
        same shape as data
    """

    if axis == 1:
        data = data.T

    ndata = np.zeros_like(data)

    for i, line in enumerate(data):
        ndata[i, :] = line_fit(line, order, box)

    if axis == 1:
        ndata = ndata.T

    return ndata

def compute_piezoresponse(amplitude_data, phase_data):
    """
    Compute the piezoresponse using the formula: Piezoresponse = Amplitude * cos(Phase)
    
    Args:
        amplitude_data (np.array): The amplitude data.
        phase_data (np.array): The phase data (in degrees).
        
    Returns:
        np.array: The calculated piezoresponse.
    """
    # Convert phase from degrees to radians
    phase_radians = np.radians(phase_data)
    
    # Compute piezoresponse
    piezoresponse_data = amplitude_data * np.cos(phase_radians)
    
    return piezoresponse_data


def correct_label(label):
    label = [x for x in label if x]  # Remove the empty lists
    label = label[0]  # Remove the unnecessary inception

    corrected_label = []

    for i in label:
        i = i.decode('UTF-8')
        if len(i) == 0:  # Skip empty channel names
            pass
        else:  # Correct the duplicate letters
            if 'Trace' in i:
                i = i.split('Trace')[0]
                corrected_label.append(i + 'Trace')
            elif 'Retrace' in i:
                i = i.split('Retrace')[0]
                corrected_label.append(i + 'Retrace')
            else:
                corrected_label.append(i)
    corrected_label = [x.encode() for x in corrected_label]
    return corrected_label



def get_notes_dict(filename):
    notes = binarywave.load(filename)['wave']['note']
    notes_split = notes.split(b'\r')

    # Convert the byte strings to regular strings and split into key-value pairs
    notes_dict = {}

    for item in notes_split:
        try:
            # Try decoding with utf-8
            decoded = item.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to latin1 (ISO-8859-1)
            decoded = item.decode('latin1')
        # Split into key and value if possible
        if ':' in decoded:
            key, value = decoded.split(':', 1)
            notes_dict[key.strip()] = value.strip()
    return notes_dict

def copy_scan_location_from_notes_dict(igor, notes_dict):
    ex(igor, 'ScanSizeSetVar_0', 'MasterPanel', val = notes_dict['ScanSize'])
    ex2(igor, 'ScanAngleSetVar_0', val = notes_dict['ScanAngle'])
    ex2(igor, 'XOffsetSetVar_0', val = notes_dict['XOffset'])
    ex2(igor, 'YOffsetSetVar_0', val = notes_dict['YOffset'])
    ex2(igor, 'FastRatioSetVar_0', val = notes_dict['FastRatio'])
    ex2(igor, 'SlowRatioSetVar_0', val = notes_dict['SlowRatio'])
    ex(igor, 'PointsLinesSetVar_0', 'MasterPanel', val = notes_dict['PointsLines'])

def get_gmv(igor):
    data = igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Wave("MasterVariablesWave")
    #if above doesn't work, go to: Programming -> Global Variables -> Master, Right Click a variable, Browser MasterVariablesWave, copy Data Folder in full
    dim = data.GetDimensions()[1]
    gmv = {}
    for i in range(dim):
        gmv[data.DimensionLabel(0,i,0)] = data.GetNumericWavePointValue(i)
    return gmv

def rotate_frame(x, y, angle_degrees):
    a = np.radians(angle_degrees)
    x_rot = x*np.cos(a)-y*np.sin(a)
    y_rot = x*np.sin(a)+y*np.cos(a)
    return x_rot, y_rot
	

def rotate_clockwise(x, y, angle_degrees):
    a = np.radians(angle_degrees)
    x_rot = x*np.cos(a)+y*np.sin(a)
    y_rot = -x*np.sin(a)+y*np.cos(a)
    return x_rot, y_rot
    
def change_offset_to_xy_point(igor, gmv, x_pos, y_pos):
    rot_x, rot_y = rotate_clockwise(x_pos, y_pos, gmv["ScanAngle"])
    new_afm_x = (rot_x * gmv["ScanSize"]/2)+gmv["XOffset"]
    new_afm_y = (rot_y * gmv["ScanSize"]/2)+gmv["YOffset"]
    print(new_afm_x)
    print(new_afm_y)
    ex2(igor, 'XOffsetSetVar_0', val = new_afm_x)
    ex2(igor, 'YOffsetSetVar_0', val = new_afm_y)
    
    

def scale_xy_points(gmv, x_pos, y_pos):
    #converts to internal values
    gmv["GridXLocMat"] = (gmv["ScanSize"]*x_pos)/(2*gmv["FastRatio"]) #height ratio
    gmv["GridYLocMat"] = (gmv["ScanSize"]*y_pos)/(2*gmv["SlowRatio"]) #width ratio

    theta = - gmv["ScanAngle"]*np.pi/180

    R1 = np.array([gmv["GridXLocMat"], gmv["GridYLocMat"]])
    R2 = np.array([[np.cos(theta), np.sin(theta)],
                 [-np.sin(theta), np.cos(theta)]])
    GridXLocMatRot, GridYLocMatRot = np.dot(R1,R2)
  
    
    XLocVMat = ((GridXLocMatRot + gmv["XOffset"])/gmv["XLVDTSens"])+gmv["XLVDTOffset"]
    YLocVMat = ((GridYLocMatRot + gmv["YOffset"])/gmv["YLVDTSens"])+gmv["YLVDTOffset"]
    
    return XLocVMat, YLocVMat

def go_to_xy_point(igor, gmv, x_pos, y_pos, transit_time, wait=True):
    #moves tip from values
    XLocVMat, YLocVMat = scale_xy_points(gmv, x_pos, y_pos)
    final_string = (f'td_SetRamp({transit_time:.6f},"$outputXloop.Setpoint",0,{XLocVMat:.6f},"$outputYloop.Setpoint",0,{YLocVMat:.6f},"",0,0,"")')
    igor.Execute(final_string)
    if wait:
        time.sleep(transit_time)
    return XLocVMat, YLocVMat


def initialise_tip_motion(igor):
    #must be run once to allow once per boot to allow tip to move
    data = igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Wave("MasterVariablesWave")
    #if above doesn't work, go to: Programming -> Global Variables -> Master, Right Click a variable, Browser MasterVariablesWave, copy Data Folder in full
    dim = data.GetDimensions()[1]

    gmv = {}
    for i in range(dim):
        gmv[data.DimensionLabel(0,i,0)] = data.GetNumericWavePointValue(i)
    
    igor.Execute('print ARExecuteControl("GoForce_1","ARDoIVPanel",1,"")')
    igor.Execute('print ARExecuteControl("ShowXYSpotCheck_1", "ARDoIVPanel", 1, "")')
    return gmv

def stop_scan(igor, wait=False):
    ex(igor, 'StopScan_0','MasterPanel')

def ex2(igor, variable = "", val = 0, string = "", name = ""):
    execution_line = 'ARSetVarFunc("'
    execution_line += variable
    execution_line += '", '
    execution_line += str(val)
    execution_line += ', "'
    execution_line += string
    execution_line += '", "'
    execution_line += name
    execution_line += '")'
    igor.Execute(execution_line)
    return execution_line

def save_3DPFM_as_hdf5(json_filename, hdf_filename,
                                     positions = 'Experimental',
                                     x0 = -8.9e-6, ):   # Multi75
   
    if os.path.exists(hdf_filename+'.hf5'):
        os.remove(hdf_filename+'.hf5')
    with open(json_filename+'.json', "r") as f:
        content = f.read()
   
    # Extract each [...] block
    blocks = content[1:-1].split('][')
   
    # Merge all dicts into one
    final_dict = {}
    for block in blocks:
        obj = json.loads(block)
        final_dict.update(obj)
   
    save_name = hdf_filename+'.hf5'
   
    i0 = 0
    i1 = -1
    j0 = 0
    j1 = -1
   
    note_list = []
    for key in final_dict.keys():
        if key[1:] == '_Filename':
            note_list.append(key[0])
    if 'x0' in final_dict:
        x0 = final_dict['x0']
        
    if ('A' in note_list) and ('X' in note_list) and ('Y' in note_list):
        Nth = 'A'
        Sth = 'Y'
        Est = 'A'
        Wst = 'X'
    elif ('N' in note_list) and ('S' in note_list) and ('E' in note_list) and ('W' in note_list):
        Nth = 'N'
        Sth = 'S'
        Est = 'E'
        Wst = 'W'
    elif any(d not in note_list for d in ['N', 'S', 'E', 'W']):
        Nth = 'N' if 'N' in note_list else 'C'
        Sth = 'S' if 'S' in note_list else 'C'
        Est = 'E' if 'E' in note_list else 'C'
        Wst = 'W' if 'W' in note_list else 'C'
        
    priority = ['B', 'C', 'A', 'E']
    Cnt = next((x for x in priority if x in note_list), None)
   
    all_topos = []
    all_amps = []
    first = True
    for note in note_list:
        data = SciFiReaders.IgorIBWReader(final_dict[note+'_Filename']).read()
        topo = data['Channel_000']
        if first:
            orig_data = data.copy()
            summed_response = []
            amplitude_data = np.array(data['Channel_001'][i0:i1, j0:j1])
            phase_data = np.array(data['Channel_003'][i0:i1, j0:j1])
            for angle in range(360):            
                piezoresponse_data = compute_piezoresponse(amplitude_data, phase_data-angle)
                summed_response.append(np.sum(np.abs(piezoresponse_data)))
            phi0 = np.argmax(summed_response)
            first = False
            orig_data = data.copy()
        all_topos.append(data['Channel_000'])
   
    offsets = []
    for n in range (len(all_topos)):
        i, j = phase_cross_correlation(np.array(all_topos[0][i0:i1, j0:j1]), np.array(all_topos[n][i0:i1, j0:j1]), normalization = None)[0]
        offsets.append([i,j])
    offsets = np.array(offsets).astype(int)
    i0 = int(np.max(offsets[:,0]))
    j0 = int(np.max(offsets[:,1]))
    i1 = int(np.min(offsets[:,0]))
    j1 = int(np.min(offsets[:,1]))
    if i1 >=0:
        i1 = -1
    if j1 >=0:
        j1 = -1
    if i0 < 0:
        i0 = 0
    if j0 < 0:
        j0 = 0
   
    t = 0
    with h5py.File(save_name, 'a') as hf:
        for note in note_list:
            data = SciFiReaders.IgorIBWReader(final_dict[note+'_Filename']).read()
            amplitude_data = np.roll(np.roll(np.array(data['Channel_001']), offsets[t][0], axis=0), offsets[t][1],axis=1)[i0:i1, j0:j1]
            phase_data = np.roll(np.roll(np.array(data['Channel_003']), offsets[t][0], axis=0), offsets[t][1],axis=1)[i0:i1, j0:j1]-phi0
            piezoresponse_data = compute_piezoresponse_c(amplitude_data, phase_data)

            if note == Nth:
                PRN = piezoresponse_data
            if note == Sth:
                PRS = piezoresponse_data
            if note == Est:
                PRE = piezoresponse_data
            if note == Wst:
                PRW = piezoresponse_data
            for key in data.keys():
                hf.require_group(note+'/'+key)
                pyNSID.hdf_io.write_nsid_dataset(data[key], hf[note+'/'+key], main_data_name=key)
            t+=1
   
    h = float(final_dict['h'])
    r0 = 2/3
   
    try:
        L = float(final_dict['tip_length'])
    except:
        L = 225e-6
    try:
        W = float(final_dict['tip_width'])
    except:
        W = 28e-6
    try:
        x_factor = abs(float(final_dict['diamond_length'])/float(final_dict['diamond_width']))
    except:
        x_factor = 1

    if positions == 'Experimental':
        xW = (W/2)*float(final_dict[Wst+'_RealLocation_x'])*x_factor
        xE = (W/2)*float(final_dict[Est+'_RealLocation_x'])*x_factor
        yN = (W/2)*float(final_dict[Nth+'_RealLocation_y'])
        yS = (W/2)*float(final_dict[Sth+'_RealLocation_y'])

    else:
        xW = (W/2)*float(final_dict[Wst+'_Location_x'])*x_factor
        xE = (W/2)*float(final_dict[Est+'_Location_x'])*x_factor
        yN = (W/2)*float(final_dict[Nth+'_Location_y'])
        yS = (W/2)*float(final_dict[Sth+'_Location_y'])
    
    #ux = (((r0*L-xW)*PRE - (r0*L+xE)*PRW)
    #      /(GE*(r0*L-xW)+GW*(r0*L-xE)))
   
   
    uzE = PRE - (xE+x0)*(xE-xW)*(PRE-PRW)/h**2
    uzW = PRW - (xW+x0)*(xE-xW)*(PRE-PRW)/h**2
    uzN = PRN - ((x0)*(xE-xW)*(PRE-PRW)/h**2) - yN*(PRN - PRS)/(yN - yS)
    uzS = PRS - ((x0)*(xE-xW)*(PRE-PRW)/h**2) - yS*(PRN - PRS)/(yN - yS)
   
    for uz in [uzE, uzW, uzN, uzS]:
        plt.imshow(np.real(uz))
        plt.colorbar()
        plt.show()
        plt.close()
   
    uz = np.mean([uzE, uzW, uzN, uzS], axis = 0)
   
    uy = h*(PRN-PRS)/(yN-yS)
    ux = (h*(PRE-PRW)/(xE-xW))-(h*uz/(r0*L))
   
   
    U_mag = np.sqrt(np.abs(ux)**2+np.abs(uy)**2+np.abs(uz)**2)
    U_mag_max = min(U_mag.max(), 3e-9)
   
    name = ['ux', 'uy', 'uz']
    data = [ux, uy, uz]
   
    with h5py.File(save_name, 'a') as hf:
        hf.require_group('3DPFM')
        for i in range(3):
            hf.create_dataset('3DPFM/'+name[i], data=data[i])
        for key in final_dict:
            val = final_dict[key]
            if type(val) == str:
                try:
                    val = float(val)
                except:
                    pass
            hf['3DPFM'].attrs[key] = val
            hf['3DPFM'].attrs['offsets'] = offsets
            hf['3DPFM'].attrs['imported_from_json'] = True

def reduce_frequency(igor, log_filename, mod=100000):
    with open (log_filename+'.txt', 'a') as f: 
        gmv = get_gmv(igor)
        saveprint('Current Frequency:', f)
        saveprint('   '+str(gmv['DriveFrequency']), f)
        reduced_freq = gmv['DriveFrequency']%mod
        if reduced_freq < 20000:
            reduced_freq += 20000
        ex(igor, "DriveFrequencySetVar_0", "MasterPanel", reduced_freq)
        saveprint('Reduced Frequency:', f)
        saveprint('   '+str(reduced_freq), f)

def set_blind_spot_x(igor,
                     T,
                     file_loc,
                     base_filename,
                     log_filename,
                     closed_LD = True):


    igor.Execute('arpxl_WriteValue("zoom",4)')
    pos = find_curr_position_on_diamond_optically(igor,
                                            file_loc,
                                            base_filename,
                                            log_filename,
                                            verbose = True)
    final_x = pos[0]
                
    json_filename = base_filename
    with open(json_filename+'.json', "r") as f:
        content = f.read()
    
    # Extract each [...] block
    blocks = content[1:-1].split('][')
    
    # Merge all dicts into one
    final_dict = {}
    for block in blocks:
        obj = json.loads(block)
        final_dict.update(obj)
    W = float(final_dict['tip_width'])
    x_factor = abs(float(final_dict['diamond_length'])/float(final_dict['diamond_width']))
    final_distance = final_x * x_factor * W/2

    x0 = -final_distance
    
    with open (log_filename+'.txt', 'a') as f: 
        gmv = get_gmv(igor)
        saveprint('x0:', f)
        saveprint('   '+str(x0), f)

    with open(base_filename+".json", "a") as f:
        json.dump([{"x0": x0,
                    "x_b": final_x,
                    "blindspot_to_distal": -(x0 + (x_factor * W/2))
                   }], f)
    
    return x0