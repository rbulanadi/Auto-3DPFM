import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from igor import binarywave
import h5py
import re
import json
import SciFiReaders
import pyNSID

# Imaging and arrays for overlay
try:
    from PIL import Image, ImageTk, ImageDraw, ImageOps
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    Image = ImageTk = ImageDraw = None # type: ignore


try:
    import numpy as np
    NP_AVAILABLE = True
except Exception:
    NP_AVAILABLE = False


# Optional drag & drop support via tkinterdnd2
# If the package is not available, the app will fall back to click-to-browse.
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except Exception:
    DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = tk.Tk # type: ignore



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

def norm (data):
    data -= np.min(data)
    data /= np.max(data)
    data = data * 255
    return data

# --- Replace this stub with your real implementation ---
# Must return a NumPy array (H x W) or (H x W x C) with values 0..255
# We'll convert it to a PIL image inside the app.
def extract_topography_from_ibw(filename):
    file = binarywave.load(filename)
    dataset = file['wave']['wData']
    scale =  float(str(file['wave']['note']).split('FastScanSize:')[-1].split('\\r')[0])/dataset[:, :, 0].T.shape[0]
    scan_size = float(str(file['wave']['note']).split('FastScanSize:')[-1].split('\\r')[0])
    desc = str(file['wave']['note']).split('ImageNote:')[-1].split('\\r')[0]
    data = (np.flipud(dataset[:,:,0].T))
    data = norm(line_flatten_image(data))
    return data

DOT_RADIUS = 12
DOT_FILL = '#e63946'  # red-ish
DOT_FILL_LOADED = '#2a9d8f'  # green when a file is attached
DOT_OUTLINE = 'white'

# Fixed drawing size (in canvas pixels)
SHAPE_W = 520
SHAPE_H = 320

# Size tweaks: smaller left sketch, larger overlay preview
LEFT_SHAPE_W = int(SHAPE_W * 0.85)
LEFT_SHAPE_H = int(SHAPE_H * 0.85)
OVERLAY_W = int(SHAPE_W * 1.4)
OVERLAY_H = int(SHAPE_H * 1.4)


class DotDrop:
    """A small circular drop/click target that accepts a file and shows its name."""
    def __init__(self, app, cx, cy, idx):
        self.app = app
        self.cx, self.cy = cx, cy
        self.idx = idx
        self.filepath = None

        # draw the dot on the canvas
        r = DOT_RADIUS
        self.dot_id = self.app.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                                  fill=DOT_FILL, outline=DOT_OUTLINE, width=2)
        # a subtle label that will show the filename (base name only)
        self.label_id = self.app.canvas.create_text(cx, cy + r + 14, text='Drop or click',
                                                    fill='#dddddd', font=('Segoe UI', 9))

        # bind mouse click to open file dialog
        self.app.canvas.tag_bind(self.dot_id, '<Button-1>', self.on_click)
        self.app.canvas.tag_bind(self.label_id, '<Button-1>', self.on_click)

        if DND_AVAILABLE:
            # Create an invisible rectangle around the dot as a generous drop target
            pad = 24
            self.drop_id = self.app.canvas.create_rectangle(cx - pad, cy - pad, cx + pad, cy + pad,
                                                            outline='', fill='')
            self.app.canvas.tag_bind(self.drop_id, '<<Drop>>', self.on_drop)
            self.app.canvas.dnd_bind('<<Drop>>', self.on_drop, target=self.drop_id)

    def on_click(self, _evt=None):
        filetypes = [
            ('All files', '*.*'),
        ]
        path = filedialog.askopenfilename(title='Choose a file for dot #{:d}'.format(self.idx),
                                          filetypes=filetypes)
        if path:
            self.set_file(path)

    def on_drop(self, evt):
        # tkinterdnd2 delivers a string that may contain braces for spaces
        raw = evt.data
        if not raw:
            return
        # Take the first file only
        path = raw
        if raw.startswith('{') and raw.endswith('}'):
            path = raw[1:-1]
        if ' ' in raw and not os.path.exists(path):
            # Multiple files were dropped. Split on braces-aware pattern.
            # Simplest approach: take until the first closing brace or space.
            path = raw.split('}')[0].strip('{}') if '}' in raw else raw.split(' ')[0]
        if os.path.exists(path):
            self.set_file(path)
        else:
            messagebox.showerror('Drop error', 'Could not read dropped file:\n{}'.format(raw))

    def set_file(self, path):
        self.filepath = path
        base = os.path.basename(path)
        self.app.canvas.itemconfigure(self.dot_id, fill=DOT_FILL_LOADED)
        self.app.canvas.itemconfigure(self.label_id, text=base, fill='#cfe8e6')
        # Load/update topography for overlay (convert numpy -> PIL inside)
        self.app.load_topography(self.idx, path)
        self.app.update_status()

    def as_dict(self):
        return {
            'index': self.idx,
            'file': self.filepath,
            'x': self.cx,
            'y': self.cy,
        }


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Five File Dots')
        self.root.geometry('760x560')
        self.root.minsize(620, 460)
        self.root.configure(bg='#0f172a')  # slate-900

        # Top bar
        self.header = tk.Frame(root, bg='#0f172a')
        self.header.pack(fill=tk.X, padx=14, pady=(14, 6))

        title = tk.Label(self.header, text='Attach files to the five dots', fg='white', bg='#0f172a',
                         font=('Segoe UI Semibold', 14))
        title.pack(side=tk.LEFT)

        # Buttons on the right side of the header
        self.save_btn = tk.Button(self.header, text='Show Attached', command=self.show_attached)
        self.save_btn.pack(side=tk.RIGHT, padx=(6,0))
        self.save_out_btn = tk.Button(self.header, text='Save Output as HDF5', command=self.on_click_save_output)
        self.save_out_btn.pack(side=tk.RIGHT, padx=(6,0))
        self.load_json_btn = tk.Button(self.header, text='Load JSON', command=self.on_click_load_json)
        self.load_json_btn.pack(side=tk.RIGHT, padx=(6,0))

        # Main split: left drawing canvas, right overlay & controls
        body = tk.Frame(root, bg='#0f172a')
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # Left: drawing canvas
        self.canvas = tk.Canvas(body, bg='#0b1220', highlightthickness=0,
                               width=int(LEFT_SHAPE_W * 1.1), height=int(LEFT_SHAPE_H * 1.1))
        self.canvas.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        # Right: two columns — controls (dx/dy) and overlay + color controls
        self.right = tk.Frame(body, bg='#0f172a')
        self.right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Overlay column (rightmost)
        self.view_col = tk.Frame(self.right, bg='#0f172a')
        #self.view_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        # was: side=tk.RIGHT, fill=tk.Y, padx=(8, 0)
        self.view_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        
        view_title = tk.Label(self.view_col, text='Overlay preview', fg='white', bg='#0f172a', font=('Segoe UI', 12))
        view_title.pack(anchor='w', pady=(0, 6))

        self.overlay_w, self.overlay_h = OVERLAY_W, OVERLAY_H
        self.overlay_canvas = tk.Canvas(self.view_col, width=self.overlay_w, height=self.overlay_h,
                                        bg='#0b1220', highlightthickness=1, highlightbackground='#334155')
        self.overlay_canvas.pack(pady=(0, 8))

        # Per-layer opacity sliders are in the 'Layer colors' section below

        # Per-layer color pickers
        self.dot_names = {1: 'Top', 2: 'Right', 3: 'Bottom', 4: 'Left', 5: 'Center'}
        default_colors = {1: '#ff5555', 2: '#55aaff', 3: '#55ff55', 4: '#ffaa55', 5: '#ff55ff'}
        self.color_vars, self.color_btns = {}, {}
        self.alpha_vars = {}
        colors_frame = tk.LabelFrame(self.view_col, text='Layer colors', fg='white', bg='#0f172a', bd=1, highlightthickness=0, labelanchor='n')
        colors_frame.pack(fill=tk.X, pady=(0, 6))
        for idx in range(1, 6):
            row = tk.Frame(colors_frame, bg='#0f172a')
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=self.dot_names[idx], fg='white', bg='#0f172a', width=8, anchor='w').pack(side=tk.LEFT)
            self.color_vars[idx] = tk.StringVar(value=default_colors[idx])
            btn = tk.Button(row, text=default_colors[idx], command=lambda j=idx: self.choose_color(j))
            btn.pack(side=tk.LEFT)
            # swatch
            sw = tk.Label(row, width=2, relief='groove')
            sw.pack(side=tk.LEFT, padx=6)
            sw.configure(bg=default_colors[idx])
            self.color_btns[idx] = (btn, sw)
            # per-layer opacity slider (independent)
            a = tk.IntVar(value=100)
            self.alpha_vars[idx] = a
            op = tk.Scale(row, from_=0, to=100, orient='horizontal', variable=a,
                          command=lambda _=None: self.render_overlay(), length=140, label='opacity (\%)')
            op.pack(side=tk.RIGHT, padx=4)

        # Controls column (dx/dy sliders)
        self.ctrl_col = tk.Frame(self.right, bg='#0f172a')
        self.ctrl_col.pack(side=tk.LEFT, fill=tk.Y)
        self.dx_vars, self.dy_vars = {}, {}
        labels = ['Top', 'Right', 'Bottom', 'Left', 'Center']
        for i, name in enumerate(labels, start=1):
            frame = tk.LabelFrame(self.ctrl_col, text=name, fg='white', bg='#0f172a', bd=1, highlightthickness=0, labelanchor='n')
            frame.pack(fill=tk.X, pady=4)
            dx = tk.IntVar(value=0)
            dy = tk.IntVar(value=0)
            self.dx_vars[i] = dx
            self.dy_vars[i] = dy
            sdx = tk.Scale(frame, from_=-30, to=30, orient='horizontal', variable=dx,
                           command=lambda _=None: self.render_overlay(), length=180, label='dx')
            sdy = tk.Scale(frame, from_=-30, to=30, orient='horizontal', variable=dy,
                           command=lambda _=None: self.render_overlay(), length=180, label='dy')
            sdx.pack(fill=tk.X)
            sdy.pack(fill=tk.X)

        # Load background image resembling the provided sketch if it exists
        self.bg_image = None
        
        self.canvas.bind('<Configure>', self.redraw)

        # Storage for topography images per dot index (1..5)
        self.topo_images = {i: None for i in range(1, 6)}
        self.overlay_tk = None  # keep reference to PhotoImage used in overlay

        # Dots are created dynamically in redraw() to form a diamond + center.
        self.dots = []

        # status label
        self.status = tk.Label(root, text='Drop a file on each dot, then use dx/dy sliders to align the overlay.',
                               fg='#94a3b8', bg='#0f172a', anchor='w')
        self.status.pack(fill=tk.X, padx=14, pady=(0, 12))

    def redraw(self, _evt=None):
        self.canvas.delete('all')
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()

        # We now always draw a fixed-size vector shape (centered), so we can add content around it
        self.draw_vector_shape(w, h)

        # Create dots in a diamond configuration plus one in the center, placed near edges
        positions = self.compute_dot_positions(w, h)
        self.dots = []
        for i, (cx, cy) in enumerate(positions, start=1):
            self.dots.append(DotDrop(self, int(cx), int(cy), i))
        # refresh overlay when layout changes
        self.render_overlay()

    def draw_vector_shape(self, w, h):
        """Draw a fixed-size shape centered on the canvas with an S-shaped left edge.
        Stores a bbox we can reuse for dot placement.
        """
        cx, cy = w // 2, h // 2
        half_w = LEFT_SHAPE_W // 2
        half_h = LEFT_SHAPE_H // 2
        left = cx - half_w
        right_flat = cx + half_w - 130  # end of flat top/bottom before the point
        top = cy - half_h
        bottom = cy + half_h
        mid_y = cy
        point_x = cx + half_w + 20  # the rightward point

        # Top edge to point and back along bottom
        self.canvas.create_line(left, top, right_flat, top, width=8, fill='white')
        self.canvas.create_line(right_flat, top, point_x, mid_y, width=12, fill='white')
        self.canvas.create_line(point_x, mid_y, right_flat, bottom, width=12, fill='white')
        self.canvas.create_line(right_flat, bottom, left, bottom, width=8, fill='white')

        # Left S edge (two curves only) – implies continuation to the left
        # First curve bows left from top to mid, second bows right from mid to bottom
        s_points = [
            (left, top),
            (left - 70, top + 60),  # control 1 (bows left)
            (left, mid_y),          # mid join
            (left + 60, mid_y + 40),# control 2 (bows right)
            (left, bottom),
        ]
        self.canvas.create_line(*sum(s_points, ()), smooth=True, width=8, fill='white')

        # Save key geometry for later placement of dots
        self.shape_bbox = (left, top, right_flat, bottom, point_x, mid_y)

    def compute_dot_positions(self, w, h):
        """Return [(x,y), ...] for 4-dot diamond near edges + center, inside current shape."""
        left, top, right_flat, bottom, point_x, mid_y = self.shape_bbox
        center_x = (left + right_flat) / 2 + 100  # shift right for symmetry
        center_y = (top + bottom) / 2

        inset = 24  # how far inside each edge to place the diamond points

        top_pt = (center_x, top + inset)
        right_pt = (point_x - 40, mid_y)
        bottom_pt = (center_x, bottom - inset)
        left_pt = (center_x - (right_pt[0] - center_x), mid_y)  # mirror for symmetry
        center_pt = (center_x, center_y)
        return [top_pt, right_pt, bottom_pt, left_pt, center_pt]

    # ----- Overlay (right panel) -----
    def load_topography(self, dot_idx, filename):
        try:
            arr = extract_topography_from_ibw(filename)
            if not NP_AVAILABLE:
                raise RuntimeError('NumPy not available')
            # Convert numpy array to PIL image
            if arr.ndim == 2:
                img = Image.fromarray(arr.astype('uint8'), mode='L').convert('RGBA')
            elif arr.ndim == 3:
                # If single-channel last, squeeze; else assume already 3/4 channels 0..255
                if arr.shape[2] == 1:
                    img = Image.fromarray(arr[:, :, 0].astype('uint8'), mode='L').convert('RGBA')
                else:
                    img = Image.fromarray(arr.astype('uint8'), mode='RGBA' if arr.shape[2] == 4 else 'RGB').convert('RGBA')
            else:
                raise ValueError('Unsupported array shape for topography: %r' % (arr.shape,))
            # Fit to overlay box while preserving aspect
            img.thumbnail((self.overlay_w, self.overlay_h), Image.LANCZOS)
            self.topo_images[dot_idx] = img
            self.render_overlay()
        except Exception as e:
            messagebox.showerror('Topography error', f'Failed to load topography for dot {dot_idx}:{e}')

    def render_overlay(self):
        if not PIL_AVAILABLE:
            return
        base = Image.new('RGBA', (self.overlay_w, self.overlay_h), (0, 0, 0, 0))
        cx, cy = self.overlay_w // 2, self.overlay_h // 2

        for idx in range(1, 6):
            img = self.topo_images.get(idx)
            if img is None:
                continue
            # Positioning
            w, h = img.size
            dx = self.dx_vars.get(idx, tk.IntVar(value=0)).get() if idx in self.dx_vars else 0
            dy = self.dy_vars.get(idx, tk.IntVar(value=0)).get() if idx in self.dy_vars else 0
            x = cx - w // 2 + dx
            y = cy - h // 2 + dy

            # Per-layer color & opacity
            col_hex = self.color_vars.get(idx).get() if hasattr(self, 'color_vars') and idx in self.color_vars else '#ffffff'
            try:
                col = tuple(int(col_hex[i:i+2], 16) for i in (1, 3, 5))
            except Exception:
                col = (255, 255, 255)
            gray = img.convert('L')
            rgb = ImageOps.colorize(gray, black=(0, 0, 0), white=col)
            layer = rgb.convert('RGBA')
            # Independent per-layer alpha
            try:
                val = self.alpha_vars[idx].get()
                alpha_val = int(max(0, min(100, val)) * 255 / 100)
            except Exception:
                alpha_val = 255
            a = Image.new('L', (w, h), alpha_val)
            layer.putalpha(a)

            try:
                base.alpha_composite(layer, dest=(x, y))
            except Exception:
                base.paste(layer, (x, y), layer)

        self.overlay_tk = ImageTk.PhotoImage(base)
        self.overlay_canvas.delete('all')
        self.overlay_canvas.create_image(self.overlay_w // 2, self.overlay_h // 2, image=self.overlay_tk)

    def choose_color(self, idx):
        current = self.color_vars[idx].get()
        chosen = colorchooser.askcolor(color=current, title=f'Choose color for {self.dot_names[idx]}')
        if chosen and chosen[1]:
            hexc = chosen[1]
            self.color_vars[idx].set(hexc)
            btn, sw = self.color_btns[idx]
            try:
                btn.configure(text=hexc)
                sw.configure(bg=hexc)
            except Exception:
                pass
            self.render_overlay()

    def update_status(self):
        attached = [d for d in self.dots if d.filepath]
        if len(attached) == 5:
            self.status.config(text='All 5 files attached. Click "Show Attached" to review.')
        else:
            self.status.config(text=f'{len(attached)}/5 dots have files attached.')

    def show_attached(self):
        lines = []
        for d in self.dots:
            name = os.path.basename(d.filepath) if d.filepath else '—'
            lines.append(f'Dot {d.idx}: {name}')
        messagebox.showinfo('Attached Files', "".join(lines))



    # --- Stubs you can implement ---
    def load_sliders_from_json(self, filename):
        """Parse the custom JSON format you provided, prefill files and estimate dx/dy.
        The JSON file contains multiple concatenated one-item arrays like [ {..} ][ {..} ] ...
        We'll:
          1) Merge all blocks into a single dict (like your example).
          2) Load files for C/N/S/E/W into the corresponding dots.
          3) Estimate initial dx/dy via phase cross-correlation vs Center (C) using the
             *overlay-sized* grayscale images so pixel shifts map directly to slider units.
          4) Fallback: if any image missing, try using *_Location_x/y if present (scaled heuristically).
        """
        import re
        import json as _json
        import numpy as _np

        base_dir = os.path.dirname(filename)
        with open(filename, 'r') as f:
            content = f.read()

        # Extract [ ... ] blocks and merge
        blocks = re.findall(r"\[.*?\]", content, flags=re.S)
        final_dict = {}
        for block in blocks:
            try:
                obj = _json.loads(block)[0]
                if isinstance(obj, dict):
                    final_dict.update(obj)
            except Exception as e:
                print('Skipping malformed block:', e)

        # Map JSON cardinal letters to dot indices in this UI
        # 1:Top(N), 2:Right(E), 3:Bottom(S), 4:Left(W), 5:Center(C)
        card_to_idx = {'N': 1, 'E': 2, 'S': 3, 'W': 4, 'C': 5}

        # Prefill files from *_Filename keys
        for C in ['C', 'N', 'S', 'E', 'W']:
            key = f'{C}_Filename'
            if key in final_dict:
                path = final_dict[key]
                if not os.path.isabs(path):
                    path = os.path.normpath(os.path.join(base_dir, path))
                if os.path.exists(path):
                    idx = card_to_idx[C]
                    dot = next((d for d in self.dots if getattr(d, 'idx', None) == idx), None)
                    if dot is not None:
                        dot.set_file(path)
                else:
                    print(f'Warning: file not found for {C}: {path}')

        # Helper: PIL image -> grayscale numpy float32 (overlay-sized)
        def _pil_to_gray(img):
            return _np.asarray(img.convert('L'), dtype=_np.float32)

        # Phase correlation (integer shift) with FFT, returns (dy, dx)
        def _phase_shift(ref, mov):
            # center-crop to common size
            h = min(ref.shape[0], mov.shape[0])
            w = min(ref.shape[1], mov.shape[1])
            R = ref[:h, :w]
            M = mov[:h, :w]
            F1 = _np.fft.fft2(R)
            F2 = _np.fft.fft2(M)
            Rcross = F1 * _np.conj(F2)
            den = _np.abs(Rcross)
            den[den == 0] = 1e-12
            Rcross /= den
            r = _np.fft.ifft2(Rcross)
            r = _np.abs(r)
            peak = _np.unravel_index(_np.argmax(r), r.shape)
            shifts = _np.array(peak, dtype=_np.int64)
            # unwrap
            if shifts[0] > h // 2:
                shifts[0] -= h
            if shifts[1] > w // 2:
                shifts[1] -= w
            return int(shifts[0]), int(shifts[1])  # (dy, dx)

        # Estimate dx/dy vs Center, using overlay images already loaded by set_file()
        ref_idx = 5
        ref_img = self.topo_images.get(ref_idx)
        have_ref = ref_img is not None
        if have_ref:
            ref = _pil_to_gray(ref_img)

        for C, idx in card_to_idx.items():
            if idx == ref_idx:
                continue
            img = self.topo_images.get(idx)
            if img is None:
                continue
            if have_ref:
                try:
                    from skimage.registration import phase_cross_correlation
                    dy, dx = phase_cross_correlation(ref, _pil_to_gray(img))[0]
                    # To align moving image to reference, move by negative shift
                    if idx in self.dx_vars:
                        self.dx_vars[idx].set(int(-dx))
                    if idx in self.dy_vars:
                        self.dy_vars[idx].set(int(-dy))
                except Exception as e:
                    print(f'Phase-correlation failed for {C}:', e)
        '''
        # Fallback / optional seed from *_Location_x/y if present (scaled heuristically)
        # Only touch sliders that are still zero after the correlation step.
        for C, idx in card_to_idx.items():
            if idx == ref_idx:
                continue
            keyx, keyy = f'{C}_Location_x', f'{C}_Location_y'
            if keyx in final_dict and keyy in final_dict:
                try:
                    dx_guess = float(final_dict[keyx])
                    dy_guess = float(final_dict[keyy])
                    # Scale guesses to pixels (heuristic gain)
                    gain = min(self.overlay_w, self.overlay_h) * 0.25
                    if idx in self.dx_vars and self.dx_vars[idx].get() == 0:
                        self.dx_vars[idx].set(int(round(dx_guess * gain)))
                    if idx in self.dy_vars and self.dy_vars[idx].get() == 0:
                        self.dy_vars[idx].set(int(round(dy_guess * gain)))
                except Exception:
                    pass
        '''
        self.render_overlay()
        self.update_status()
        self.final_dict = final_dict
        self.json_base_dir = base_dir



    def save_output_to_file(self, save_name):

        
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
        final_dict = getattr(self, "final_dict", None)
        base = getattr(self, "json_base_dir", "")

        # ---- Build offsets from current sliders (order: C, N, S, E, W) ----
        note_order = ['C', 'N', 'S', 'E', 'W']
        idx_map = {'N': 1, 'E': 2, 'S': 3, 'W': 4, 'C': 5}  # dot index in your UI
        
        offsets = []
        for note in note_order:
            idx = idx_map[note]
            idy_val = int(self.dy_vars[idx].get())  # rows (axis=0)
            idx_val = int(self.dx_vars[idx].get())  # cols (axis=1)
            offsets.append([idy_val, idx_val])
        
        offsets = np.array(offsets, dtype=int)  # shape (5, 2) as [ [idy, idx], ... ]

        
        t = 0
        note_list = ['C', 'N', 'S', 'E', 'W']
            
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

            
        with h5py.File(save_name, 'a') as hf:
            first = True
            for note in note_list:
                        
                if final_dict is not None and f"{note}_Filename" in final_dict:
                    p = final_dict[f"{note}_Filename"]
                    path = os.path.normpath(os.path.join(base, p)) if not os.path.isabs(p) else p
                else:
                    
                    idx_map = {'N': 1, 'E': 2, 'S': 3, 'W': 4, 'C': 5}  # UI dot indices
                    # fallback to the file attached to the corresponding dot
                    dot_idx = idx_map[note]
                    dot = next((d for d in self.dots if getattr(d, "idx", None) == dot_idx and d.filepath), None)
                    if dot is not None:
                        path = dot.filepath

                

                data = SciFiReaders.IgorIBWReader(path).read()
                if first:
                    orig_data = data.copy()
                    summed_response = [] 
                    amplitude_data = data['Channel_001'][i0:i1, j0:j1]
                    phase_data = data['Channel_003'][i0:i1, j0:j1]
                    for angle in range(360):            
                        piezoresponse_data = compute_piezoresponse(amplitude_data, phase_data-angle)
                        summed_response.append(np.sum(np.abs(piezoresponse_data)))
                    phi0 = np.argmax(summed_response)
                    first = False
                amplitude_data = np.roll(np.roll(np.array(data['Channel_001']), offsets[t][0], axis=0), offsets[t][1],axis=1)[i0:i1, j0:j1]
                phase_data = np.roll(np.roll(np.array(data['Channel_003']), offsets[t][0], axis=0), offsets[t][1],axis=1)[i0:i1, j0:j1]-phi0
                piezoresponse_data = compute_piezoresponse_c(amplitude_data, phase_data)
            
                if note == 'N':
                    PRN = piezoresponse_data
                if note == 'S':
                    PRS = piezoresponse_data
                if note == 'E':
                    PRE = piezoresponse_data
                if note == 'W':
                    PRW = piezoresponse_data
                for key in data.keys():
                    hf.require_group(note+'/'+key)
                    pyNSID.hdf_io.write_nsid_dataset(data[key], hf[note+'/'+key], main_data_name=key)
                t+=1
        
        try:
            h = float(final_dict['h'])
        except:
            h = 1e-5
        r0 = 2/3
        
        try:
            L = float(final_dict['tip_length'])
        except:
            L = 240e-6
        try:
            W = float(final_dict['tip_width'])
        except:
            W = 25e-6
        try:
            x_factor = abs(float(final_dict['diamond_length'])/float(final_dict['diamond_width']))
        except:
            x_factor = 1

        try:
            xW = (W/2)*-float(final_dict['W_Location_x'])*x_factor
        except:
            xW = 15e-6
        try:
            xE = (W/2)*float(final_dict['E_Location_x'])*x_factor
        except:
            xE = 15e-6
        GE = xE/h
        GW = xW/h

        try:
            yN = (W/2)*float(final_dict['N_Location_y'])
        except:
            yN = 15e-6
        try:
            yS = (W/2)*-float(final_dict['S_Location_y'])
        except:
            yS = 15e-6
            
        GN = yN/h
        GS = yS/h
        
        ux = (((r0*L-xW)*PRE - (r0*L+xE)*PRW)
              /(GE*(r0*L-xW)+GW*(r0*L-xE)))
        
        uy = ((PRN-PRS)
              /(GN+GS))
        
        uzx = ((r0*L)*(GW*PRE + GE*PRW)
              /(GE*(r0*L-xW)+GW*(r0*L-xE)))
        
        uzy = ((GS*PRN+GN*PRS)
              /(GN+GS))
        
        uz = np.mean([uzx, uzy], axis = 0)
        
        U_mag = np.sqrt(np.abs(ux)**2+np.abs(uy)**2+np.abs(uz)**2)
        U_mag_max = min(U_mag.max(), 3e-9)
        
        
        name = ['ux', 'uy', 'uz']
        data = [ux, uy, uz]
        
        with h5py.File(save_name, 'a') as hf:
            hf.require_group('3DPFM')
            for i in range(3):
                hf.create_dataset('3DPFM/'+name[i], data=data[i])
            try:
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
            except:
                hf['3DPFM'].attrs['imported_from_json'] = False
                hf['3DPFM'].attrs['offsets'] = offsets
            
                

    # --- UI handlers to pick files and invoke the stubs ---
    def on_click_load_json(self):
        try:
            fn = filedialog.askopenfilename(title='Load alignment JSON', filetypes=[('JSON files', '*.json'), ('All files', '*.*')])
            if fn:
                self.load_sliders_from_json(fn)
        except Exception as e:
            messagebox.showerror('Load JSON error', str(e))

    def on_click_save_output(self):
        try:
            fn = filedialog.asksaveasfilename(title='Save HF5 File', defaultextension='.hf5',
                                              filetypes=[('HF5 File', '*.hf5'), ('All files', '*.*')])
            if fn:
                self.save_output_to_file(fn)
        except Exception as e:
            messagebox.showerror('Save output error', str(e))


def main():
    root = TkinterDnD() if DND_AVAILABLE else tk.Tk()
    if DND_AVAILABLE:
        root.drop_target_register(DND_FILES)

    app = App(root)  # build UI first

    # --- Maximize the window ---
    if sys.platform.startswith('win'):
        # Windows: true maximize
        root.state('zoomed')
    elif sys.platform.startswith('linux'):
        # Many Linux window managers honor this attribute
        try:
            root.attributes('-zoomed', True)
        except tk.TclError:
            # Fallback to filling the screen
            root.update_idletasks()
            root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    elif sys.platform == 'darwin':
        # macOS has no "zoomed" state; fake it by sizing to the screen
        root.update_idletasks()
        root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
        # (Optional) If you *want* true fullscreen instead:
        # root.attributes('-fullscreen', True)  # Esc to exit
        # root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))
    # ---------------------------

    root.mainloop()



if __name__ == '__main__':
    main()
