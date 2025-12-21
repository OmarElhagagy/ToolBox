import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import time
import os
# from video_capture import VideoCap # Uncomment if you have this file
from image_capture import ImageCap

# Create folder directory to save images
folder = r"\images"
cwd = os.getcwd()
path = cwd + folder
if not os.path.exists(path):
    os.makedirs(path)

# Extended filter list - Added OCR filters here
fil = ['color', 'gray', 'threshold', 'adaptive_threshold', 'otsu_threshold', 'increaseContrast', 'decreaseContrast', 
       'contrast_stretch', 'logTransformation', 'powerLowEnhancement', 'gamma_0_5', 'gamma_1_5', 'negativeEnhancement', 
       'gauss', 'sobel', 'laplace', 'canny', 'min', 'max', 'median', 'average', 'unsharp', 'prewitt', 
       'histogramEqualization', 'adaptive_hist_eq', 'sharpen', 'smooth', 'flip_horizontal', 'flip_vertical',
       'rotate', 'resize', 'add_noise', 'bit_plane', 'morph_open', 'morph_close', 'erosion', 'dilation',
       'bgr_to_rgb', 'bgr_to_hsv', 'bgr_to_lab', 'find_contours', 'template_match',
       'ocr_extract', 'ocr_boxes', 'ocr_preprocess', 'cone', 'paramedian', 'circular']

filter_dic = {}

def select_filter(filter_name, status):
    filter_dic = {x: False for x in fil}
    if filter_name in filter_dic:
        assert type(status) == bool
        filter_dic[filter_name] = status
    return filter_dic

class App:
    isImageInstantiated = False
    isVideoInstantiated = False

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(window)
        # Let the notebook span the full width so internal rows/cols can expand
        self.notebook.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)

        # Configure main window grid so lower widgets (images, OCR panel) can expand when maximized
        # Rows: 0 = notebook, 1 = controls, 3 = image / OCR area
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0)
        self.window.grid_rowconfigure(2, weight=0)
        self.window.grid_rowconfigure(3, weight=1)

        # Columns: allow left OCR panel and right image columns to expand
        for c in (0, 2, 3):
            self.window.grid_columnconfigure(c, weight=1)
        # keep column 1 reserved for spacing / controls
        self.window.grid_columnconfigure(1, weight=0)
        
        # Create frames for different categories
        self.basic_frame = ttk.Frame(self.notebook)
        self.transform_frame = ttk.Frame(self.notebook)
        self.filter_frame = ttk.Frame(self.notebook)
        self.morph_frame = ttk.Frame(self.notebook)
        self.histogram_frame = ttk.Frame(self.notebook)
        self.threshold_frame = ttk.Frame(self.notebook)
        self.color_frame = ttk.Frame(self.notebook)
        self.advanced_frame = ttk.Frame(self.notebook)
        self.ocr_frame = ttk.Frame(self.notebook) # <--- New OCR Frame
        
        self.notebook.add(self.basic_frame, text="Basic")
        self.notebook.add(self.transform_frame, text="Transforms")
        self.notebook.add(self.filter_frame, text="Filters")
        self.notebook.add(self.morph_frame, text="Morphology")
        self.notebook.add(self.histogram_frame, text="Histogram")
        self.notebook.add(self.threshold_frame, text="Threshold")
        self.notebook.add(self.color_frame, text="Color")
        self.notebook.add(self.advanced_frame, text="Advanced")
        self.notebook.add(self.ocr_frame, text="OCR") # <--- Add OCR Tab
        
        self.setup_basic_tab()
        self.setup_transform_tab()
        self.setup_filter_tab()
        self.setup_morph_tab()
        self.setup_histogram_tab()
        self.setup_threshold_tab()
        self.setup_color_tab()
        self.setup_advanced_tab()
        self.setup_ocr_tab() # <--- Setup OCR controls
        
        # Main control buttons
        control_frame = ttk.Frame(window)
        # Put controls across the top; span columns so layout doesn't push other widgets
        control_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky='ew')
        
        ttk.Button(control_frame, text="Choose Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Open Camera", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Image", command=self.snapshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset to Original", command=self.reset_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Back", command=self.back).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", command=window.destroy).pack(side=tk.LEFT, padx=5)
        
        # OCR Text Display - Initially hidden
        ocr_display_frame = ttk.LabelFrame(window, text="Extracted Text & Language")
        self.ocr_display_frame = ocr_display_frame  # Save reference

        self.ocr_text_display = tk.Text(ocr_display_frame, width=40, height=20, wrap=tk.WORD)
        self.ocr_text_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(ocr_display_frame, command=self.ocr_text_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ocr_text_display.config(yscrollcommand=scrollbar.set)

        # Hide by default; show when OCR tab is selected
        self.ocr_display_frame.grid_remove()
        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.delay = 15
        try:
            self.window.tk.call('wm', 'iconphoto', self.window._w, tk.PhotoImage(file='test-images/icon.png'))
        except:
            pass
        self.window.mainloop()

    def setup_basic_tab(self):
        ttk.Button(self.basic_frame, text="Color/No Filter", command=self.no_filter).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.basic_frame, text="Grayscale", command=self.gray_filter).grid(row=0, column=1, padx=5, pady=5)
        
        # Rotation controls
        ttk.Label(self.basic_frame, text="Rotate (degrees):").grid(row=1, column=0, padx=5, pady=5)
        self.rotate_entry = ttk.Entry(self.basic_frame, width=10)
        self.rotate_entry.grid(row=1, column=1, padx=5, pady=5)
        self.rotate_entry.insert(0, "90")
        ttk.Button(self.basic_frame, text="Apply Rotation", command=self.rotate_image).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Button(self.basic_frame, text="Flip Horizontal", command=self.flip_horizontal).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.basic_frame, text="Flip Vertical", command=self.flip_vertical).grid(row=2, column=1, padx=5, pady=5)
        
        # Resize controls
        ttk.Label(self.basic_frame, text="Resize (width,height):").grid(row=3, column=0, padx=5, pady=5)
        self.resize_entry = ttk.Entry(self.basic_frame, width=15)
        self.resize_entry.grid(row=3, column=1, padx=5, pady=5)
        self.resize_entry.insert(0, "400,400")
        ttk.Button(self.basic_frame, text="Apply Resize", command=self.resize_image).grid(row=3, column=2, padx=5, pady=5)
        
        # Noise options (Salt, Pepper, Both)
        noise_frame = ttk.LabelFrame(self.basic_frame, text="Noise Options")
        noise_frame.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.noise_var = tk.StringVar(value="both")
        ttk.Radiobutton(noise_frame, text="Salt & Pepper", variable=self.noise_var, value="both").grid(row=0, column=0, padx=2, pady=2)
        ttk.Radiobutton(noise_frame, text="Salt Only", variable=self.noise_var, value="salt").grid(row=0, column=1, padx=2, pady=2)
        ttk.Radiobutton(noise_frame, text="Pepper Only", variable=self.noise_var, value="pepper").grid(row=0, column=2, padx=2, pady=2)
        noise_btn = ttk.Button(noise_frame, text="Add Noise", command=self.add_noise)
        noise_btn.grid(row=0, column=3, padx=4, pady=2)
        # Add tooltips after all widgets are created (after all setup_..._tab calls)
        self.window.after(100, self._add_tooltips)

    def setup_transform_tab(self):
        ttk.Button(self.transform_frame, text="Log Transform", command=self.logTransformation).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Power Transform", command=self.powerLowEnhancement).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.transform_frame, text="Gamma:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.gamma_entry = ttk.Entry(self.transform_frame, width=6)
        self.gamma_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.gamma_entry.insert(0, "0.5")
        gamma_btn = ttk.Button(self.transform_frame, text="Apply Gamma", command=self.apply_gamma)
        gamma_btn.grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Negative", command=self.negativeEnhancement).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Contrast Stretch", command=self.contrast_stretch).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Increase Contrast", command=self.increaseContrast_filter).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Decrease Contrast", command=self.decreaseContrast_filter).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Bit Plane Slicing", command=self.bit_plane_slicing).grid(row=4, column=0, padx=5, pady=5)

    def setup_filter_tab(self):
        # Stabilize the grid for the filter frame to avoid overlapping controls
        for c in range(2):
            self.filter_frame.grid_columnconfigure(c, weight=1, minsize=140)
        for r in range(8):
            self.filter_frame.grid_rowconfigure(r, minsize=48)

        # Row 0: Gaussian + Median options
        ttk.Button(self.filter_frame, text="Gaussian Blur", command=self.gauss_filter).grid(row=0, column=0, padx=8, pady=6, sticky='w')

        # Median filter with options (placed to the right of Gaussian)
        median_frame = ttk.LabelFrame(self.filter_frame, text="Median Filter Options")
        median_frame.grid(row=0, column=1, padx=8, pady=6, sticky="nsew")
        self.median_var = tk.StringVar(value="median")
        ttk.Radiobutton(median_frame, text="Median", variable=self.median_var, value="median").grid(row=0, column=0, padx=4, pady=2)
        ttk.Radiobutton(median_frame, text="Min", variable=self.median_var, value="min").grid(row=0, column=1, padx=4, pady=2)
        ttk.Radiobutton(median_frame, text="Max", variable=self.median_var, value="max").grid(row=0, column=2, padx=4, pady=2)
        median_btn = ttk.Button(median_frame, text="Apply Median Filter Option", command=self.median_filter)
        median_btn.grid(row=0, column=3, padx=4, pady=2)

        # Row 1: Average blur and Sobel options
        ttk.Button(self.filter_frame, text="Average Blur", command=self.average_filter).grid(row=1, column=0, padx=8, pady=6, sticky='w')
        sobel_frame = ttk.LabelFrame(self.filter_frame, text="Sobel Edge Detection")
        sobel_frame.grid(row=1, column=1, padx=8, pady=6, sticky="nsew")
        self.sobel_var = tk.StringVar(value="all")
        ttk.Radiobutton(sobel_frame, text="Horizontal", variable=self.sobel_var, value="horizontal").grid(row=0, column=0, padx=3, pady=2)
        ttk.Radiobutton(sobel_frame, text="Vertical", variable=self.sobel_var, value="vertical").grid(row=0, column=1, padx=3, pady=2)
        ttk.Radiobutton(sobel_frame, text="Diagonal", variable=self.sobel_var, value="diagonal").grid(row=0, column=2, padx=3, pady=2)
        ttk.Radiobutton(sobel_frame, text="All", variable=self.sobel_var, value="all").grid(row=0, column=3, padx=3, pady=2)
        sobel_btn = ttk.Button(sobel_frame, text="Apply Sobel Edge Detection", command=self.sobel_filter)
        sobel_btn.grid(row=0, column=4, padx=4, pady=2)

        # Add tooltips for new buttons after creation
        self._add_tooltip(median_btn, "Apply Median Filter Option", "Remove salt-and-pepper noise or highlight min/max values. Use when image has random black/white dots or to smooth while preserving edges.")
        self._add_tooltip(sobel_btn, "Apply Sobel Edge Detection", "Detect edges in the image using the Sobel method. Use to highlight boundaries and transitions in brightness.")
        # Row 2: Laplacian & Canny
        ttk.Button(self.filter_frame, text="Laplacian Edge", command=self.laplace_filter).grid(row=2, column=0, padx=8, pady=6, sticky='w')
        ttk.Button(self.filter_frame, text="Canny Edge", command=self.canny_edge).grid(row=2, column=1, padx=8, pady=6, sticky='w')

        # Row 3: Prewitt & Sharpen
        ttk.Button(self.filter_frame, text="Prewitt Edge", command=self.prewitt_filter).grid(row=3, column=0, padx=8, pady=6, sticky='w')
        ttk.Button(self.filter_frame, text="Sharpen", command=self.sharpen).grid(row=3, column=1, padx=8, pady=6, sticky='w')

        # Row 4: Unsharp & Smooth
        ttk.Button(self.filter_frame, text="Unsharp Mask", command=self.unsharp_filter).grid(row=4, column=0, padx=8, pady=6, sticky='w')
        ttk.Button(self.filter_frame, text="Smooth", command=self.smooth).grid(row=4, column=1, padx=8, pady=6, sticky='w')

        # Other filters further down
        ttk.Button(self.filter_frame, text="Cone Filter", command=self.cone_filter).grid(row=5, column=0, padx=8, pady=6, sticky='w')
        ttk.Button(self.filter_frame, text="Paramedian Filter", command=self.paramedian_filter).grid(row=5, column=1, padx=8, pady=6, sticky='w')
        ttk.Button(self.filter_frame, text="Circular Filter", command=self.circular_filter).grid(row=6, column=0, padx=8, pady=6, sticky='w')

    def setup_morph_tab(self):
        ttk.Button(self.morph_frame, text="Erosion", command=self.erosion).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Dilation", command=self.dilation).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Morphological Open", command=self.morph_open).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Morphological Close", command=self.morph_close).grid(row=1, column=1, padx=5, pady=5)

    def setup_histogram_tab(self):
        ttk.Button(self.histogram_frame, text="Show Histogram", command=self.show_histogram).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.histogram_frame, text="Histogram Equalization", command=self.histogram_filter).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.histogram_frame, text="Adaptive Hist. Eq.", command=self.adaptive_hist_eq).grid(row=1, column=0, padx=5, pady=5)

    def setup_threshold_tab(self):
        ttk.Button(self.threshold_frame, text="Simple Threshold", command=self.threshold_filter).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.threshold_frame, text="Adaptive Threshold", command=self.adaptive_threshold).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.threshold_frame, text="Otsu's Threshold", command=self.otsu_threshold).grid(row=1, column=0, padx=5, pady=5)

    def setup_color_tab(self):
        ttk.Button(self.color_frame, text="BGR to RGB", command=self.bgr_to_rgb).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.color_frame, text="BGR to HSV", command=self.bgr_to_hsv).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.color_frame, text="BGR to LAB", command=self.bgr_to_lab).grid(row=1, column=0, padx=5, pady=5)

    def setup_advanced_tab(self):
        ttk.Button(self.advanced_frame, text="Find Contours", command=self.find_contours).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.advanced_frame, text="Template Matching", command=self.template_matching).grid(row=0, column=1, padx=5, pady=5)
    
    # New OCR Setup
    def setup_ocr_tab(self):
        ttk.Label(self.ocr_frame, text="OCR Operations").grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(self.ocr_frame, text="Extract Text", command=self.ocr_extract).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.ocr_frame, text="Draw Text Boxes", command=self.ocr_draw_boxes).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.ocr_frame, text="Show Preprocessed Image", command=self.ocr_preprocess).grid(row=2, column=0, padx=5, pady=5)


    def on_tab_changed(self, event):
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")
        
        if tab_text == "OCR":
            # Show OCR text panel when OCR tab is selected
            self.ocr_display_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")
        else:
            # Hide otherwise
            self.ocr_display_frame.grid_remove()


    def select_video(self):
        # self.video_source = 0
        # self.vid = VideoCap(self.video_source, self.window)
        # self.vid.all_filters = select_filter('color', True)
        # self.vid.update()
        # self.isVideoInstantiated = True
        pass

    def select_image(self):
        self.img = ImageCap(self.window)
        
        # Check if user actually selected an image (didn't cancel)
        if hasattr(self.img, 'original_image') and self.img.original_image is not None:
            self.img.all_filters = select_filter('color', True)
            self.img.update()
            self.isImageInstantiated = True
        else:
            self.isImageInstantiated = False

    def snapshot(self):
        import cv2
        if self.isImageInstantiated:
            cv2.imwrite(path + r"\image-" + time.strftime("%d-%m-%Y-%H-%M-%S") + '.jpg', 
                       cv2.cvtColor(self.img.filtered_image, cv2.COLOR_RGB2BGR))
            print('Image saved!')
        elif self.isVideoInstantiated:
            cv2.imwrite(path + r"\image-" + time.strftime("%d-%m-%Y-%H-%M-%S") + '.jpg', self.vid.frame)
            print('Image saved!')

    def reset_original(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('color', True)
            self.img.reset_to_original()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('color', True)

    def back(self):
        if self.isImageInstantiated:
            self.img.back_to_previous()

    # Filter methods
    def gray_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('gray', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('gray', True)

    def no_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('color', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('color', True)

    def rotate_image(self):
        if self.isImageInstantiated:
            try:
                angle = float(self.rotate_entry.get())
                self.img.rotation_angle = angle
                self.img.all_filters = select_filter('rotate', True)
                self.img.update()
            except ValueError:
                print("Invalid angle")

    def flip_horizontal(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('flip_horizontal', True)
            self.img.update()

    def flip_vertical(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('flip_vertical', True)
            self.img.update()

    def resize_image(self):
        if self.isImageInstantiated:
            try:
                dims = self.resize_entry.get().split(',')
                self.img.resize_dims = (int(dims[0]), int(dims[1]))
                self.img.all_filters = select_filter('resize', True)
                self.img.update()
            except:
                print("Invalid dimensions")

    def add_noise(self):
        if self.isImageInstantiated:
            # Pass noise type to image handler
            self.img.noise_type = getattr(self, 'noise_var', None).get() if hasattr(self, 'noise_var') else 'both'
            self.img.all_filters = select_filter('add_noise', True)
            self.img.update()

    def _add_tooltip(self, widget, name, text):
        # Helper to add a tooltip to a single widget
        import tkinter.ttk as ttk_mod
        try:
            from tkinter import Tooltip
        except ImportError:
            # Fallback: simple tooltip class
            class Tooltip:
                def __init__(self, widget, text):
                    self.widget = widget
                    self.text = text
                    self.tipwindow = None
                    widget.bind("<Enter>", self.show)
                    widget.bind("<Leave>", self.hide)
                def show(self, event=None):
                    if self.tipwindow or not self.text:
                        return
                    x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0,0,0,0)
                    x = x + self.widget.winfo_rootx() + 25
                    y = y + cy + self.widget.winfo_rooty() + 20
                    self.tipwindow = tw = tk.Toplevel(self.widget)
                    tw.wm_overrideredirect(1)
                    tw.wm_geometry(f"+{x}+{y}")
                    label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                                    background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                                    font=("tahoma", "8", "normal"))
                    label.pack(ipadx=1)
                def hide(self, event=None):
                    tw = self.tipwindow
                    self.tipwindow = None
                    if tw:
                        tw.destroy()
        Tooltip(widget, text)

    def _add_tooltips(self):
        # Helper to add tooltips to all buttons (very brief, beginner-friendly)
        import tkinter.ttk as ttk_mod
        try:
            from tkinter import Tooltip
        except ImportError:
            # Fallback: simple tooltip class
            class Tooltip:
                def __init__(self, widget, text):
                    self.widget = widget
                    self.text = text
                    self.tipwindow = None
                    widget.bind("<Enter>", self.show)
                    widget.bind("<Leave>", self.hide)
                def show(self, event=None):
                    if self.tipwindow or not self.text:
                        return
                    x, y, _, cy = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0,0,0,0)
                    x = x + self.widget.winfo_rootx() + 25
                    y = y + cy + self.widget.winfo_rooty() + 20
                    self.tipwindow = tw = tk.Toplevel(self.widget)
                    tw.wm_overrideredirect(1)
                    tw.wm_geometry(f"+{x}+{y}")
                    label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                                    background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                                    font=("tahoma", "8", "normal"))
                    label.pack(ipadx=1)
                def hide(self, event=None):
                    tw = self.tipwindow
                    self.tipwindow = None
                    if tw:
                        tw.destroy()
        # Map: button text -> tooltip
        tooltips = {
            "Choose Image": "Open an image file to process.",
            "Open Camera": "Start the webcam to capture images.",
            "Save Image": "Save the current processed image.",
            "Reset to Original": "Restore the image to its original state.",
            "Back": "Undo the last operation.",
            "Close": "Close the application.",
            "Color/No Filter": "Show the original image without any filter.",
            "Grayscale": "Convert the image to black and white (grayscale).",
            "Apply Rotation": "Rotate the image by the specified angle.",
            "Flip Horizontal": "Flip the image left-to-right.",
            "Flip Vertical": "Flip the image upside down.",
            "Apply Resize": "Resize the image to the given width and height.",
            "Add Noise": "Add random white (salt), black (pepper), or both types of noise to the image. Used to simulate real-world image corruption.",
            "Log Transform": "Brighten dark areas using a logarithmic transformation.",
            "Power Transform": "Adjust image brightness using a power-law curve.",
            "Apply Gamma": "Change image brightness using a custom gamma value. Lower values brighten, higher values darken.",
            "Negative": "Invert all colors in the image.",
            "Contrast Stretch": "Increase the difference between dark and light areas.",
            "Increase Contrast": "Make the image more vivid by increasing contrast.",
            "Decrease Contrast": "Make the image more muted by decreasing contrast.",
            "Bit Plane Slicing": "Show the image using only certain bits, highlighting details.",
            "Gaussian Blur": "Blur the image to reduce noise and detail.",
            "Median": "Remove salt-and-pepper noise while preserving edges.",
            "Min": "Highlight the darkest areas (min filter).",
            "Max": "Highlight the brightest areas (max filter).",
            "Apply": "Apply the selected filter or operation.",
            "Average Blur": "Blur the image by averaging neighboring pixels.",
            "Laplacian Edge": "Detect edges using the Laplacian method.",
            "Canny Edge": "Detect edges using the Canny method.",
            "Prewitt Edge": "Detect edges using the Prewitt method.",
            "Sharpen": "Make the image details clearer.",
            "Unsharp Mask": "Sharpen the image by enhancing edges.",
            "Smooth": "Reduce image noise and detail.",
            "Cone Filter": "Apply a custom cone-shaped filter.",
            "Paramedian Filter": "Apply a paramedian filter for noise reduction.",
            "Circular Filter": "Apply a circular filter for smoothing.",
            "Erosion": "Remove small white spots (noise) from the image.",
            "Dilation": "Expand white areas in the image.",
            "Morphological Open": "Remove small objects from the foreground.",
            "Morphological Close": "Fill small holes in the foreground.",
            "Show Histogram": "Show the distribution of pixel values.",
            "Histogram Equalization": "Improve contrast using histogram equalization.",
            "Adaptive Hist. Eq.": "Enhance contrast adaptively.",
            "Simple Threshold": "Convert image to black and white using a fixed value.",
            "Adaptive Threshold": "Convert image to black and white using local values.",
            "Otsu's Threshold": "Automatically find the best threshold for black and white.",
            "BGR to RGB": "Convert image color from BGR to RGB.",
            "BGR to HSV": "Convert image color from BGR to HSV.",
            "BGR to LAB": "Convert image color from BGR to LAB.",
            "Find Contours": "Detect and outline shapes in the image.",
            "Template Matching": "Find a small image (template) inside the main image.",
            "Extract Text": "Extract text from the image using OCR.",
            "Draw Text Boxes": "Draw boxes around detected text in the image.",
            "Show Preprocessed Image": "Show the image after preparing it for OCR."
        }
        # Attach tooltips to all ttk.Buttons and ttk.Radiobuttons
        def attach_tooltips_to_frame(frame):
            for child in frame.winfo_children():
                if isinstance(child, (ttk_mod.Button, ttk_mod.Radiobutton)):
                    txt = child.cget("text")
                    if txt in tooltips:
                        Tooltip(child, tooltips[txt])
                elif isinstance(child, (ttk_mod.LabelFrame, ttk_mod.Frame)):
                    attach_tooltips_to_frame(child)
        # Main frames
        for frame in [self.basic_frame, self.transform_frame, self.filter_frame, self.morph_frame, self.histogram_frame, self.threshold_frame, self.color_frame, self.advanced_frame, self.ocr_frame]:
            attach_tooltips_to_frame(frame)
        # Control frame (top buttons)
        for widget in self.window.winfo_children():
            if isinstance(widget, ttk_mod.Frame):
                attach_tooltips_to_frame(widget)

    def logTransformation(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('logTransformation', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('logTransformation', True)


    def apply_gamma(self):
        if self.isImageInstantiated:
            try:
                gamma_val = float(self.gamma_entry.get())
            except Exception:
                print("Invalid gamma value")
                return
            # Set gamma_value attribute on ImageCap for use in gamma_custom
            setattr(self.img, 'gamma_value', gamma_val)
            # Set gamma_custom True without replacing all_filters if possible
            if hasattr(self.img.all_filters, '__setitem__'):
                self.img.all_filters['gamma_custom'] = True
            else:
                self.img.all_filters = select_filter('gamma_custom', True)
            print(f"all_filters after gamma: {self.img.all_filters}")
            self.img.update()
            # Show the filtered image, not the original
            self.img.update_panel(self.img.filtered_image, self.img.filtered_image)

    def powerLowEnhancement(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('powerLowEnhancement', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('powerLowEnhancement', True)

    def negativeEnhancement(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('negativeEnhancement', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('negativeEnhancement', True)

    def contrast_stretch(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('contrast_stretch', True)
            self.img.update()

    def bit_plane_slicing(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('bit_plane', True)
            self.img.update()

    def increaseContrast_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('increaseContrast', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('increaseContrast', True)

    def decreaseContrast_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('decreaseContrast', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('decreaseContrast', True)

    def gauss_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('gauss', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('gauss', True)

    def median_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('median', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('median', True)

    def average_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('average', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('average', True)

    def sobel_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('sobel', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('sobel', True)

    def laplace_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('laplace', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('laplace', True)

    def canny_edge(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('canny', True)
            self.img.update()

    def prewitt_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('prewitt', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('prewitt', True)

    def sharpen(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('sharpen', True)
            self.img.update()

    def unsharp_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('unsharp', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('unsharp', True)

    def smooth(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('smooth', True)
            self.img.update()

    def erosion(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('erosion', True)
            self.img.update()

    def dilation(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('dilation', True)
            self.img.update()

    def morph_open(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('morph_open', True)
            self.img.update()

    def morph_close(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('morph_close', True)
            self.img.update()

    def min_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('min', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('min', True)

    def max_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('max', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('max', True)

    def show_histogram(self):
        if self.isImageInstantiated:
            self.img.show_histogram()

    def histogram_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('histogramEqualization', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('histogramEqualization', True)

    def adaptive_hist_eq(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('adaptive_hist_eq', True)
            self.img.update()

    def threshold_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('threshold', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('threshold', True)

    def adaptive_threshold(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('adaptive_threshold', True)
            self.img.update()

    def otsu_threshold(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('otsu_threshold', True)
            self.img.update()

    def bgr_to_rgb(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('bgr_to_rgb', True)
            self.img.update()

    def bgr_to_hsv(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('bgr_to_hsv', True)
            self.img.update()

    def bgr_to_lab(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('bgr_to_lab', True)
            self.img.update()

    def find_contours(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('find_contours', True)
            self.img.update()

    def template_matching(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('template_match', True)
            self.img.update()

    # OCR Callback Methods
    def ocr_extract(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('ocr_extract', True)
            self.img.update()

            detected_lang = self.img.detect_language()
            self.ocr_text_display.delete(1.0, tk.END)
            self.ocr_text_display.insert(1.0, f"🌐 Language: {detected_lang}\n\n{'='*40}\n\n{self.img.ocr_text}")

        if hasattr(self, 'ocr_text_display'):
            detected_lang = self.img.detect_language()
            self.ocr_text_display.delete(1.0, tk.END)
            self.ocr_text_display.insert(1.0, f"Detected Language: {detected_lang}\n\n{self.img.ocr_text}")

    def ocr_draw_boxes(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('ocr_boxes', True)
            self.img.update()

    def ocr_preprocess(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('ocr_preprocess', True)
            self.img.update()

    def cone_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('cone', True)
            self.img.update()

    def paramedian_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('paramedian', True)
            self.img.update()

    def circular_filter(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('circular', True)
            self.img.update()

App(tk.Tk(), 'Enhanced ToolBox')