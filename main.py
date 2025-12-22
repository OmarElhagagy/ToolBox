import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import time
import os
import pytesseract
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
       'ocr_extract', 'ocr_boxes', 'ocr_preprocess', 'cone', 'paramedian', 'circular', 'region_growing', 'log_filter', 'compress_analyze']

filter_dic = {}

def select_filter(filter_name, status):
    filter_dic = {x: False for x in fil}
    if filter_name in filter_dic:
        assert type(status) == bool
        filter_dic[filter_name] = status
    return filter_dic

class App:
    def compress_analyze_func_lossless(self):
        if self.isImageInstantiated:
            try:
                # Lossless compression (PNG)
                encoded, orig_shape = self.img.compress_image_lossless()
                if encoded is None:
                    self.compression_metrics_label.config(text="Lossless compression failed.")
                    return
                reconstructed = self.img.decompress_image_lossless(encoded, orig_shape)
                original = self.img.current_image
                if reconstructed.shape != original.shape:
                    reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
                original_size = original.nbytes
                compressed_size = encoded.nbytes
                cr = original_size / compressed_size if compressed_size > 0 else 0
                rmse = self.img.calculate_rmse(original, reconstructed)
                psnr = self.img.calculate_psnr(original, reconstructed)
                metrics_text = f"Compression Ratio: {cr:.2f}\nRMSE: {rmse:.2f}\nPSNR: {psnr:.2f} dB"
                self.compression_metrics_label.config(text=metrics_text)
                self.img.filtered_image = reconstructed.copy()
                self.img.current_image = self.img.filtered_image.copy()
                self.img.update_panel(self.img.original_image, self.img.filtered_image)
            except Exception as e:
                self.compression_metrics_label.config(text="Lossless compression error.")
    isImageInstantiated = False
    isVideoInstantiated = False

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        
        self.notebook = ttk.Notebook(window)
        self.notebook.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0)
        self.window.grid_rowconfigure(2, weight=0)
        self.window.grid_rowconfigure(3, weight=1)

        for c in (0, 2, 3):
            self.window.grid_columnconfigure(c, weight=1)
        self.window.grid_columnconfigure(1, weight=0)
        
        self.basic_frame = ttk.Frame(self.notebook)
        self.transform_frame = ttk.Frame(self.notebook)
        self.filter_frame = ttk.Frame(self.notebook)
        self.morph_frame = ttk.Frame(self.notebook)
        self.histogram_frame = ttk.Frame(self.notebook)
        self.threshold_frame = ttk.Frame(self.notebook)
        self.color_frame = ttk.Frame(self.notebook)
        self.advanced_frame = ttk.Frame(self.notebook)
        self.ocr_frame = ttk.Frame(self.notebook)
        self.segmentation_frame = ttk.Frame(self.notebook)
        self.compression_frame = ttk.Frame(self.notebook)


        
        self.notebook.add(self.basic_frame, text="Basic")
        self.notebook.add(self.transform_frame, text="Transforms")
        self.notebook.add(self.filter_frame, text="Filters")
        self.notebook.add(self.morph_frame, text="Morphology")
        self.notebook.add(self.histogram_frame, text="Histogram")
        self.notebook.add(self.threshold_frame, text="Threshold")
        self.notebook.add(self.color_frame, text="Color")
        self.notebook.add(self.advanced_frame, text="Advanced")
        self.notebook.add(self.ocr_frame, text="OCR")
        self.notebook.add(self.segmentation_frame, text="Segmentation")
        self.notebook.add(self.compression_frame, text="Compression")


        
        self.setup_basic_tab()
        self.setup_transform_tab()
        self.setup_filter_tab()
        self.setup_morph_tab()
        self.setup_histogram_tab()
        self.setup_threshold_tab()
        self.setup_color_tab()
        self.setup_advanced_tab()
        self.setup_ocr_tab()
        self.setup_segmentation_tab()
        self.setup_compression_tab()
        
        control_frame = ttk.Frame(window)
        control_frame.grid(row=1, column=0, columnspan=4, pady=10, sticky='ew')
        
        ttk.Button(control_frame, text="Choose Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Open Camera", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Image", command=self.snapshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset to Original", command=self.reset_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Back", command=self.back).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", command=window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Dedicated frame for image display (row=3, col=0-3)
        self.image_display_frame = ttk.Frame(window)
        self.image_display_frame.grid(row=3, column=0, columnspan=4, sticky="nsew", padx=10, pady=10)
        self.window.image_display_frame = self.image_display_frame  # For ImageCap to access

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
        # Professional, clean layout for Basic tab
        for c in range(4):
            self.basic_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(6):
            self.basic_frame.grid_rowconfigure(r, minsize=36)

        # --- Basic Filters Section ---
        basic_label = ttk.Label(self.basic_frame, text="Basic Filters", font=("Segoe UI", 10, "bold"))
        basic_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.basic_frame, text="Color/No Filter", command=self.no_filter).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.basic_frame, text="Grayscale", command=self.gray_filter).grid(row=1, column=1, padx=8, pady=4, sticky='ew')

        # --- Rotation Section ---
        rotate_label = ttk.Label(self.basic_frame, text="Rotation", font=("Segoe UI", 10, "bold"))
        rotate_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Label(self.basic_frame, text="Rotate (degrees):").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.rotate_entry = ttk.Entry(self.basic_frame, width=10)
        self.rotate_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.rotate_entry.insert(0, "90")
        ttk.Button(self.basic_frame, text="Apply Rotation", command=self.rotate_image).grid(row=3, column=2, padx=5, pady=5, sticky='ew')

        # --- Flip Section ---
        flip_label = ttk.Label(self.basic_frame, text="Flip", font=("Segoe UI", 10, "bold"))
        flip_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.basic_frame, text="Flip Horizontal", command=self.flip_horizontal).grid(row=5, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.basic_frame, text="Flip Vertical", command=self.flip_vertical).grid(row=5, column=1, padx=8, pady=4, sticky='ew')

        # --- Resize Section ---
        resize_label = ttk.Label(self.basic_frame, text="Resize", font=("Segoe UI", 10, "bold"))
        resize_label.grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Label(self.basic_frame, text="Resize (width,height):").grid(row=7, column=0, padx=5, pady=5, sticky='e')
        self.resize_entry = ttk.Entry(self.basic_frame, width=15)
        self.resize_entry.grid(row=7, column=1, padx=5, pady=5, sticky='w')
        self.resize_entry.insert(0, "400,400")
        ttk.Button(self.basic_frame, text="Apply Resize", command=self.resize_image).grid(row=7, column=2, padx=5, pady=5, sticky='ew')

        # --- Noise Section ---
        noise_label = ttk.Label(self.basic_frame, text="Noise Options", font=("Segoe UI", 10, "bold"))
        noise_label.grid(row=8, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        noise_frame = ttk.Frame(self.basic_frame)
        noise_frame.grid(row=9, column=0, columnspan=3, sticky="ew", padx=8, pady=2)
        self.noise_var = tk.StringVar(value="both")
        ttk.Radiobutton(noise_frame, text="Salt & Pepper", variable=self.noise_var, value="both").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(noise_frame, text="Salt Only", variable=self.noise_var, value="salt").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(noise_frame, text="Pepper Only", variable=self.noise_var, value="pepper").pack(side=tk.LEFT, padx=2)
        noise_btn = ttk.Button(noise_frame, text="Add Noise", command=self.add_noise)
        noise_btn.pack(side=tk.LEFT, padx=6)
        # Add tooltips after all widgets are created (after all setup_..._tab calls)
        self.window.after(100, self._add_tooltips)

    def setup_transform_tab(self):
        for c in range(3):
            self.transform_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(6):
            self.transform_frame.grid_rowconfigure(r, minsize=36)

        # --- Intensity Transformations ---
        intensity_label = ttk.Label(self.transform_frame, text="Intensity Transformations", font=("Segoe UI", 10, "bold"))
        intensity_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.transform_frame, text="Log Transform", command=self.logTransformation).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.transform_frame, text="Power Transform", command=self.powerLowEnhancement).grid(row=1, column=1, padx=8, pady=4, sticky='ew')

        # --- Gamma Section ---
        gamma_label = ttk.Label(self.transform_frame, text="Gamma Correction", font=("Segoe UI", 10, "bold"))
        gamma_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Label(self.transform_frame, text="Gamma:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
        self.gamma_entry = ttk.Entry(self.transform_frame, width=6)
        self.gamma_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.gamma_entry.insert(0, "0.5")
        gamma_btn = ttk.Button(self.transform_frame, text="Apply Gamma", command=self.apply_gamma)
        gamma_btn.grid(row=3, column=2, padx=5, pady=5, sticky='ew')

        # --- Other Transforms ---
        other_label = ttk.Label(self.transform_frame, text="Other Transforms", font=("Segoe UI", 10, "bold"))
        other_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.transform_frame, text="Negative", command=self.negativeEnhancement).grid(row=5, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.transform_frame, text="Contrast Stretch", command=self.contrast_stretch).grid(row=5, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.transform_frame, text="Increase Contrast", command=self.increaseContrast_filter).grid(row=6, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.transform_frame, text="Decrease Contrast", command=self.decreaseContrast_filter).grid(row=6, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.transform_frame, text="Bit Plane Slicing", command=self.bit_plane_slicing).grid(row=7, column=0, padx=8, pady=4, sticky='ew')

    def setup_filter_tab(self):
        # Professional, clean layout for Filters tab
        for c in range(4):
            self.filter_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(7):
            self.filter_frame.grid_rowconfigure(r, minsize=36)

        # --- Blur & Smoothing Section ---
        blur_label = ttk.Label(self.filter_frame, text="Blur & Smoothing", font=("Segoe UI", 10, "bold"))
        blur_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.filter_frame, text="Gaussian Blur", command=self.gauss_filter).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Average Blur", command=self.average_filter).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Smooth", command=self.smooth).grid(row=1, column=2, padx=8, pady=4, sticky='ew')

        # --- Median/Min/Max Section ---
        median_label = ttk.Label(self.filter_frame, text="Median/Min/Max", font=("Segoe UI", 10, "bold"))
        median_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        median_frame = ttk.Frame(self.filter_frame)
        median_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=2)
        self.median_var = tk.StringVar(value="median")
        ttk.Radiobutton(median_frame, text="Median", variable=self.median_var, value="median").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(median_frame, text="Min", variable=self.median_var, value="min").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(median_frame, text="Max", variable=self.median_var, value="max").pack(side=tk.LEFT, padx=2)
        median_btn = ttk.Button(median_frame, text="Apply Median Filter Option", command=self.median_filter)
        median_btn.pack(side=tk.LEFT, padx=6)

        # --- Edge Detection Section ---
        edge_label = ttk.Label(self.filter_frame, text="Edge Detection", font=("Segoe UI", 10, "bold"))
        edge_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.filter_frame, text="Canny Edge", command=self.canny_edge).grid(row=5, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Laplacian Edge", command=self.laplace_filter).grid(row=5, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="LoG Filter", command=self.log_filter_func).grid(row=5, column=4, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Prewitt Edge", command=self.prewitt_filter).grid(row=5, column=2, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Sharpen", command=self.sharpen).grid(row=5, column=3, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Unsharp Mask", command=self.unsharp_filter).grid(row=6, column=0, padx=8, pady=4, sticky='ew')

        # --- Sobel Section ---
        sobel_label = ttk.Label(self.filter_frame, text="Sobel Edge Detection", font=("Segoe UI", 10, "bold"))
        sobel_label.grid(row=2, column=2, columnspan=2, sticky="w", padx=4, pady=(8,2))
        sobel_frame = ttk.Frame(self.filter_frame)
        sobel_frame.grid(row=3, column=2, columnspan=2, sticky="ew", padx=8, pady=2)
        self.sobel_var = tk.StringVar(value="all")
        ttk.Radiobutton(sobel_frame, text="Horizontal", variable=self.sobel_var, value="horizontal").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sobel_frame, text="Vertical", variable=self.sobel_var, value="vertical").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sobel_frame, text="Diagonal", variable=self.sobel_var, value="diagonal").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(sobel_frame, text="All", variable=self.sobel_var, value="all").pack(side=tk.LEFT, padx=2)
        sobel_btn = ttk.Button(sobel_frame, text="Apply Sobel Edge Detection", command=self.sobel_filter)
        sobel_btn.pack(side=tk.LEFT, padx=6)

        # --- Specialty Filters Section ---
        specialty_label = ttk.Label(self.filter_frame, text="Specialty Filters", font=("Segoe UI", 10, "bold"))
        specialty_label.grid(row=7, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.filter_frame, text="Paramedian Filter", command=self.paramedian_filter).grid(row=8, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Circular Filter", command=self.circular_filter).grid(row=8, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.filter_frame, text="Cone Filter", command=self.cone_filter).grid(row=8, column=2, padx=8, pady=4, sticky='ew')

        # Add tooltips for new buttons after creation
        self._add_tooltip(median_btn, "Apply Median Filter Option", "Remove salt-and-pepper noise or highlight min/max values. Use when image has random black/white dots or to smooth while preserving edges.")
        self._add_tooltip(sobel_btn, "Apply Sobel Edge Detection", "Detect edges in the image using the Sobel method. Use to highlight boundaries and transitions in brightness.")

    def setup_morph_tab(self):
        for c in range(2):
            self.morph_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(3):
            self.morph_frame.grid_rowconfigure(r, minsize=36)

        morph_label = ttk.Label(self.morph_frame, text="Morphological Operations", font=("Segoe UI", 10, "bold"))
        morph_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.morph_frame, text="Erosion", command=self.erosion).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.morph_frame, text="Dilation", command=self.dilation).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.morph_frame, text="Morphological Open", command=self.morph_open).grid(row=2, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.morph_frame, text="Morphological Close", command=self.morph_close).grid(row=2, column=1, padx=8, pady=4, sticky='ew')

    def setup_histogram_tab(self):
        for c in range(2):
            self.histogram_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(2):
            self.histogram_frame.grid_rowconfigure(r, minsize=36)

        hist_label = ttk.Label(self.histogram_frame, text="Histogram Operations", font=("Segoe UI", 10, "bold"))
        hist_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.histogram_frame, text="Show Histogram", command=self.show_histogram).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.histogram_frame, text="Histogram Equalization", command=self.histogram_filter).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.histogram_frame, text="Adaptive Hist. Eq.", command=self.adaptive_hist_eq).grid(row=2, column=0, padx=8, pady=4, sticky='ew')

    def setup_threshold_tab(self):
        for c in range(2):
            self.threshold_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(2):
            self.threshold_frame.grid_rowconfigure(r, minsize=36)

        thresh_label = ttk.Label(self.threshold_frame, text="Thresholding", font=("Segoe UI", 10, "bold"))
        thresh_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.threshold_frame, text="Simple Threshold", command=self.threshold_filter).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.threshold_frame, text="Adaptive Threshold", command=self.adaptive_threshold).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.threshold_frame, text="Otsu's Threshold", command=self.otsu_threshold).grid(row=2, column=0, padx=8, pady=4, sticky='ew')

    def setup_color_tab(self):
        for c in range(2):
            self.color_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(2):
            self.color_frame.grid_rowconfigure(r, minsize=36)

        color_label = ttk.Label(self.color_frame, text="Color Space Conversions", font=("Segoe UI", 10, "bold"))
        color_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.color_frame, text="BGR to RGB", command=self.bgr_to_rgb).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.color_frame, text="BGR to HSV", command=self.bgr_to_hsv).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.color_frame, text="BGR to LAB", command=self.bgr_to_lab).grid(row=2, column=0, padx=8, pady=4, sticky='ew')

    def setup_advanced_tab(self):
        for c in range(2):
            self.advanced_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(2):
            self.advanced_frame.grid_rowconfigure(r, minsize=36)

        adv_label = ttk.Label(self.advanced_frame, text="Advanced Operations", font=("Segoe UI", 10, "bold"))
        adv_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.advanced_frame, text="Find Contours", command=self.find_contours).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.advanced_frame, text="Template Matching", command=self.template_matching).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
    
    # New OCR Setup
    def setup_ocr_tab(self):
        for c in range(2):
            self.ocr_frame.grid_columnconfigure(c, weight=1, minsize=120)
        for r in range(2):
            self.ocr_frame.grid_rowconfigure(r, minsize=36)

        ocr_label = ttk.Label(self.ocr_frame, text="OCR Operations", font=("Segoe UI", 10, "bold"))
        ocr_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        ttk.Button(self.ocr_frame, text="Extract Text", command=self.ocr_extract).grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        ttk.Button(self.ocr_frame, text="Draw Text Boxes", command=self.ocr_draw_boxes).grid(row=1, column=1, padx=8, pady=4, sticky='ew')
        ttk.Button(self.ocr_frame, text="Show Preprocessed Image", command=self.ocr_preprocess).grid(row=2, column=0, padx=8, pady=4, sticky='ew')


    def setup_segmentation_tab(self):
        for c in range(3):
            self.segmentation_frame.grid_columnconfigure(c, weight=1, minsize=120)
        
        seg_label = ttk.Label(self.segmentation_frame, text="Image Segmentation", font=("Segoe UI", 10, "bold"))
        seg_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))
        
        ttk.Label(self.segmentation_frame, text="Seed Point (x,y):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.seed_entry = ttk.Entry(self.segmentation_frame, width=10)
        self.seed_entry.insert(0, "2,2")
        self.seed_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(self.segmentation_frame, text="Threshold:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
        self.threshold_entry = ttk.Entry(self.segmentation_frame, width=10)
        self.threshold_entry.insert(0, "50")
        self.threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Button(self.segmentation_frame, text="Apply Region Growing", command=self.region_growing_func).grid(row=3, column=0, columnspan=2, padx=8, pady=4, sticky='ew')

    def setup_compression_tab(self):
        for c in range(3):
            self.compression_frame.grid_columnconfigure(c, weight=1, minsize=120)

        comp_label = ttk.Label(self.compression_frame, text="Image Compression Analysis", font=("Segoe UI", 10, "bold"))
        comp_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=(8,2))

        ttk.Label(self.compression_frame, text="Quality (1-100):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.quality_entry = ttk.Entry(self.compression_frame, width=10)
        self.quality_entry.insert(0, "50")
        self.quality_entry.grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # Button with lossy compression info
        compress_btn = ttk.Button(self.compression_frame, text="Compress & Analyze (Lossy)", command=self.compress_analyze_func)
        compress_btn.grid(row=2, column=0, padx=8, pady=4, sticky='ew')
        # Tooltip for lossy compression
        def show_tooltip_lossy(event):
            if not hasattr(self, '_compress_tooltip_lossy'):
                self._compress_tooltip_lossy = tk.Toplevel(self.compression_frame)
                self._compress_tooltip_lossy.wm_overrideredirect(True)
                label = tk.Label(
                    self._compress_tooltip_lossy,
                    text="JPEG-like color compression: uses DCT and quantization on color channels. The reconstructed image will be similar but not identical to the original.",
                    background="#ffffe0", relief="solid", borderwidth=1, font=("Segoe UI", 9))
                label.pack(ipadx=1)
            x = event.x_root + 10
            y = event.y_root + 10
            self._compress_tooltip_lossy.wm_geometry(f"+{x}+{y}")
            self._compress_tooltip_lossy.deiconify()
        def hide_tooltip_lossy(event):
            if hasattr(self, '_compress_tooltip_lossy'):
                self._compress_tooltip_lossy.withdraw()
        compress_btn.bind("<Enter>", show_tooltip_lossy)
        compress_btn.bind("<Leave>", hide_tooltip_lossy)

        # Button with lossless compression info
        compress_btn_lossless = ttk.Button(self.compression_frame, text="Compress & Analyze (Lossless)", command=self.compress_analyze_func_lossless)
        compress_btn_lossless.grid(row=2, column=1, padx=8, pady=4, sticky='ew')
        # Tooltip for lossless compression
        def show_tooltip_lossless(event):
            if not hasattr(self, '_compress_tooltip_lossless'):
                self._compress_tooltip_lossless = tk.Toplevel(self.compression_frame)
                self._compress_tooltip_lossless.wm_overrideredirect(True)
                label = tk.Label(
                    self._compress_tooltip_lossless,
                    text="Lossless PNG compression: preserves all color information. The reconstructed image will be exactly identical to the original.",
                    background="#e0ffe0", relief="solid", borderwidth=1, font=("Segoe UI", 9))
                label.pack(ipadx=1)
            x = event.x_root + 10
            y = event.y_root + 10
            self._compress_tooltip_lossless.wm_geometry(f"+{x}+{y}")
            self._compress_tooltip_lossless.deiconify()
        def hide_tooltip_lossless(event):
            if hasattr(self, '_compress_tooltip_lossless'):
                self._compress_tooltip_lossless.withdraw()
        compress_btn_lossless.bind("<Enter>", show_tooltip_lossless)
        compress_btn_lossless.bind("<Leave>", hide_tooltip_lossless)

        # Label to show compression metrics
        self.compression_metrics_label = ttk.Label(self.compression_frame, text="", font=("Segoe UI", 9))
        self.compression_metrics_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        info_label = ttk.Label(self.compression_frame, text="Shows: Compression Ratio, RMSE, PSNR", font=("Segoe UI", 8, "italic"))
        info_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)


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
            save_path = path + r"\image-" + time.strftime("%d-%m-%Y-%H-%M-%S") + '.jpg'
            cv2.imwrite(save_path, cv2.cvtColor(self.img.filtered_image, cv2.COLOR_RGB2BGR))
            print(f'Image saved at: {save_path}')
        elif self.isVideoInstantiated:
            save_path = path + r"\image-" + time.strftime("%d-%m-%Y-%H-%M-%S") + '.jpg'
            cv2.imwrite(save_path, self.vid.frame)
            print(f'Image saved at: {save_path}')

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
            "Show Preprocessed Image": "Show the image after preparing it for OCR.",
            "LoG Filter": "Detects edges using the Laplacian of Gaussian (LoG) method. Useful for highlighting fine details and boundaries in images, especially when you want to see where objects start and end.",
            "Apply Region Growing": "Segments the image starting from a chosen point, grouping together similar pixels. Use this to separate objects from the background by clicking on a part of the object you want to extract.",
            "Compress & Analyze": "Compresses the image and shows how much space is saved and how much quality is lost. Useful for understanding how image compression affects file size and clarity.",
            "Apply Median Filter Option": "Removes random black and white dots (salt-and-pepper noise) or highlights the darkest/brightest areas, depending on your choice. Helps clean up noisy images.",
            "Apply Sobel Edge Detection": "Finds edges in the image using the Sobel method. Good for seeing outlines and borders of objects.",
            "Apply Gamma": "Change image brightness using a custom gamma value. Lower values brighten, higher values darken."
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
        for frame in [
            self.basic_frame, self.transform_frame, self.filter_frame, self.morph_frame, self.histogram_frame,
            self.threshold_frame, self.color_frame, self.advanced_frame, self.ocr_frame,
            self.segmentation_frame, self.compression_frame
        ]:
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
            self.img.gamma_value = gamma_val
            # Reset all filters to False, then set gamma_custom True
            self.img.all_filters = {k: False for k in self.img.all_filters.keys()}
            self.img.all_filters['gamma_custom'] = True
            self.img.update()
            # Show the original and filtered images correctly
            self.img.update_panel(self.img.original_image, self.img.filtered_image)

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
            # Always use automatic language detection, default to 'eng' for OCR
            self.img.ocr_language = 'eng'
            self.img.all_filters = select_filter('ocr_extract', True)
            self.img.update()
            detected_lang = self.img.detect_language()
            self.ocr_text_display.delete(1.0, tk.END)
            self.ocr_text_display.insert(1.0, f"🌐 Detected Language: {detected_lang}\n\n{'='*40}\n\n{self.img.ocr_text}")

    def ocr_draw_boxes(self):
        if self.isImageInstantiated:
            self.img.ocr_language = 'eng'
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

    def region_growing_func(self):
        if self.isImageInstantiated:
            try:
                coords = self.seed_entry.get().split(',')
                self.img.seed_point = (int(coords[0]), int(coords[1]))
                self.img.similarity_threshold = int(self.threshold_entry.get())
                self.img.all_filters = select_filter('region_growing', True)
                self.img.update()
            except:
                print("Invalid parameters")

    def log_filter_func(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('log_filter', True)
            self.img.update()

    def compress_analyze_func(self):
        if self.isImageInstantiated:
            try:
                quality_str = self.quality_entry.get()
                quality = int(quality_str)
            except ValueError:
                self.compression_metrics_label.config(text="Invalid quality value: must be an integer between 1 and 100.")
                return
            if not (1 <= quality <= 100):
                self.compression_metrics_label.config(text="Invalid quality value: must be between 1 and 100.")
                return
            self.img.compression_quality = quality
            self.img.all_filters = select_filter('compress_analyze', True)
            import io, sys
            buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = buf
            self.img.update()
            sys.stdout = old_stdout
            output = buf.getvalue()
            cr, rmse, psnr = None, None, None
            for line in output.splitlines():
                if "Compression Ratio" in line:
                    cr = line.split(":", 1)[1].strip()
                elif "RMSE" in line:
                    rmse = line.split(":", 1)[1].strip()
                elif "PSNR" in line:
                    psnr = line.split(":", 1)[1].strip()
            metrics_text = ""
            if cr and rmse and psnr:
                metrics_text = f"Compression Ratio: {cr}\nRMSE: {rmse}\nPSNR: {psnr}"
            else:
                metrics_text = "Compression analysis failed."
            self.compression_metrics_label.config(text=metrics_text)

App(tk.Tk(), 'Enhanced ToolBox')