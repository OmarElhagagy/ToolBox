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
       'ocr_extract', 'ocr_boxes', 'ocr_preprocess'] # <--- OCR keys added

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
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
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
        control_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(control_frame, text="Choose Image", command=self.select_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Open Camera", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Save Image", command=self.snapshot).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset to Original", command=self.reset_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Back", command=self.back).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Close", command=window.destroy).pack(side=tk.LEFT, padx=5)
        
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
        
        ttk.Button(self.basic_frame, text="Add Noise", command=self.add_noise).grid(row=4, column=0, padx=5, pady=5)

    def setup_transform_tab(self):
        ttk.Button(self.transform_frame, text="Log Transform", command=self.logTransformation).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Power Transform", command=self.powerLowEnhancement).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Gamma 0.5", command=self.gamma_0_5).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Gamma 1.5", command=self.gamma_1_5).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Negative", command=self.negativeEnhancement).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Contrast Stretch", command=self.contrast_stretch).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Increase Contrast", command=self.increaseContrast_filter).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Decrease Contrast", command=self.decreaseContrast_filter).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.transform_frame, text="Bit Plane Slicing", command=self.bit_plane_slicing).grid(row=4, column=0, padx=5, pady=5)

    def setup_filter_tab(self):
        ttk.Button(self.filter_frame, text="Gaussian Blur", command=self.gauss_filter).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Median Blur", command=self.median_filter).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Average Blur", command=self.average_filter).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Sobel Edge", command=self.sobel_filter).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Laplacian Edge", command=self.laplace_filter).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Canny Edge", command=self.canny_edge).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Prewitt Edge", command=self.prewitt_filter).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Sharpen", command=self.sharpen).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Unsharp Mask", command=self.unsharp_filter).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(self.filter_frame, text="Smooth", command=self.smooth).grid(row=4, column=1, padx=5, pady=5)

    def setup_morph_tab(self):
        ttk.Button(self.morph_frame, text="Erosion", command=self.erosion).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Dilation", command=self.dilation).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Morphological Open", command=self.morph_open).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Morphological Close", command=self.morph_close).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Min Filter", command=self.min_filter).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.morph_frame, text="Max Filter", command=self.max_filter).grid(row=2, column=1, padx=5, pady=5)

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
        ttk.Label(self.ocr_frame, text="OCR Operations (Requires Tesseract)").grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        ttk.Button(self.ocr_frame, text="Extract Text (to Console)", command=self.ocr_extract).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(self.ocr_frame, text="Draw Text Boxes", command=self.ocr_draw_boxes).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.ocr_frame, text="Show Preprocessed Image", command=self.ocr_preprocess).grid(row=2, column=0, padx=5, pady=5)

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
            self.img.all_filters = select_filter('add_noise', True)
            self.img.update()

    def logTransformation(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('logTransformation', True)
            self.img.update()
        elif self.isVideoInstantiated:
            self.vid.all_filters = select_filter('logTransformation', True)

    def gamma_0_5(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('gamma_0_5', True)
            self.img.update()

    def gamma_1_5(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('gamma_1_5', True)
            self.img.update()

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

    def ocr_draw_boxes(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('ocr_boxes', True)
            self.img.update()

    def ocr_preprocess(self):
        if self.isImageInstantiated:
            self.img.all_filters = select_filter('ocr_preprocess', True)
            self.img.update()

App(tk.Tk(), 'Enhanced ToolBox')