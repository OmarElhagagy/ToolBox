import tkinter
import tkinter.filedialog
import cv2
import PIL.Image
import PIL.ImageTk
import numpy as np
from matplotlib import pyplot as plt
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


try:
    import pytesseract
    from pytesseract import Output
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract library not found. Run: pip install pytesseract")
except Exception as e:
    TESSERACT_AVAILABLE = False
    print(f"Warning: Tesseract config failed: {e}")

class ImageCap:
    def __init__(self, window=None):
        self.window = window
        self.all_filters = {}
        self.ocr_text = ""
        self.ocr_data = None
        self.sobel_direction = 'all'
        self.median_mode = 'median'
        
        img_path = tkinter.filedialog.askopenfilename()
        if len(img_path) > 0:
            self.original_image = cv2.imread(img_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.current_image = self.original_image.copy()
            self.filtered_image = self.original_image.copy()
            self.previous_image = self.original_image.copy()
            self.panelA = None
            self.panelB = None
            self.rotation_angle = 90
            self.resize_dims = (400, 400)
            self.update_panel(self.original_image, self.original_image)
        else:
            self.original_image = None

    def update_panel(self, original_image, filtered_image):
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        if len(filtered_image.shape) == 2:
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            
        original_image = np.clip(original_image, 0, 255).astype(np.uint8)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        
        original_pil = PIL.Image.fromarray(original_image)
        filtered_pil = PIL.Image.fromarray(filtered_image)

        master = self.window if self.window is not None else None

        try:
            master.update_idletasks()
        except Exception:
            pass

        try:
            win_w = master.winfo_width() or 0
            win_h = master.winfo_height() or 0
        except Exception:
            win_w = 0
            win_h = 0

        if hasattr(self, 'panelA') and self.panelA is not None:
            try:
                panel_w = max(100, self.panelA.winfo_width())
                panel_h = max(100, self.panelA.winfo_height())
            except Exception:
                panel_w = max(200, int(win_w * 0.42))
                panel_h = max(200, int(win_h * 0.6))
        else:
            if win_w <= 1:
                panel_w = 500
                panel_h = 500
            else:
                panel_w = max(200, int(win_w * 0.42))
                panel_h = max(200, int(win_h * 0.6))

        def fit_within(max_w, max_h, img):
            """Scale image to fit entirely within (max_w, max_h) without upscaling."""
            img_w, img_h = img.size
            if img_w == 0 or img_h == 0:
                return img.resize((max_w, max_h), PIL.Image.LANCZOS)
            
            # Calculate scale to fit both width and height, never upscale
            scale = min(max_w / img_w, max_h / img_h, 1.0)
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            return img.resize((new_w, new_h), PIL.Image.LANCZOS)

        original_pil = fit_within(panel_w, panel_h, original_pil)
        filtered_pil = fit_within(panel_w, panel_h, filtered_pil)
        original_tk = PIL.ImageTk.PhotoImage(original_pil)
        filtered_tk = PIL.ImageTk.PhotoImage(filtered_pil)

        master = self.window if self.window is not None else None

        if self.panelA is None or self.panelB is None:
            self.panelA = tkinter.Label(master, image=original_tk)
            self.panelA.image = original_tk
            self.panelA.grid(row=3, column=2, padx=10, pady=10, sticky='nsew')
            
            self.panelB = tkinter.Label(master, image=filtered_tk)
            self.panelB.image = filtered_tk
            self.panelB.grid(row=3, column=3, padx=10, pady=10, sticky='nsew')
            try:
                master.grid_columnconfigure(2, weight=1)
                master.grid_columnconfigure(3, weight=1)
                master.grid_rowconfigure(3, weight=1)
            except Exception:
                pass
        else:
            self.panelA.configure(image=original_tk)
            self.panelB.configure(image=filtered_tk)
            self.panelA.image = original_tk
            self.panelB.image = filtered_tk

    def detect_language(self):
        """Detect language of extracted OCR text"""
        if not globals().get('LANGDETECT_AVAILABLE', False):
            return "Unknown (langdetect not installed)"
        if not self.ocr_text or len(self.ocr_text.strip()) < 3:
            return "Unknown (text too short)"
        try:
            lang_code = detect(self.ocr_text)
            lang_names = {
                'ar': 'Arabic', 'en': 'English', 'fr': 'French', 
                'de': 'German', 'es': 'Spanish', 'zh-cn': 'Chinese',
                'ja': 'Japanese', 'ru': 'Russian', 'it': 'Italian'
            }
            return lang_names.get(lang_code, lang_code)
        except:
            return "Unknown"

    def reset_to_original(self):
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.filtered_image = self.original_image.copy()
        self.ocr_text = ""
        self.ocr_data = None
        self.update_panel(self.original_image, self.original_image)

    def back_to_previous(self):
        if hasattr(self, 'previous_image') and self.previous_image is not None:
            temp = self.current_image.copy()
            self.current_image = self.previous_image.copy()
            self.previous_image = temp
            self.filtered_image = self.current_image.copy()
            self.update_panel(self.original_image, self.filtered_image)

    def show_histogram(self):
        if not hasattr(self, 'current_image'): return
        img = self.current_image
        plt.figure()
        if len(img.shape) == 2:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(hist, color='gray')
        else:
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Histogram')
        plt.xlim([0, 256])
        plt.show()

    def preprocess_for_ocr(self, image):
        """Preprocess image for better OCR results"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Adaptive thresholding for better text contrast
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed

    def perform_ocr(self, lang='eng', preprocess=True):
        """Extract text from image using Tesseract OCR"""
        if not TESSERACT_AVAILABLE:
            return "Tesseract OCR not installed. Install: pip install pytesseract"
        
        if not hasattr(self, 'current_image') or self.current_image is None:
            return "No image loaded"
        
        try:
            # Preprocess image if requested
            if preprocess:
                img_for_ocr = self.preprocess_for_ocr(self.current_image)
            else:
                if len(self.current_image.shape) == 3:
                    img_for_ocr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
                else:
                    img_for_ocr = self.current_image
            
            # Perform OCR with detailed data
            custom_config = r'--oem 3 --psm 6'
            self.ocr_data = pytesseract.image_to_data(img_for_ocr, lang=lang, 
                                                       config=custom_config, 
                                                       output_type=Output.DICT)
            
            # Extract text
            self.ocr_text = pytesseract.image_to_string(img_for_ocr, lang=lang, 
                                                         config=custom_config)
            
            return self.ocr_text.strip() if self.ocr_text else "No text detected"
            
        except Exception as e:
            return f"OCR Error: {str(e)}"

    def draw_ocr_boxes(self, confidence_threshold=60):
        """Draw bounding boxes around detected text"""
        if not TESSERACT_AVAILABLE or self.ocr_data is None:
            return self.current_image.copy()
        
        img_with_boxes = self.current_image.copy()
        if len(img_with_boxes.shape) == 2:
            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2RGB)
        
        n_boxes = len(self.ocr_data['text'])
        for i in range(n_boxes):
            if int(self.ocr_data['conf'][i]) > confidence_threshold:
                (x, y, w, h) = (self.ocr_data['left'][i], 
                               self.ocr_data['top'][i], 
                               self.ocr_data['width'][i], 
                               self.ocr_data['height'][i])
                
                # Draw rectangle
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add confidence text
                text = self.ocr_data['text'][i]
                if text.strip():
                    conf = int(self.ocr_data['conf'][i])
                    label = f"{conf}%"
                    cv2.putText(img_with_boxes, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return img_with_boxes

    def update(self):
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
            
        self.previous_image = self.current_image.copy()
        
        def get_f(name):
            return self.all_filters.get(name, False)

        if get_f('color'):
            self.filtered_image = self.current_image.copy()

        elif get_f('ocr_extract'):
            text = self.perform_ocr(preprocess=True)
            print(f"\n=== OCR Results ===\n{text}\n=================\n")
            self.filtered_image = self.current_image.copy()

        elif get_f('ocr_boxes'):
            self.perform_ocr(preprocess=True)
            self.filtered_image = self.draw_ocr_boxes(confidence_threshold=60)
            self.current_image = self.filtered_image.copy()

        elif get_f('ocr_preprocess'):
            processed = self.preprocess_for_ocr(self.current_image)
            self.filtered_image = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('gray'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
                self.filtered_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                self.filtered_image = self.current_image.copy()
            self.current_image = self.filtered_image.copy()

        elif get_f('gauss'):
            self.filtered_image = cv2.GaussianBlur(self.current_image, (21, 21), 0)
            self.current_image = self.filtered_image.copy()

        elif get_f('sobel'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobel = np.absolute(sobel)
            sobel = np.uint8(sobel)
            self.filtered_image = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('laplace'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
            laplace = np.absolute(laplace)
            laplace = np.uint8(laplace)
            self.filtered_image = cv2.cvtColor(laplace, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('canny'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            canny = cv2.Canny(gray, 50, 150)
            self.filtered_image = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('threshold'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.filtered_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('adaptive_threshold'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            self.filtered_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('otsu_threshold'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.filtered_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('median'):
            self.filtered_image = cv2.medianBlur(self.current_image, 5)
            self.current_image = self.filtered_image.copy()

        elif get_f('average'):
            self.filtered_image = cv2.blur(self.current_image, (5, 5))
            self.current_image = self.filtered_image.copy()

        elif get_f('unsharp'):
            gaussian = cv2.GaussianBlur(self.current_image, (0, 0), 2.0)
            self.filtered_image = cv2.addWeighted(self.current_image, 2.0, gaussian, -1.0, 0)
            self.filtered_image = np.clip(self.filtered_image, 0, 255).astype(np.uint8)
            self.current_image = self.filtered_image.copy()

        elif get_f('sharpen'):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            self.filtered_image = cv2.filter2D(self.current_image, -1, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('smooth'):
            self.filtered_image = cv2.bilateralFilter(self.current_image, 9, 75, 75)
            self.current_image = self.filtered_image.copy()

        elif get_f('logTransformation'):
            img_float = self.current_image.astype(np.float64)
            c = 255 / np.log(1 + np.max(img_float))
            log_image = c * np.log(1 + img_float)
            self.filtered_image = np.clip(log_image, 0, 255).astype(np.uint8)
            self.current_image = self.filtered_image.copy()

        elif get_f('negativeEnhancement'):
            self.filtered_image = 255 - self.current_image
            self.current_image = self.filtered_image.copy()

        elif get_f('powerLowEnhancement'):
            normalized = self.current_image / 255.0
            gamma = 0.8
            transformed = np.power(normalized, gamma)
            self.filtered_image = np.clip(transformed * 255, 0, 255).astype(np.uint8)
            self.current_image = self.filtered_image.copy()

        elif get_f('gamma_0_5'):
            normalized = self.current_image / 255.0
            transformed = np.power(normalized, 0.5)
            self.filtered_image = np.clip(transformed * 255, 0, 255).astype(np.uint8)
            self.current_image = self.filtered_image.copy()

        elif get_f('gamma_1_5'):
            normalized = self.current_image / 255.0
            transformed = np.power(normalized, 1.5)
            self.filtered_image = np.clip(transformed * 255, 0, 255).astype(np.uint8)
            self.current_image = self.filtered_image.copy()

        elif get_f('contrast_stretch'):
            if len(self.current_image.shape) == 3:
                self.filtered_image = np.zeros_like(self.current_image)
                for i in range(3):
                    channel = self.current_image[:, :, i]
                    p2, p98 = np.percentile(channel, (2, 98))
                    self.filtered_image[:, :, i] = np.clip((channel - p2) * 255.0 / (p98 - p2 + 1e-8), 0, 255)
            else:
                p2, p98 = np.percentile(self.current_image, (2, 98))
                self.filtered_image = np.clip((self.current_image - p2) * 255.0 / (p98 - p2 + 1e-8), 0, 255)
            self.filtered_image = self.filtered_image.astype(np.uint8)
            self.current_image = self.filtered_image.copy()

        elif get_f('increaseContrast'):
            self.filtered_image = cv2.convertScaleAbs(self.current_image, alpha=1.5, beta=10)
            self.current_image = self.filtered_image.copy()

        elif get_f('decreaseContrast'):
            self.filtered_image = cv2.convertScaleAbs(self.current_image, alpha=0.7, beta=0)
            self.current_image = self.filtered_image.copy()

        elif get_f('min'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.filtered_image = cv2.erode(self.current_image, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('max'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            self.filtered_image = cv2.dilate(self.current_image, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('erosion'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.filtered_image = cv2.erode(self.current_image, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('dilation'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.filtered_image = cv2.dilate(self.current_image, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('morph_open'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.filtered_image = cv2.morphologyEx(self.current_image, cv2.MORPH_OPEN, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('morph_close'):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.filtered_image = cv2.morphologyEx(self.current_image, cv2.MORPH_CLOSE, kernel)
            self.current_image = self.filtered_image.copy()

        elif get_f('prewitt'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            prewittx = cv2.filter2D(gray, cv2.CV_32F, kernelx)
            prewitty = cv2.filter2D(gray, cv2.CV_32F, kernely)
            prewitt = np.sqrt(prewittx**2 + prewitty**2)
            prewitt = np.clip(prewitt, 0, 255).astype(np.uint8)
            self.filtered_image = cv2.cvtColor(prewitt, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('histogramEqualization'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            equalized = cv2.equalizeHist(gray)
            self.filtered_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('adaptive_hist_eq'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = clahe.apply(gray)
            self.filtered_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('flip_horizontal'):
            self.filtered_image = cv2.flip(self.current_image, 1)
            self.current_image = self.filtered_image.copy()

        elif get_f('flip_vertical'):
            self.filtered_image = cv2.flip(self.current_image, 0)
            self.current_image = self.filtered_image.copy()

        elif get_f('rotate'):
            h, w = self.current_image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            self.filtered_image = cv2.warpAffine(self.current_image, matrix, (w, h))
            self.current_image = self.filtered_image.copy()

        elif get_f('resize'):
            self.filtered_image = cv2.resize(self.current_image, self.resize_dims, interpolation=cv2.INTER_LINEAR)
            self.current_image = self.filtered_image.copy()

        elif get_f('add_noise'):
            # Salt and pepper noise
            noise_img = self.current_image.copy()
            prob = 0.05  # probability of noise
            
            # Salt noise (white)
            salt = np.random.random(self.current_image.shape[:2]) < prob/2
            noise_img[salt] = 255
            
            # Pepper noise (black)
            pepper = np.random.random(self.current_image.shape[:2]) < prob/2
            noise_img[pepper] = 0
            
            self.filtered_image = noise_img
            self.current_image = self.filtered_image.copy()

        elif get_f('cone'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            median_filtered = cv2.medianBlur(gray, 5)
            gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 0)
            cone_filtered = cv2.addWeighted(median_filtered, 0.7, gaussian_filtered, 0.3, 0)
            # Fixed: ensure result is uint8 before color conversion
            cone_filtered = cone_filtered.astype(np.uint8)
            self.filtered_image = cv2.cvtColor(cone_filtered, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('paramedian'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            # Paramedian filter (hybrid of median and mean)
            median_filtered = cv2.medianBlur(gray, 5)
            mean_filtered = cv2.blur(gray, (5, 5))
            paramedian = cv2.addWeighted(median_filtered, 0.5, mean_filtered, 0.5, 0)
            self.filtered_image = cv2.cvtColor(paramedian, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('circular'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            # Fixed: ensure result is uint8 before color conversion
            circular_filtered = cv2.medianBlur(gray, 7)
            circular_filtered = circular_filtered.astype(np.uint8)
            self.filtered_image = cv2.cvtColor(circular_filtered, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()

        elif get_f('bit_plane'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            bit_plane_7 = ((gray >> 7) & 1) * 255
            self.filtered_image = cv2.cvtColor(bit_plane_7, cv2.COLOR_GRAY2RGB)
            self.current_image = self.filtered_image.copy()
        
        

        elif get_f('bgr_to_rgb'):
            if len(self.current_image.shape) == 3:
                self.filtered_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            else:
                self.filtered_image = self.current_image.copy()
            self.current_image = self.filtered_image.copy()

        elif get_f('bgr_to_hsv'):
            if len(self.current_image.shape) == 3:
                self.filtered_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)
            else:
                self.filtered_image = self.current_image.copy()
            self.current_image = self.filtered_image.copy()

        elif get_f('bgr_to_lab'):
            if len(self.current_image.shape) == 3:
                self.filtered_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2LAB)
            else:
                self.filtered_image = self.current_image.copy()
            self.current_image = self.filtered_image.copy()

        elif get_f('find_contours'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.filtered_image = self.current_image.copy()
            if len(self.filtered_image.shape) == 2:
                self.filtered_image = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(self.filtered_image, contours, -1, (0, 255, 0), 2)
            self.current_image = self.filtered_image.copy()

        elif get_f('template_match'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            h, w = gray.shape
            th, tw = max(h // 8, 20), max(w // 8, 20)
            template = gray[h//3:h//3+th, w//3:w//3+tw]
            if template.size > 0:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(result)
                self.filtered_image = self.current_image.copy()
                if len(self.filtered_image.shape) == 2:
                    self.filtered_image = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2RGB)
                cv2.rectangle(self.filtered_image, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 2)
                self.current_image = self.filtered_image.copy()
            else:
                self.filtered_image = self.current_image.copy()

        self.update_panel(self.original_image, self.filtered_image)