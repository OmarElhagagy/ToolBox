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


# NEW CODE HERE:
def get_available_languages():
    """Get list of available Tesseract languages"""
    if not TESSERACT_AVAILABLE:
        return ['eng']
    try:
        tessdata_path = r"C:\Program Files\Tesseract-OCR\tessdata"
        import os
        langs = []
        for file in os.listdir(tessdata_path):
            if file.endswith('.traineddata'):
                langs.append(file.replace('.traineddata', ''))
        return sorted(langs) if langs else ['eng']
    except:
        return ['eng']

class ImageCap:
    def compress_image_lossless(self):
        """
        Compress the current image using PNG encoding (lossless, color supported).
        Returns the compressed byte array and original shape.
        """
        img = self.current_image
        # PNG encoding (lossless, color)
        success, encoded = cv2.imencode('.png', img)
        if not success:
            return None, img.shape
        return encoded, img.shape

    def decompress_image_lossless(self, encoded, shape):
        """
        Decompress PNG-encoded image back to numpy array (color supported).
        """
        decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        # Ensure shape matches original
        if decoded.shape != shape:
            decoded = cv2.resize(decoded, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return decoded
    def __init__(self, window=None):
        self.noise_type = 'both'  # default
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



    def calculate_compression_ratio(self, original, compressed):
        """
        Calculate the compression ratio between the original and compressed images.
        Note: The compression method used in this toolbox is lossy (DCT with quantization),
        so the reconstructed image will not be identical to the original.
        """
        return original.nbytes / compressed.nbytes if compressed.nbytes > 0 else 0

    def calculate_rmse(self, original, reconstructed):
        return np.sqrt(np.mean((original.astype(float) - reconstructed.astype(float)) ** 2))

    def calculate_psnr(self, original, reconstructed):
        mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((255 ** 2) / mse)

    def update_panel(self, original_image, filtered_image):
        # Use the dedicated image_display_frame for image placement
        master = None
        if hasattr(self.window, 'image_display_frame'):
            master = self.window.image_display_frame
        else:
            master = self.window

        # Get available size for image display
        master.update_idletasks()
        width = master.winfo_width()
        height = master.winfo_height()
        # Fallback if not yet rendered
        if width < 100 or height < 100:
            width, height = 800, 500
        # Reserve some space for padding
        width = max(200, width - 40)
        height = max(200, height - 40)

        def resize_to_fit(img, maxsize=(width//2, height)):
            h, w = img.shape[:2]
            scale = min(maxsize[0]/w, maxsize[1]/h, 1.0)
            new_w, new_h = int(w*scale), int(h*scale)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img
            img = np.clip(img, 0, 255).astype(np.uint8)
            pil_img = PIL.Image.fromarray(img)
            pil_img = pil_img.resize((new_w, new_h), PIL.Image.LANCZOS)
            return PIL.ImageTk.PhotoImage(pil_img)

        original_tk = resize_to_fit(original_image)
        filtered_tk = resize_to_fit(filtered_image)

        # Remove old panels if they exist (to avoid overlap)
        if self.panelA is not None:
            self.panelA.destroy()
        if self.panelB is not None:
            self.panelB.destroy()

        self.panelA = tkinter.Label(master, image=original_tk)
        self.panelA.image = original_tk
        self.panelA.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        self.panelB = tkinter.Label(master, image=filtered_tk)
        self.panelB.image = filtered_tk
        self.panelB.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)

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

    def region_growing(self, img, seed, threshold):
        h, w = img.shape
        segmented = np.zeros_like(img)
        visited = np.zeros((h, w), dtype=bool)
        seed_val = img[seed[1], seed[0]]
        queue = [seed]
        
        while queue:
            x, y = queue.pop(0)
            if visited[y, x]:
                continue
            if abs(int(img[y, x]) - int(seed_val)) <= threshold:
                segmented[y, x] = 255
                visited[y, x] = True
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                        queue.append((nx, ny))
        return segmented

    def calculate_compression_ratio(self, original, compressed):
        return original.nbytes / compressed.nbytes if compressed.nbytes > 0 else 0

    def calculate_rmse(self, original, reconstructed):
        return np.sqrt(np.mean((original.astype(float) - reconstructed.astype(float)) ** 2))

    def calculate_psnr(self, original, reconstructed):
        mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((255 ** 2) / mse)

    def compress_image(self, quality=50):
        """
        JPEG-like color image compression using DCT and quantization on YCbCr channels.
        Returns compressed blocks for each channel, original shape, and padded shape.
        """
        img = self.current_image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Convert to YCbCr
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        h, w, c = img_ycbcr.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        padded = np.pad(img_ycbcr, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        compressed_blocks = [[] for _ in range(3)]
        # Process each channel separately
        for ch in range(3):
            channel = padded[:, :, ch]
            for i in range(0, padded.shape[0], 8):
                for j in range(0, padded.shape[1], 8):
                    block = channel[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    quant_factor = 100 - quality
                    quantized = np.round(dct_block / (1 + quant_factor / 10))
                    compressed_blocks[ch].append(quantized)
        return compressed_blocks, (h, w), (padded.shape[0], padded.shape[1])

    def decompress_image(self, compressed_blocks, original_shape, padded_shape, quality=50):
        """
        Reconstruct color image from JPEG-like compressed blocks (YCbCr channels).
        """
        h_pad, w_pad = padded_shape
        reconstructed = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
        quant_factor = 100 - quality
        for ch in range(3):
            block_idx = 0
            for i in range(0, h_pad, 8):
                for j in range(0, w_pad, 8):
                    dequantized = compressed_blocks[ch][block_idx] * (1 + quant_factor / 10)
                    idct_block = cv2.idct(dequantized)
                    reconstructed[i:i+8, j:j+8, ch] = idct_block
                    block_idx += 1
        # Remove padding
        h, w = original_shape
        reconstructed = reconstructed[:h, :w, :]
        # Convert back to RGB
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        reconstructed_rgb = cv2.cvtColor(reconstructed, cv2.COLOR_YCrCb2RGB)
        return reconstructed_rgb

    def perform_ocr(self, lang='eng', preprocess=True):
        """Extract text from image using Tesseract OCR, supporting multi-language"""
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
            # Pass selected language to pytesseract
            custom_config = r'--oem 3 --psm 6'
            self.ocr_data = pytesseract.image_to_data(img_for_ocr, lang=lang, config=custom_config, output_type=Output.DICT)
            self.ocr_text = pytesseract.image_to_string(img_for_ocr, lang=lang, config=custom_config)
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
            lang = getattr(self, 'ocr_language', 'eng')
            text = self.perform_ocr(lang=lang, preprocess=True)
            print(f"\n=== OCR Results ===\n{text}\n=================\n")
            self.filtered_image = self.current_image.copy()

        elif get_f('ocr_boxes'):
            lang = getattr(self, 'ocr_language', 'eng')
            self.perform_ocr(lang=lang, preprocess=True)
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

        elif get_f('log_filter'):
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image
            kernel = np.array([[0, 0, -1, 0, 0],
                            [0, -1, -2, -1, 0],
                            [-1, -2, 16, -2, -1],
                            [0, -1, -2, -1, 0],
                            [0, 0, -1, 0, 0]], dtype=np.float32)
            log_filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            log_filtered = np.clip(np.absolute(log_filtered), 0, 255).astype(np.uint8)
            self.filtered_image = cv2.cvtColor(log_filtered, cv2.COLOR_GRAY2RGB)
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
            print(f"Gamma applied: {gamma_val}, min: {self.filtered_image.min()}, max: {self.filtered_image.max()}")
            self.current_image = self.filtered_image.copy()

        elif get_f('gamma_custom'):
            gamma_val = getattr(self, 'gamma_value', 0.5)
            try:
                gamma_val = float(gamma_val)
            except Exception:
                gamma_val = 0.5
            # Apply gamma to the current filtered image only
            normalized = self.filtered_image / 255.0
            transformed = np.power(normalized, gamma_val)
            self.filtered_image = np.clip(transformed * 255, 0, 255).astype(np.uint8)
            print(f"Gamma applied: {gamma_val}, min: {self.filtered_image.min()}, max: {self.filtered_image.max()}")
            # Do NOT update self.current_image here; only filtered_image is changed

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
            # Salt, pepper, or both noise
            noise_img = self.current_image.copy()
            prob = 0.05  # probability of noise
            noise_type = getattr(self, 'noise_type', 'both')
            if noise_type == 'salt':
                salt = np.random.random(self.current_image.shape[:2]) < prob
                noise_img[salt] = 255
            elif noise_type == 'pepper':
                pepper = np.random.random(self.current_image.shape[:2]) < prob
                noise_img[pepper] = 0
            else:  # both
                salt = np.random.random(self.current_image.shape[:2]) < prob/2
                noise_img[salt] = 255
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

        elif get_f('region_growing'):
            # Convert to grayscale for segmentation
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
                orig_rgb = self.current_image.copy()
            else:
                gray = self.current_image
                orig_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            seed = getattr(self, 'seed_point', (2, 2))
            threshold = getattr(self, 'similarity_threshold', 50)
            segmented = self.region_growing(gray, seed, threshold)
            # Overlay: highlight segmented region in red on original image
            overlay = orig_rgb.copy()
            # Create a red mask where segmented==255
            red_mask = np.zeros_like(overlay)
            red_mask[..., 0] = 255  # Red channel
            # Use alpha blending to overlay red on segmented region
            alpha = 0.5
            mask = segmented == 255
            overlay[mask] = cv2.addWeighted(overlay[mask], 1 - alpha, red_mask[mask], alpha, 0)
            self.filtered_image = overlay
            self.current_image = self.filtered_image.copy()

        elif get_f('compress_analyze'):
            quality = getattr(self, 'compression_quality', 50)
            # Compress
            compressed_blocks, orig_shape, padded_shape = self.compress_image(quality)
            # Decompress
            reconstructed = self.decompress_image(compressed_blocks, orig_shape, padded_shape, quality)
            # Ensure both images are color and same shape
            original = self.current_image
            if reconstructed.shape != original.shape:
                reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
            # Compression ratio: compare original and compressed sizes
            original_size = original.nbytes
            num_blocks = len(compressed_blocks[0])
            compressed_size = num_blocks * 64 * 4 * 3  # 64 values per block, 4 bytes per float, 3 channels
            cr = original_size / compressed_size if compressed_size > 0 else 0
            rmse = self.calculate_rmse(original, reconstructed)
            psnr = self.calculate_psnr(original, reconstructed)
            print(f"\n=== Compression Analysis ===")
            print(f"Quality: {quality}%")
            print(f"Compression Ratio: {cr:.2f}:1")
            print(f"RMSE: {rmse:.2f}")
            print(f"PSNR: {psnr:.2f} dB")
            print(f"===========================\n")
            self.filtered_image = reconstructed.copy()
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