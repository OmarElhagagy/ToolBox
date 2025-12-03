import tkinter
import tkinter.filedialog
import cv2
import PIL.Image
import PIL.ImageTk
import numpy as np
from matplotlib import pyplot as plt

class ImageCap:
    def __init__(self, window=None):
        self.window = window
        self.all_filters = {} # Initialize to empty dict to prevent crashes
        
        img_path = tkinter.filedialog.askopenfilename()
        if len(img_path) > 0:
            self.original_image = cv2.imread(img_path)
            # Convert BGR to RGB
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
            # Handle case where user cancels file selection
            self.original_image = None

    def update_panel(self, original_image, filtered_image):
        # Ensure images are RGB and uint8
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        if len(filtered_image.shape) == 2:
            filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2RGB)
            
        original_image = np.clip(original_image, 0, 255).astype(np.uint8)
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        
        # Resize for display
        original_pil = PIL.Image.fromarray(original_image)
        filtered_pil = PIL.Image.fromarray(filtered_image)
        original_pil = original_pil.resize((400, 400), PIL.Image.LANCZOS)
        filtered_pil = filtered_pil.resize((400, 400), PIL.Image.LANCZOS)
        original_tk = PIL.ImageTk.PhotoImage(original_pil)
        filtered_tk = PIL.ImageTk.PhotoImage(filtered_pil)

        # Use self.window as the master for the labels
        master = self.window if self.window is not None else None

        if self.panelA is None or self.panelB is None:
            self.panelA = tkinter.Label(master, image=original_tk)
            self.panelA.image = original_tk
            # These grid positions (Row 3, Col 2 & 3) place the images to the right/bottom of your controls
            self.panelA.grid(row=3, column=2, padx=10, pady=10)
            
            self.panelB = tkinter.Label(master, image=filtered_tk)
            self.panelB.image = filtered_tk
            self.panelB.grid(row=3, column=3, padx=10, pady=10)
        else:
            self.panelA.configure(image=original_tk)
            self.panelB.configure(image=filtered_tk)
            self.panelA.image = original_tk
            self.panelB.image = filtered_tk

    def reset_to_original(self):
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.filtered_image = self.original_image.copy()
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

    def update(self):
        if not hasattr(self, 'original_image') or self.original_image is None:
            return
            
        self.previous_image = self.current_image.copy()
        
        # Helper to safely get filter status
        def get_f(name):
            return self.all_filters.get(name, False)

        if get_f('color'):
            self.filtered_image = self.current_image.copy()

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
            noise = np.random.normal(0, 25, self.current_image.shape)
            noisy = self.current_image.astype(np.float64) + noise
            self.filtered_image = np.clip(noisy, 0, 255).astype(np.uint8)
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