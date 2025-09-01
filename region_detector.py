#!/usr/bin/env python3
"""
Region Detector Script
Monitors a user-defined screen region for code changes and sends Discord notifications.
"""

import pyautogui
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import simpledialog
import requests
import time
import threading
import json
import os
import hashlib
from datetime import datetime
import re

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\tools\tesseract\tesseract.exe"
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

class RegionDetector:
    def __init__(self):
        self.region = None
        self.webhook_url = None
        self.monitoring = False
        self.last_code_hash = None
        self.last_text = ""
        self.sent_codes = set()  # Track sent codes to prevent duplicates
        self.last_detected_codes = set()  # Track the most recently detected codes
        self.redeemed_codes = set()  # Track codes that have been redeemed
        self.redemption_notified = False  # Track if redemption notification was sent
        self.last_notification_time = 0  # Track last notification time for cooldown
        
        
        # Create data directory for storing state
        self.data_dir = "detector_data"
        os.makedirs(self.data_dir, exist_ok=True)
        self.state_file = os.path.join(self.data_dir, "sent_codes.json")
        
        # Load previously sent codes
        self.load_sent_codes()
        
    def load_sent_codes(self):
        """Load previously sent codes from file"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.sent_codes = set(data.get('sent_codes', []))
                    self.redeemed_codes = set(data.get('redeemed_codes', []))
                    print(f"Loaded {len(self.sent_codes)} previously sent codes, {len(self.redeemed_codes)} redeemed codes")
        except Exception as e:
            print(f"Error loading sent codes: {e}")
            self.sent_codes = set()
    
    def save_sent_codes(self):
        """Save sent codes to file"""
        try:
            data = {
                'sent_codes': list(self.sent_codes),
                'redeemed_codes': list(self.redeemed_codes),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving sent codes: {e}")
    
    def select_region(self):
        """Allow user to select a screen region using a GUI"""
        root = tk.Tk()
        root.title("Region Selector")
        root.geometry("400x200")
        root.resizable(False, False)
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (400 // 2)
        y = (root.winfo_screenheight() // 2) - (200 // 2)
        root.geometry(f"400x200+{x}+{y}")
        
        tk.Label(root, text="Click 'Select Region' and drag to select the area to monitor", 
                font=("Arial", 12), wraplength=350).pack(pady=20)
        
        def start_selection():
            root.withdraw()  # Hide the main window
            self._select_region_interactive()
            if self.region:
                root.destroy()
            else:
                root.deiconify()  # Show the main window again
        
        tk.Button(root, text="Select Region", command=start_selection, 
                 font=("Arial", 12), bg="#4CAF50", fg="white", 
                 padx=20, pady=10).pack(pady=10)
        
        tk.Button(root, text="Cancel", command=root.destroy, 
                 font=("Arial", 12), bg="#f44336", fg="white", 
                 padx=20, pady=5).pack(pady=5)
        
        root.mainloop()
        return self.region is not None
    
    def _select_region_interactive(self):
        """Interactive region selection"""
        try:
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            
            # Create a selection window
            window_name = "Select Region - Click and drag to select area, press ENTER to confirm, ESC to cancel"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 800)
            
            # Variables for mouse callback
            drawing = False
            start_point = None
            end_point = None
            temp_img = screenshot_cv.copy()
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, start_point, end_point, temp_img
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    start_point = (x, y)
                    end_point = (x, y)
                
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    temp_img = screenshot_cv.copy()
                    end_point = (x, y)
                    cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)
                    cv2.imshow(window_name, temp_img)
                
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    end_point = (x, y)
                    cv2.rectangle(temp_img, start_point, end_point, (0, 255, 0), 2)
                    cv2.imshow(window_name, temp_img)
            
            cv2.setMouseCallback(window_name, mouse_callback)
            cv2.imshow(window_name, screenshot_cv)
            
            print("Instructions:")
            print("- Click and drag to select the region to monitor")
            print("- Press ENTER to confirm selection")
            print("- Press ESC to cancel")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    if start_point and end_point:
                        # Convert to screen coordinates
                        x1, y1 = start_point
                        x2, y2 = end_point
                        
                        # Ensure proper ordering
                        left = min(x1, x2)
                        top = min(y1, y2)
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                        
                        if width > 10 and height > 10:  # Minimum size check
                            self.region = (left, top, width, height)
                            print(f"Region selected: {self.region}")
                            break
                        else:
                            print("Selected region too small. Please try again.")
                    else:
                        print("No region selected. Please try again.")
                
                elif key == 27:  # ESC key
                    print("Selection cancelled.")
                    break
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in region selection: {e}")
            self.region = None
    
    def setup_webhook(self):
        """Get Discord webhook URL from user"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        webhook_url = simpledialog.askstring(
            "Discord Webhook", 
            "Enter your Discord webhook URL:",
            show='*'  # Hide the URL for privacy
        )
        
        root.destroy()
        
        if webhook_url and webhook_url.strip():
            self.webhook_url = webhook_url.strip()
            return True
        return False
    
    def extract_text_from_image(self, image):
        """Extract text from image using OCR"""
        try:
            print("🔍 Starting OCR detection...")
            
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Try basic OCR first for character detection
            result = self.basic_ocr_approach(gray)
            if result:
                print(f"✅ OCR result: '{result}'")
                return self.clean_ocr_text(result)
            
            # Fallback to general text OCR
            result = self.general_text_ocr(gray)
            if result:
                print(f"✅ General OCR result: '{result}'")
                return self.clean_ocr_text(result)
            
            print("❌ No text detected")
            return ""
        
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def basic_ocr_approach(self, gray_image):
        """Optimized OCR for separated character boxes"""
        try:
            height, width = gray_image.shape
            print(f"   📏 Processing image: {width}x{height}")
            
            # Aggressive scaling for small character boxes
            scale_factor = 5 if min(height, width) < 200 else 4
            scaled = cv2.resize(gray_image, (width * scale_factor, height * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
            print(f"   📈 Scaled to: {scaled.shape[1]}x{scaled.shape[0]}")
            
            # Enhanced preprocessing for character boxes
            # Step 1: Noise reduction while preserving character edges
            denoised = cv2.bilateralFilter(scaled, 9, 75, 75)
            
            # Step 2: Strong contrast enhancement for clear character separation
            enhanced = cv2.convertScaleAbs(denoised, alpha=3.0, beta=25)
            
            # Step 3: Adaptive threshold to handle varying backgrounds
            adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Step 4: Morphological operations to clean up characters
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
            
            # Try multiple OCR configurations for separated characters
            ocr_configs = [
                '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',  # Single word
                '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',  # Single line
                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',  # Block
                '--psm 11 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'  # Sparse
            ]
            
            best_result = ""
            best_confidence = 0
            
            for config in ocr_configs:
                try:
                    # Get OCR result with confidence
                    data = pytesseract.image_to_data(cleaned, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Extract characters with confidence
                    chars = []
                    confidences = []
                    
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        conf = int(data['conf'][i])
                        
                        if text and conf > 30:  # Decent confidence threshold
                            # Split multi-character strings into individual characters
                            for char in text:
                                if char.isalnum():
                                    chars.append(char.upper())
                                    confidences.append(conf)
                    
                    if chars and len(chars) <= 8:  # Reasonable character count
                        result_text = ' '.join(chars)
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        # Prefer results with exactly 5 characters
                        if len(chars) == 5:
                            avg_confidence += 15  # Bonus for expected count
                        
                        if avg_confidence > best_confidence:
                            best_confidence = avg_confidence
                            best_result = result_text
                            print(f"   ✅ Config {config[:10]}...: '{result_text}' (conf: {avg_confidence:.1f})")
                
                except Exception as e:
                    continue
            
            return best_result if best_result else None
            
        except Exception as e:
            print(f"   ❌ Basic OCR failed: {e}")
            return None
    

    def general_text_ocr(self, gray_image):
        """General OCR approach with character validation"""
        try:
            height, width = gray_image.shape
            
            # Conservative scaling to avoid noise
            scale_factor = 2
            scaled = cv2.resize(gray_image, (width * scale_factor, height * scale_factor), 
                              interpolation=cv2.INTER_CUBIC)
            
            # Gentle enhancement
            enhanced = cv2.convertScaleAbs(scaled, alpha=1.8, beta=15)
            
            # Apply threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try different OCR modes
            configs = [
                '--psm 8',  # Single word
                '--psm 7',  # Single line
                '--psm 6'   # Block
            ]
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(thresh, config=config).strip()
                    if text and 5 <= len(text) <= 50:  # Reasonable length
                        return text
                except:
                    continue
            
            return None
            
        except Exception as e:
            return None
    
    def post_process_characters(self, text):
        """Post-process detected characters to extract clean alphanumeric codes"""
        if not text:
            return text
            
        # Extract alphanumeric characters
        chars = [char.upper() for char in text if char.isalnum()]
        
        if len(chars) >= 3:
            result = ' '.join(chars[:8])  # Limit to 8 characters max
            print(f"   🔧 Post-processed: '{result}'")
            return result
        else:
            return text
    
    def clean_ocr_text(self, text):
        """Clean up common OCR recognition errors"""
        # Common OCR character mistakes
        replacements = {
            '@': '0',  # @ often mistaken for 0
            '©': '0',  # © often mistaken for 0
            'O': '0',  # O often mistaken for 0 in codes
            '|': 'I',  # | often mistaken for I
            '!': 'I',  # ! often mistaken for I
            '5': 'S',  # Sometimes 5 and S are confused
            '1': 'I',  # Sometimes 1 and I are confused
        }
        
        cleaned = text
        for mistake, correction in replacements.items():
            # Only replace if it's likely a code context (short alphanumeric strings)
            if len(text) <= 20 and any(c.isalnum() for c in text):
                cleaned = cleaned.replace(mistake, correction)
        
        return cleaned
    
    def detect_code_and_status(self, text):
        """Detect code changes and redemption status with spam prevention"""
        # Clean up the text
        cleaned_text = text.strip()
        
        # Filter out obvious garbage/spam text
        if len(cleaned_text) > 100:  # If text is too long, it's probably garbage
            print(f"⚠️ Text too long ({len(cleaned_text)} chars), filtering...")
            # Try to extract just the 5-character pattern
            pattern_match = re.search(r'\b[a-zA-Z0-9]\s+[a-zA-Z0-9]\s+[a-zA-Z0-9]\s+[a-zA-Z0-9]\s+[a-zA-Z0-9]\b', cleaned_text)
            if pattern_match:
                cleaned_text = pattern_match.group(0)
                print(f"   ✅ Extracted pattern: '{cleaned_text}'")
            else:
                # Check for CODE REDEEMED in the garbage
                if 'code redeemed' in cleaned_text.lower():
                    cleaned_text = "CODE REDEEMED"
                    print(f"   ✅ Found CODE REDEEMED in text")
                else:
                    print(f"   ❌ No valid pattern found, ignoring")
                    return [], False
        
        # Very specific redemption detection - only trigger on "CODE REDEEMED" text
        text_lower = cleaned_text.lower()
        
        # Only detect redemption when the specific "CODE REDEEMED" text is visible
        is_redeemed = 'code redeemed' in text_lower
        
        # Debug output (shortened)
        display_text = cleaned_text[:50] + "..." if len(cleaned_text) > 50 else cleaned_text
        print(f"📝 Detected: '{display_text}'")
        
        if is_redeemed:
            print(f"🔴 REDEMPTION STATUS DETECTED!")
        else:
            print(f"✅ Code appears active")
        
        # Return the cleaned text as our "code" - we'll monitor changes to this
        return [cleaned_text] if cleaned_text else [], is_redeemed
    
    def create_enhanced_screenshot(self, original_image, detected_codes, text, is_redeemed=False):
        """Create an enhanced screenshot with code detection highlighting"""
        try:
            # Create a copy of the original image
            enhanced = original_image.copy()
            draw = ImageDraw.Draw(enhanced)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 16)
                small_font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add a semi-transparent overlay at the top for information
            overlay = Image.new('RGBA', enhanced.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Draw background for text
            info_height = 60
            overlay_draw.rectangle([(0, 0), (enhanced.width, info_height)], 
                                 fill=(0, 0, 0, 180))
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            overlay_draw.text((10, 5), f"Detected: {timestamp}", 
                            fill=(255, 255, 255, 255), font=small_font)
            
            # Just show the timestamp - no code text overlay
            pass
            
            # Try to highlight detected codes in the image
            if detected_codes:
                # Use OCR to get word positions
                try:
                    # Convert PIL to OpenCV format
                    cv_image = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    
                    # Get detailed OCR data with bounding boxes
                    ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                    
                    # Highlight detected codes
                    for i, word in enumerate(ocr_data['text']):
                        if word.strip():
                            # Check if this word is part of any detected code
                            for code in detected_codes:
                                if word.strip() in code or code in word.strip():
                                    # Get bounding box coordinates
                                    x = ocr_data['left'][i]
                                    y = ocr_data['top'][i]
                                    w = ocr_data['width'][i]
                                    h = ocr_data['height'][i]
                                    
                                    # Draw highlight rectangle
                                    highlight_color = (255, 0, 0, 100) if is_redeemed else (0, 255, 0, 100)
                                    overlay_draw.rectangle([(x-2, y+info_height-2), (x+w+2, y+h+info_height+2)], 
                                                         outline=highlight_color[:3], width=3)
                                    break
                except Exception as e:
                    print(f"⚠️ Could not highlight codes in image: {e}")
            
            # Composite the overlay onto the original image
            enhanced = Image.alpha_composite(enhanced.convert('RGBA'), overlay)
            enhanced = enhanced.convert('RGB')
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️ Error creating enhanced screenshot: {e}")
            return original_image
    
    def send_discord_notification(self, message, image_path, is_redeemed=False):
        """Send notification to Discord webhook"""
        try:
            if not self.webhook_url:
                print("❌ No webhook URL configured!")
                return False
                
            print(f"🔗 Using webhook URL: {self.webhook_url[:50]}...")
            
            # Prepare the message
            embed_color = 0xff0000 if is_redeemed else 0x00ff00  # Red for redeemed, green for new code
            
            # Create embed based on type
            embed = {
                "title": "🚫 Code Redemption Alert" if is_redeemed else "🆕 New Code Detected!",
                "description": message,
                "color": embed_color,
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {
                    "text": "Region Detector Bot • Code Status Monitor" if is_redeemed else "Region Detector Bot • Code Change Monitor"
                },
                "fields": [
                    {
                        "name": "⚠️ Status" if is_redeemed else "✅ Status",
                        "value": "Code has been redeemed/used" if is_redeemed else "Code updated successfully",
                        "inline": True
                    },
                    {
                        "name": "🕒 Detected At",
                        "value": datetime.now().strftime("%H:%M:%S"),
                        "inline": True
                    }
                ]
            }
            
            # Prepare files for upload
            files = {}
            if os.path.exists(image_path):
                print(f"📎 Attaching screenshot: {image_path}")
                with open(image_path, 'rb') as f:
                    files['file'] = f.read()
                embed["image"] = {"url": "attachment://screenshot.png"}
            else:
                print(f"⚠️ Screenshot not found: {image_path}")
            
            payload = {
                "embeds": [embed]
            }
            
            print(f"📤 Sending payload with {len(payload['embeds'])} embed(s)")
            
            # Send the message
            if files:
                response = requests.post(
                    self.webhook_url,
                    data={"payload_json": json.dumps(payload)},
                    files={"screenshot.png": files['file']},
                    timeout=30
                )
            else:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    timeout=30
                )
            
            print(f"📨 Response status: {response.status_code}")
            
            # Discord webhooks return 200 or 204 for success
            if response.status_code in [200, 204]:
                print(f"✅ Discord notification sent successfully!")
                return True
            else:
                print(f"❌ Failed to send Discord notification: {response.status_code}")
                print(f"Response headers: {dict(response.headers)}")
                print(f"Response text: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"⏰ Request timed out while sending Discord notification")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"🌐 Connection error while sending Discord notification: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending Discord notification: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def monitor_region(self):
        """Main monitoring loop"""
        print(f"🔍 Starting monitoring of region: {self.region}")
        print("Press Ctrl+C to stop monitoring")
        
        screenshot_count = 0
        
        try:
            while self.monitoring:
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Take screenshot of the selected region
                x, y, width, height = self.region
                screenshot = pyautogui.screenshot(region=(x, y, width, height))
                
                # Save screenshot
                screenshot_path = os.path.join(self.data_dir, f"screenshot_{timestamp}.png")
                screenshot.save(screenshot_path)
                
                # Extract text using OCR
                current_text = self.extract_text_from_image(screenshot)
                
                if current_text:
                    print(f"📝 Detected text: {current_text[:100]}...")
                    
                    # Detect codes and redeemed status
                    codes, is_redeemed = self.detect_code_and_status(current_text)
                    
                    # Create hash of current text for change detection
                    current_hash = hashlib.md5(current_text.encode()).hexdigest()
                    
                    # Check if text has changed
                    text_changed = current_hash != self.last_code_hash
                    
                    if text_changed:
                        print("🔄 Text content changed!")
                        
                        # Simple change detection - if we have text and it's different from before
                        if codes and codes != self.last_detected_codes:
                            # Check if this code is already redeemed
                            if any(code in self.redeemed_codes for code in codes):
                                print(f"🚫 Code change detected but code is already redeemed: {codes}")
                                print(f"   Not sending notification for redeemed code")
                            else:
                                # Add cooldown to prevent spam (minimum 10 seconds between notifications)
                                current_time = time.time()
                                if current_time - self.last_notification_time < 10:
                                    print(f"⏰ Cooldown active - skipping notification (last sent {current_time - self.last_notification_time:.1f}s ago)")
                                else:
                                    print(f"🎯 Code changed: '{self.last_detected_codes}' → '{codes[0]}'")
                                    
                                    # Reset redemption notification flag for new codes
                                    self.redemption_notified = False
                                    
                                    # Create enhanced screenshot
                                    try:
                                        enhanced_screenshot = self.create_enhanced_screenshot(screenshot, codes, current_text, is_redeemed)
                                        enhanced_path = os.path.join(self.data_dir, f"change_{timestamp}.png")
                                        enhanced_screenshot.save(enhanced_path)
                                        screenshot_to_send = enhanced_path
                                    except Exception as e:
                                        print(f"⚠️ Could not create enhanced screenshot: {e}")
                                        screenshot_to_send = screenshot_path
                                    
                                    message = ""  # Empty message - just show the fields
                                    print(f"📤 Sending notification for code change...")
                                    
                                    if self.send_discord_notification(message, screenshot_to_send, False):
                                        print(f"✅ Change notification sent successfully")
                                        self.last_notification_time = current_time  # Update cooldown timer
                                        notification_sent = True
                                    else:
                                        print(f"❌ Failed to send change notification")
                        
                        elif codes:
                            print(f"🔄 Same text as before: '{codes[0]}'")
                        
                        else:
                            print("📝 No text detected in current scan")
                        
                    # Check for redemption status (outside of text_changed block)
                    if is_redeemed and not self.redemption_notified:
                        print(f"🔴 REDEMPTION STATUS DETECTED!")
                        print(f"   🚨 Code redemption found - sending ONE-TIME notification")
                        
                        # Mark current codes as redeemed
                        if codes:
                            self.redeemed_codes.update(codes)
                            print(f"   📝 Marked codes as redeemed: {codes}")
                        
                        # Create screenshot for redemption
                        try:
                            enhanced_screenshot = self.create_enhanced_screenshot(screenshot, codes, current_text, True)
                            enhanced_path = os.path.join(self.data_dir, f"REDEEMED_{timestamp}.png")
                            enhanced_screenshot.save(enhanced_path)
                            screenshot_to_send = enhanced_path
                        except Exception as e:
                            screenshot_to_send = screenshot_path
                        
                        message = ""
                        print(f"📤 Sending ONE-TIME redemption notification...")
                        if self.send_discord_notification(message, screenshot_to_send, True):
                            print(f"✅ 🚨 REDEMPTION notification sent successfully!")
                            self.redemption_notified = True  # Mark as notified to prevent spam
                            self.save_sent_codes()  # Save the redeemed status
                        else:
                            print(f"❌ Failed to send redemption notification")
                    
                    elif is_redeemed and self.redemption_notified:
                        print(f"🔴 Redemption status still detected, but notification already sent (preventing spam)")
                        
                        # Update tracking variables
                        self.last_code_hash = current_hash
                        self.last_text = current_text
                    self.last_detected_codes = codes
                
                else:
                    print("📷 Screenshot taken, no text detected")
                
                # Clean up old screenshots (keep only last 10)
                self.cleanup_old_screenshots()
                
                # Wait 5 seconds before next screenshot
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error in monitoring loop: {e}")
    
    def cleanup_old_screenshots(self):
        """Remove old screenshots to save disk space"""
        try:
            screenshots = [f for f in os.listdir(self.data_dir) if f.startswith('screenshot_') and f.endswith('.png')]
            screenshots.sort()
            
            # Keep only the last 10 screenshots
            if len(screenshots) > 10:
                for old_screenshot in screenshots[:-10]:
                    os.remove(os.path.join(self.data_dir, old_screenshot))
        except Exception as e:
            print(f"Error cleaning up screenshots: {e}")
    
    def start_monitoring(self):
        """Start the monitoring process"""
        if not self.region:
            print("❌ No region selected!")
            return False
        
        if not self.webhook_url:
            print("❌ No webhook URL configured!")
            return False
        
        self.monitoring = True
        
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=self.monitor_region)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.monitoring = False
        print("🛑 Monitoring stopped")

def main():
    print("🚀 Region Detector - Code Change Monitor")
    print("=" * 50)
    
    detector = RegionDetector()
    
    try:
        # Step 1: Select region
        print("📍 Step 1: Select the region to monitor")
        if not detector.select_region():
            print("❌ Region selection cancelled or failed")
            return
        
        print(f"✅ Region selected: {detector.region}")
        
        # Step 2: Setup Discord webhook
        print("\n🔗 Step 2: Configure Discord webhook")
        if not detector.setup_webhook():
            print("❌ Webhook setup cancelled or failed")
            return
        
        print("✅ Discord webhook configured")
        
        # Step 3: Start monitoring
        print("\n🔍 Step 3: Starting monitoring...")
        if detector.start_monitoring():
            print("✅ Monitoring started successfully!")
            
            # Keep the main thread alive
            try:
                while detector.monitoring:
                    time.sleep(1)
            except KeyboardInterrupt:
                detector.stop_monitoring()
        else:
            print("❌ Failed to start monitoring")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
