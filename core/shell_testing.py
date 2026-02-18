import cv2
import pytesseract

# Set the location of document
img_path = r"C:\chatbot\ask_me\media\documents\Screenshot_174.png"

img = cv2.imread(img_path)

if img is None:
    print("❌ Image not loading")
else:
    print("✅ Image loaded")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)

    # Resize larger
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Use OTSU instead of adaptive
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh, config="--oem 3 --psm 6 -l eng")

    print("\n========= OCR RESULT =========")
    print(text)
    print("==============================")
