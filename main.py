import cv2
import pytesseract
import tkinter
# from tkinter import filedialog
# Import module
from tkinter import *
import requests
import PIL.Image, PIL.ImageTk
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog
import numpy as np


def importVideo():
    # faceDetector(mode='computerCamera')
    print('video')


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def automatic_brightness_and_contrast(image, clip_hist_percent=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def detect(img_rgb):
    copy_image = img_rgb.copy()
    input_height = img_rgb.shape[0]
    input_width = img_rgb.shape[1]
    hsv_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    #Print the hsv_frame
    # cv2.imshow("License Plate Detection", hsv_frame)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")

    # yellow color
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(yellow_mask, yellow_mask, mask=yellow_mask)

    # step 1
    # cv2.imshow("License Plate Detection", yellow)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")

    # close morph
    k = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

    # step 2
    # cv2.imshow("License Plate Detection", closing)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")

    # Detected yellow area
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # List of final crops
    crops = []

    # Loop over contours and find license plates
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Conditions on crops dimensions and area
        if h * 6 > w > 2 * h and h > 0.1 * w and w * h > input_height * input_width * 0.0001:
            # Make a crop from the RGB image, the crop is slided a bit at left to detect bleu area
            crop_img = img_rgb[y:y + h, x - round(w / 10):x]
            crop_img = crop_img.astype('uint8')
            # Compute bleu color density at the left of the crop
            # Bleu color condition
            try:
                hsv_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                #Print the hsv_frame
                # cv2.imshow("License Plate Detection", hsv_frame)
                # cv2.waitKey(0)
                # cv2.destroyWindow("License Plate Detection")
                low_bleu = np.array([100, 150, 0])
                high_bleu = np.array([140, 255, 255])
                bleu_mask = cv2.inRange(hsv_frame, low_bleu, high_bleu)
                bleu_summation = bleu_mask.sum()

            except:
                bleu_summation = 0

            # Condition on bleu color density at the left of the crop
            if bleu_summation > 550:

                # Compute yellow color density in the crop
                # Make a crop from the RGB image
                imgray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                crop_img_yellow = img_rgb[y:y + h, x:x + w]
                crop_img_yellow = crop_img_yellow.astype('uint8')

                # Detect yellow color
                hsv_frame = cv2.cvtColor(crop_img_yellow, cv2.COLOR_BGR2HSV)
                low_yellow = np.array([20, 100, 100])
                high_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)

                # Compute yellow density
                yellow_summation = yellow_mask.sum()

                # Condition on yellow color density in the crop
                if yellow_summation > 255 * crop_img.shape[0] * crop_img.shape[0] * 0.4:

                    # Make a crop from the gray image
                    crop_gray = imgray[y:y + h, x:x + w]
                    crop_gray = crop_gray.astype('uint8')

                    # Detect chars inside yellow crop with specefic dimension and area
                    th = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                               11, 2)
                    contours2, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # Init number of chars
                    chars = 0
                    for c in contours2:
                        area2 = cv2.contourArea(c)
                        x2, y2, w2, h2 = cv2.boundingRect(c)
                        if w2 * h2 > h * w * 0.01 and h2 > w2 and area2 < h * w * 0.9:
                            chars += 1

                    # Condition on the number of chars
                    if 20 > chars > 4:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        pts = np.array(box)
                        warped = four_point_transform(copy_image, pts)
                        crops.append(warped)

                        # Using cv2.putText() method
                        img_rgb = cv2.putText(img_rgb, 'License Plate', (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                              (0, 255, 255), 2, cv2.LINE_AA)

                        cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)

    return img_rgb, crops


def process(src):
    # Brigthness and contrast adjustment
    # cv2.imwrite("temp/steps/3_detected_plate.png", src)
    #step 3
    # cv2.imshow("License Plate Detection", src)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")

    adjusted, a, b = automatic_brightness_and_contrast(src)
    #Step 4
    # cv2.imshow("License Plate Detection", adjusted)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")

    # cv2.imwrite("temp/steps/4_Brigthness_contrast_adjustment.png", adjusted)
    # BGR to gray
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    # step 5
    # cv2.imshow("License Plate Detection", gray)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")

    # cv2.imwrite("temp/steps/5_gray.png", gray)
    # Binary thresh
    # ret, th = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #Step 6
    # cv2.imshow("License Plate Detection", th)
    # cv2.waitKey(0)
    # cv2.destroyWindow("License Plate Detection")
    # cv2.imwrite("temp/steps/6_threshold.png", th)
    return th


def test():
    path = '/Users/orkalev/Desktop/cars/2.jpg'
    image = cv2.imread(path)
    #img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detection, crops = detect(image)
    crop = process(crops[0])

    cv2.imshow("License Plate Detection", detection)
    cv2.waitKey(0)
    cv2.destroyWindow("License Plate Detection")


def importImage():
    # open a file chooser dialog and allow the user to select an input image
    path = tkFileDialog.askopenfilename()
    #path = '/Users/orkalev/Desktop/cars/2.jpg'
    if len(path) > 0:
        # Read the image file
        image = cv2.imread(path)
        detection, crops = detect(image)
        i = 1
        for crop in crops:
            crop = process(crop)
            cv2.imshow("License Plate Detection", crop)
            cv2.waitKey(0)
            cv2.destroyWindow("License Plate Detection")
            # cv2.imwrite('temp/crop' + str(i) + '.jpg', crop)
            # text = pytesseract.image_to_string(Image.open('temp/crop1.jpg'))
            text = pytesseract.image_to_string(crop, lang='eng', config='--psm 6')
            print(text)
            i += 1
        # cv2.imwrite('temp/detection.jpg', detection)

        # # Convert to Grayscale Image
        # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # # Canny Edge Detection
        # canny_edge = cv2.Canny(gray_image, 170, 200)
        #
        # # Find contours based on Edges
        # contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        #
        # # Initialize license Plate contour and x,y coordinates
        # contour_with_license_plate = None
        # license_plate = None
        # x = None
        # y = None
        # w = None
        # h = None
        #
        # # Find the contour with 4 potential corners and creat ROI around it
        # for contour in contours:
        #     # Find Perimeter of contour and it should be a closed contour
        #     perimeter = cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        #     if len(approx) == 4:  # see whether it is a Rect
        #         contour_with_license_plate = approx
        #         x, y, w, h = cv2.boundingRect(contour)
        #         license_plate = gray_image[y:y + h, x:x + w]
        #         break
        #
        # # Removing Noise from the detected image, before sending to Tesseract
        # #cv2.imshow("License Plate Detection", license_plate)
        # license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
        # (thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)
        #
        #
        #
        # # Text Recognition
        # #text = pytesseract.image_to_string(license_plate)
        # text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 6')
        # #text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 6')
        # #text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        #
        #
        # # Draw License Plate and write the Text
        # image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # carImageAfterDetection = cv2.putText(image, text, (x - 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)
        # #cv2.createButton("Get Car Info", carInfo, None, cv2.QT_PUSH_BUTTON, 1)
        #
        # print("License Plate :", text)

        # # Load a image using OpenCV
        # cv_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #
        # #Create new window for the Car Detection
        # windownCarDetection = Tk()
        #
        # #Get the image dimensions
        # height, width, no_channels = cv_img.shape
        #
        # #Create a canvas that can fit the aboce image
        # canvasCarDetecation = Canvas(windownCarDetection, width = width, height = height)
        # canvasCarDetecation.pack(fill="both", expand=True)
        #
        # # Use PIl (Pillow) to convert the NumPy ndaaray to a PhotoImage
        # #photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
        #
        # #Add a PhotoImage to the Canvas
        # canvasCarDetecation.create_image(0, 0, image=cv_img, anchor='nw')
        #
        # windownCarDetection.mainloop()

        #
        # # Create object
        # rootCarDetected = Tk()
        #
        # # Adjust size
        # rootCarDetected.geometry("510x400")
        #
        # rootCarDetected.title('Licence Detection')
        #
        # # # Add image file
        # # bg = PhotoImage(file="test2.png")
        #
        # # Create Canvas after find the number
        # canvasNumberFound = Canvas(root, width=510,
        #                        height=400)
        #
        # canvasNumberFound.pack(fill="both", expand=True)
        #
        # # Display image
        # canvasNumberFound.create_image(0, 0, image=image,
        #                      anchor="nw")
        #
        # # Add Text
        # canvasNumberFound.create_text(280, 30, text="Car number found")
        #
        # # Create Buttons
        # getCarInfo = tkinter.Button(root, text="Get Car Info", command=carInfo)
        #
        #
        # # Display Buttons
        # getCarInfo_canvas = canvasNumberFound.create_window(30, 10,
        #                                                  anchor="nw",
        #                                                  window=getCarInfo)
        #
        # root.importImage()
        # print(text)
        cv2.imshow("License Plate Detection", detection)
        cv2.waitKey(0)
        cv2.destroyWindow("License Plate Detection")

        # Get the data from API source
        payload = {'resource_id': '053cea08-09bc-40ec-8f7a-156f0677aff3', 'q': '5455354'}
        r = requests.get('https://data.gov.il/api/3/action/datastore_search', params=payload)
        res = r.json()
        record1 = res['result']['records']

        record = record1[0]
        mispar_rechev = record["mispar_rechev"]
        tozeret_cd = record["tozeret_cd"]
        tozeret_nm = record["tozeret_nm"]
        degem_nm = record["degem_nm"]
        ramat_gimur = record["ramat_gimur"]
        ramat_eivzur_betihuty = record["ramat_eivzur_betihuty"]
        kvutzat_zihum = record["kvutzat_zihum"]
        shnat_yitzur = record["shnat_yitzur"]
        degem_manoa = record["degem_manoa"]
        mivchan_acharon_dt = record["mivchan_acharon_dt"]
        tokef_dt = record["tokef_dt"]
        baalut = record["baalut"]
        misgeret = record["misgeret"]
        tzeva_rechev = record["tzeva_rechev"]
        zmig_kidmi = record["zmig_kidmi"]
        zmig_ahori = record["zmig_ahori"]
        sug_delek_nm = record["sug_delek_nm"]
        horaat_rishum = record["horaat_rishum"]
        kinuy_mishari = record["kinuy_mishari"]
        canvas1.create_text(730, 50, text=str(mispar_rechev))
        canvas1.create_text(770, 70, text=tozeret_nm)
        canvas1.create_text(710, 90, text=ramat_gimur)
        canvas1.create_text(760, 110, text=ramat_eivzur_betihuty)
        canvas1.create_text(750, 130, text=shnat_yitzur)
        canvas1.create_text(820, 150, text=mivchan_acharon_dt)
        canvas1.create_text(820, 170, text=tokef_dt)
        canvas1.create_text(740, 190, text=baalut)
        canvas1.create_text(790, 210, text=misgeret)
        canvas1.create_text(730, 230, text=tzeva_rechev)
        canvas1.create_text(740, 250, text=sug_delek_nm)
        canvas1.create_text(740, 270, text=kinuy_mishari)

        canvas1.update()
        return mispar_rechev


def carInfo():
    print('blablabla')


def exitUI():
    # faceDetector(mode='videoFile')
    print('exit')


# Create object
root = Tk()

# Adjust size
root.geometry("900x400")

root.title('Licence Detection')

# Add image file
bg = PhotoImage(file="test2.png")

# Create Canvas
canvas1 = Canvas(root, width=510,
                 height=400)

canvas1.pack(fill="both", expand=True)

# Display image
canvas1.create_image(0, 0, image=bg,
                     anchor="nw")

# Add Text
canvas1.create_text(280, 30, text="Welcome to the licence detector")

# Add the field text
canvas1.create_text(660, 50, text="Car Number: ")
canvas1.create_text(660, 70, text="Manufacturer country:")
canvas1.create_text(660, 90, text="Level:")
canvas1.create_text(660, 110, text="Fitting safety level:")
canvas1.create_text(660, 130, text="Production year:")
canvas1.create_text(650, 150, text="Last vehicle licensing test:")
canvas1.create_text(650, 170, text="Next vehicle licensing test:")
canvas1.create_text(660, 190, text="Current ownership:")
canvas1.create_text(650, 210, text="Car build number:")
canvas1.create_text(660, 230, text="Color:")
canvas1.create_text(660, 250, text="Fuel type:")
canvas1.create_text(660, 270, text="Trade alias:")

# Create Buttons
importVideoButton = tkinter.Button(root, text="Import Video", command=importVideo)
importImageButton = tkinter.Button(root,text= "Import Image",command=importImage)
#importImageButton = tkinter.Button(root, text="Import Image", command=test)
exitButton = tkinter.Button(root, text="Exit", command=exitUI)

# Display Buttons
importVideoButton_canvas = canvas1.create_window(30, 10,
                                                 anchor="nw",
                                                 window=importVideoButton)

importImageButton_canvas = canvas1.create_window(30, 40,
                                                 anchor="nw",
                                                 window=importImageButton)

exitButton_canvas = canvas1.create_window(30, 70, anchor="nw",
                                          window=exitButton)

# Execute tkinter
root.mainloop()
