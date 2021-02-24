import cv2
import pytesseract
import tkinter
from tkinter import *
import requests
import tkinter.filedialog as tkFileDialog
import numpy as np
from random import choice


def filter_contrast(image):
    contrastPrsent = 10
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscaleHistogram = cv2.calcHist([grayImage], [0], None, [256], [0, 256]) # Get the grayscale histogram
    acc = []
    acc.append(float(grayscaleHistogram[0]))
    for index in range(1, len(grayscaleHistogram)):
        acc.append(acc[index - 1] + float(grayscaleHistogram[index]))

    contrastPrsent *= (acc[-1] / 100.0)
    contrastPrsent /= 2.0

    minGray = 0
    while acc[minGray] < contrastPrsent:
        minGray += 1

    maxGray = len(grayscaleHistogram) - 1
    while acc[maxGray] >= (acc[-1] - contrastPrsent):
        maxGray -= 1

    return cv2.convertScaleAbs(image, alpha=255 / (maxGray - minGray), beta=-minGray * 255 / (maxGray - minGray))


def detect_plate(originalImage):
    imageHeight, imageWidth, c = originalImage.shape
    copyImage = originalImage.copy()       # Copy Image
    hsvColorImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV) # RGB -> HSV (for yellow sepration )
    yellowImage = cv2.inRange(hsvColorImage, np.array([17, 90, 90]), np.array([30, 255, 255]))      # get all range from low to high
    yellowGrayImage = cv2.bitwise_and(yellowImage, yellowImage, mask=yellowImage)   # bit wise and to transporm to gray image
    k = np.ones((5, 5), np.uint8)      #Creat structer element

    # Double closing to the image
    closingMorpho = cv2.morphologyEx(yellowGrayImage, cv2.MORPH_CLOSE, k)   # Fill litel holes using morphology close opration
    closingMorpho = cv2.morphologyEx(closingMorpho, cv2.MORPH_CLOSE, k)

    # Detected yellow area
    contours, her = cv2.findContours(closingMorpho, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # contours aka claster

    # Loop over contours and find license plates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)     # find the rect of the shape

        # Conditions on crops dimensions and area
        if h * 6 > w > 2 * h and h > 0.1 * w and w * h > imageHeight * imageWidth * 0.0001:        #check the size of the rect
            cropImage = originalImage[y:y + h, x - round(w / 10):x]    # crop the plant
            cropImage = cropImage.astype('uint8')

            # Compute yellow color density in the crop
            # Make a crop from the RGB image
            imgray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            cropImageYellow = originalImage[y:y + h, x:x + w]
            cropImageYellow = cropImageYellow.astype('uint8')

            # Detect yellow color
            hsvColorImage = cv2.cvtColor(cropImageYellow, cv2.COLOR_BGR2HSV)
            yellowImage = cv2.inRange(hsvColorImage, np.array([20, 100, 100]), np.array([30, 255, 255]))

            # Condition on yellow color density in the crop
            if yellowImage.sum() > 255 * cropImage.shape[0] * cropImage.shape[0] * 0.4:

                # Make a crop from the gray image
                corpImageGray = imgray[y:y + h, x:x + w]
                corpImageGray = corpImageGray.astype('uint8')

                # At this point we know that the crop image is the yellow plant


                # Detect chars inside yellow crop with specefic dimension and area
                th = cv2.adaptiveThreshold(corpImageGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           11, 2)   # make a mask(black and white) img
                                                    # from the croped yellow plate
                contours2, her = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                #find contoures like before
                #then run for each contour in contours and try to match bounding box for each letter

                # Init number of chars
                chars = 0
                for c in contours2:
                    area2 = cv2.contourArea(c)
                    x2, y2, w2, h2 = cv2.boundingRect(c)
                    if w2 * h2 > h * w * 0.01 and h2 > w2 and area2 < h * w * 0.9:
                        chars += 1

                # Condition on the number of chars
                if 20 > chars > 4:
                    box = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
                    pts = np.array(box)
                    # Order the rect corners
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]
                    rect[2] = pts[np.argmax(s)]
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]
                    rect[3] = pts[np.argmax(diff)]

                    # Transform the points
                    (tl, tr, br, bl) = rect
                    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                    maxWidth = max(int(widthA), int(widthB))
                    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                    maxHeight = max(int(heightA), int(heightB))
                    dst = np.array([
                        [0, 0],
                        [maxWidth - 1, 0],
                        [maxWidth - 1, maxHeight - 1],
                        [0, maxHeight - 1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(rect, dst)
                    adjusted = cv2.warpPerspective(copyImage, M, (maxWidth, maxHeight))

                    #adjusted = filter_contrast(four_point_transform(copyImage, pts))
                    plate = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
                    licenseNum = pytesseract.image_to_string(plate, config='--psm 13 -c tessedit_char_whitelist=0123456789')
                    # Put the license number on the photo
                    originalImage = cv2.putText(originalImage, licenseNum , (x - 100, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                          (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.drawContours(originalImage, [box], 0, (0, 0, 255), 2)


    return originalImage, licenseNum

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
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result, alpha, beta


def detect(img_rgb):

    img = img_rgb.copy()
    input_height = img_rgb.shape[0]
    input_width = img_rgb.shape[1]
    hsv_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

    # yellow color
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)
    yellow = cv2.bitwise_and(yellow_mask, yellow_mask, mask=yellow_mask)

    cv2.imwrite("temp/steps/1_yellow_color_detection.png", yellow)
    # Close morph
    k = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k)

    cv2.imwrite("temp/steps/2_closing_morphology.png", closing)
    # Detect yellow area
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List of final crops
    crops = []

    # Loop over contours and find license plates
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Conditions on crops dimensions and area
        if h*6 > w > 2 * h and h > 0.1 * w and w * h > input_height * input_width * 0.0001:

            # Make a crop from the RGB image, the crop is slided a bit at left to detect bleu area
            crop_img = img_rgb[y:y + h, x-round(w/10):x]
            crop_img = crop_img.astype('uint8')

            # Compute bleu color density at the left of the crop
            # Bleu color condition
            try:
                hsv_frame = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
                low_bleu = np.array([100,150,0])
                high_bleu = np.array([140,255,255])
                bleu_mask = cv2.inRange(hsv_frame, low_bleu, high_bleu)
                bleu_summation = bleu_mask.sum()

            except:
                bleu_summation = 0

            # Condition on bleu color density at the left of the crop
            if bleu_summation > 550:

                # Compute yellow color density in the crop
                # Make a crop from the RGB image
                imgray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                crop_img_yellow = img_rgb[y:y + h, x:x+w]
                crop_img_yellow = crop_img_yellow.astype('uint8')

                # Detect yellow color
                hsv_frame = cv2.cvtColor(crop_img_yellow, cv2.COLOR_BGR2HSV)
                low_yellow = np.array([20, 100, 100])
                high_yellow = np.array([30, 255, 255])
                yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)

                # Compute yellow density
                yellow_summation = yellow_mask.sum()

                # Condition on yellow color density in the crop
                if yellow_summation > 255*crop_img.shape[0]*crop_img.shape[0]*0.4:

                    # Make a crop from the gray image
                    crop_gray = imgray[y:y + h, x:x + w]
                    crop_gray = crop_gray.astype('uint8')

                    # Detect chars inside yellow crop with specefic dimension and area
                    th = cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
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
                        warped = four_point_transform(img, pts)
                        crops.append(warped)

                        # Using cv2.putText() method
                        img_rgb = cv2.putText(img_rgb, 'License Plate', (x-20, y), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 2, cv2.LINE_AA)

                        cv2.drawContours(img_rgb, [box], 0, (0, 0, 255), 2)

    return img_rgb, crops


def importVideo():
    path = tkFileDialog.askopenfilename()
    cap = cv2.VideoCapture(path)

    if cap.isOpened() == False:
        print("Error opening video stream or file")

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame, crop = detect(frame)
            # Display the resulting frame

            cv2.putText(frame, 'Press \'Q\' to exit !',(50, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 255), 2)
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()



def importImage():
    # open a file chooser dialog and allow the user to select an input image
    canvas1.delete("CarInfo")
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        # Read the image file
        image = cv2.imread(path)
        cv2.imshow("The car before detection", image)
        cv2.waitKey(0)
        cv2.destroyWindow("The car before detection")

        detection, licenseNum = detect_plate(image)
        print(licenseNum)

        cv2.imshow("The car after detection", detection)
        cv2.waitKey(0)
        cv2.destroyWindow("The car after detection")

        # Get the data from API source
        payload = {'resource_id': '053cea08-09bc-40ec-8f7a-156f0677aff3', 'q': licenseNum}
        r = requests.get('https://data.gov.il/api/3/action/datastore_search', params=payload)
        res = r.json()
        record1 = res['result']['records']
        if len(record1) == 0:
            print("The car " + licenseNum + "is not at the data set")
        else:
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
            canvas1.create_text(740, 50, text=str(mispar_rechev), anchor="w", tag="CarInfo")
            canvas1.create_text(740, 70, text=tozeret_nm, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 90, text=ramat_gimur, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 110, text=ramat_eivzur_betihuty, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 130, text=shnat_yitzur, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 150, text=mivchan_acharon_dt, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 170, text=tokef_dt, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 190, text=baalut, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 210, text=misgeret, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 230, text=tzeva_rechev, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 250, text=sug_delek_nm, anchor="w", tag="CarInfo")
            canvas1.create_text(740, 270, text=kinuy_mishari, anchor="w", tag="CarInfo")

            canvas1.update()
        return




def exitUI():
    exit(0)

# Create object
root = Tk()

# Adjust size
root.geometry("900x400")

root.title('Car Scanner')

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
canvas1.create_text(320, 30, text="Welcome to the car scanner", font=('helvetica', 18, 'bold'))

# Add the field text
canvas1.create_text(550, 50, text="Car Number: ", anchor="w")
canvas1.create_text(550, 70, text="Manufacturer country:", anchor="w")
canvas1.create_text(550, 90, text="Level:", anchor="w")
canvas1.create_text(550, 110, text="Fitting safety level:", anchor="w")
canvas1.create_text(550, 130, text="Production year:", anchor="w")
canvas1.create_text(550, 150, text="Last vehicle licensing test:", anchor="w")
canvas1.create_text(550, 170, text="Next vehicle licensing test:", anchor="w")
canvas1.create_text(550, 190, text="Current ownership:", anchor="w")
canvas1.create_text(550, 210, text="Car build number:", anchor="w")
canvas1.create_text(550, 230, text="Color:", anchor="w")
canvas1.create_text(550, 250, text="Fuel type:", anchor="w")
canvas1.create_text(550, 270, text="Trade alias:", anchor="w")

# Create Buttons
importVideoButton = tkinter.Button(root, text="Import Video", command=importVideo,fg='blue',height=2, width= 12)
importImageButton = tkinter.Button(root, text= "Import Image", command=importImage,fg='blue', height=2, width= 12)
#importImageButton = tkinter.Button(root, text="Import Image", command=test)
exitButton = tkinter.Button(root, text="Exit", command=exitUI, fg='red', height=2, width= 5)

# Display Buttons
importVideoButton_canvas = canvas1.create_window(30, 10,
                                                 anchor="nw",
                                                 window=importVideoButton)

importImageButton_canvas = canvas1.create_window(30, 50,
                                                 anchor="nw",
                                                 window=importImageButton)

exitButton_canvas = canvas1.create_window(30, 90, anchor="nw",
                                          window=exitButton)

# Execute tkinter
root.mainloop()
