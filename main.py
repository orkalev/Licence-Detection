import cv2
import pytesseract
import tkinter
#from tkinter import filedialog
# Import module
from tkinter import *
import requests
import PIL.Image, PIL.ImageTk
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog as tkFileDialog





def importVideo():
   # faceDetector(mode='computerCamera')
    print('video')


def importImage():
    # open a file chooser dialog and allow the user to select an input image
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        # Read the image file
        image = cv2.imread(path)
        # Convert to Grayscale Image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Canny Edge Detection
        canny_edge = cv2.Canny(gray_image, 170, 200)

        # Find contours based on Edges
        contours, new = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        # Initialize license Plate contour and x,y coordinates
        contour_with_license_plate = None
        license_plate = None
        x = None
        y = None
        w = None
        h = None

        # Find the contour with 4 potential corners and creat ROI around it
        for contour in contours:
            # Find Perimeter of contour and it should be a closed contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            if len(approx) == 4:  # see whether it is a Rect
                contour_with_license_plate = approx
                x, y, w, h = cv2.boundingRect(contour)
                license_plate = gray_image[y:y + h, x:x + w]
                break

        # Removing Noise from the detected image, before sending to Tesseract
        #cv2.imshow("License Plate Detection", license_plate)
        license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
        (thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)



        # Text Recognition
        #text = pytesseract.image_to_string(license_plate)
        text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 6')
        #text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 6')
        #text = pytesseract.image_to_string(license_plate, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')


        # Draw License Plate and write the Text
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        carImageAfterDetection = cv2.putText(image, text, (x - 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)
        #cv2.createButton("Get Car Info", carInfo, None, cv2.QT_PUSH_BUTTON, 1)

        print("License Plate :", text)

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

        cv2.imshow("License Plate Detection", image)
        cv2.waitKey(0)
        cv2.destroyWindow("License Plate Detection")

        #Get the data from API source
        payload = {'resource_id': '053cea08-09bc-40ec-8f7a-156f0677aff3', 'q': '4232112'}
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
        canvas1.create_text(700, 90, text=ramat_gimur)
        canvas1.create_text(760, 110, text=ramat_eivzur_betihuty)
        canvas1.create_text(750, 130, text=shnat_yitzur)
        canvas1.create_text(820, 150, text=mivchan_acharon_dt)
        canvas1.create_text(820, 170, text=tokef_dt)
        canvas1.create_text(740, 190, text= baalut)
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
bg = PhotoImage(file = "test2.png")

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
importVideoButton = tkinter.Button(root,text= "Import Video",command=importVideo)
importImageButton = tkinter.Button(root,text= "Import Image",command=importImage)
exitButton = tkinter.Button(root,text= "Exit",command=exitUI)


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

