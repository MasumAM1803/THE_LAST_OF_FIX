import cv2
import matplotlib.pyplot as plt
import math


# ======================== FEATURE EXTRACTION ===========================
def invariantMomentHu(image):
    height = image.shape[0]
    width = image.shape[1]
    height *= 2
    width *= 2
    image = cv2.resize(image, (width, height))
    image2 = image

    # === PROCESS FEATURE EXTRACTION ===

    # FEATURE EXTRACTION
    image3, contour, hierarki = cv2.findContours(image,
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # invarian moment Hu
    mom = cv2.moments(contour[0])
    humoment = cv2.HuMoments(mom)
    # PROOF
    # print(humoment)

    feature = []
    # USING logaritma
    for i in range(0, 7):
        if humoment[i] == 0:
            humoment[i] = 0
        else:
            humoment[i] = -1 * math.copysign(1.0, humoment[i]) * \
                          math.log10(abs(humoment[i]))
        # PROOF
        # print(humoment[i][0])
        feature.append(humoment[i][0])


    # PROOF
    # print("Feature")
    # print(feature)

    return feature


def start(file_name):
    image = cv2.imread(file_name, 0)
    feature = invariantMomentHu(image)

    return [feature]

# TRY THIS CODE
def main():
    image = cv2.imread('./dataset/nyoba/0/150 (2).jpg17.png', 0)
    feature = invariantMomentHu(image)
    #print(feature[6])
    return

# FOR TRYING CODE
#main()
