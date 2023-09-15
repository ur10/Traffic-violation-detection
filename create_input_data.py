import os
from tkinter import image_names
import cv2 as cv
import numpy as np

src_path = "/home/ur10/data/training_data_817Images/final_test_set M_R_H_NH_after_preprocessing/"
target_path = "/home/ur10/continual_data/motor_rider_data_CL/"
files = os.listdir(src_path)
sorted_files = sorted(files)


# TODO - get the image bounding box in the correct shape
# Get the Intersection between the motorcycle and the rider
# Store the points as input data

def read_annotation():
    motor_boxes = []
    rider_boxes = []
    for filename in sorted_files:

        if filename.endswith('.txt'):
            image_name = filename.replace('.txt', '.jpg')
            with open(src_path + filename) as file:
                # print('hree')
                if os.path.getsize(src_path + filename) <= 0:
                    print("No annotation")

                img = cv.imread(src_path + image_name)
                height, width = img.shape[0], img.shape[1]
                m_box = []
                r_box = []
                string = []
                for line in file:
                    clas, x, y, w, h = map(float, line.split(' '))

                    x1 = int((x - w / 2) * width)
                    y1 = int((y - h / 2) * height)
                    x2 = int((x + w / 2) * width)
                    y2 = int((y + h / 2) * height)

                    modified_line = ""
                    if clas == 0:  # rider class
                        r_box.append([x1, y1, x2, y2])
                        modified_line = '0' + line[1:]
                        # cv.rectangle(img, (x1,y1 ), (x2, y2), (0, 0, 255), 1)
                    elif clas == 3:  # moto class
                        m_box.append([x1, y1, x2, y2])
                        modified_line = '1' + line[1:]
                        # cv.rectangle(img, (x1,y1 ), (x2, y2), (0, 255, 255), 1)

                    string.append(modified_line)

            with open(target_path + filename, 'w') as file:
                for line in string:
                    file.write(line)
                # cv.imshow('Imaeg',img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
                # motor_boxes.append(m_box)
                # rider_boxes.append(r_box)
                # input_boxes = get_motor_rider_intersection(motor_boxes, rider_boxes)
                # for pair_box in input_boxes:
                #     color = list(np.random.random(size=3) * 256)
                #     for box in pair_box:
                #         cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)
                # cv.imshow('Imaeg',img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
    return motor_boxes, rider_boxes


def check_intersection(box1, box2):
    x1, y1, width1, height1 = box1[0], box1[1], abs(box1[2] - box1[0]), abs(box1[3] - box1[1])
    x2, y2, width2, height2 = box2[0], box2[1], abs(box2[2] - box2[0]), abs(box2[3] - box2[1])

    # Calculate the coordinates of the corners of the boxes
    top_left1 = (x1, y1)
    top_right1 = (x1 + width1, y1)
    bottom_left1 = (x1, y1 + height1)
    bottom_right1 = (x1 + width1, y1 + height1)

    top_left2 = (x2, y2)
    top_right2 = (x2 + width2, y2)
    bottom_left2 = (x2, y2 + height2)
    bottom_right2 = (x2 + width2, y2 + height2)

    if (bottom_left1[1] < top_left2[1]) or (bottom_right1[1] > top_left2[1]):
        return False
    if (bottom_right1[0] < top_left2[0]) or (bottom_left1[0] > top_right2[0]):
        return False
    if (bottom_left2[1] < top_left1[1]) or (bottom_right2[1] > top_left1[1]):
        return False
    if (bottom_right2[0] < top_left1[0]) or (bottom_left2[0] > top_right1[0]):
        return False

    return True

    # # Check for intersection
    # if (top_left1[0] <= bottom_right2[0] and bottom_right1[0] >= top_left2[0] and
    #     top_left1[1] <= bottom_right2[1] and bottom_right1[1] >= top_left2[1]):
    #     return True  # Boxes intersect
    # else:
    #     return False  # Boxes do not intersect


def get_motor_rider_intersection(motor_boxes, rider_boxes):
    input_boxes = []

    for m_box in motor_boxes:
        pair_boxes = [m_box]
        for r_box in rider_boxes:
            if check_intersection(m_box, r_box):
                if len(pair_boxes) < 5:
                    pair_boxes.append(r_box)
        input_boxes.append(pair_boxes)

    return input_boxes


def get_input_data(input_boxes, width, height):
    input_data = []

    for pair_boxes in input_boxes:
        pair_data = []
        for box in pair_boxes:
            x_c = (box[0] + box[2]) / (2 * width)
            y_c = (box[1] + box[3]) / (2 * height)
            w_m = abs(box[0] - box[1]) / width
            h_m = abs(box[1] - box[3]) / height

            pair_data.append([x_c, y_c, w_m, h_m])
        input_data.append(pair_data)
    return input_data


def main():
    motor_boxes, rider_boxes = read_annotation()
    input_boxes = get_motor_rider_intersection(motor_boxes, rider_boxes)


if __name__ == "__main__":
    main()

# count = count + 1
# print(len(empty_files))
