import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import pandas as pd
import glob
import os
import cv2 as cv
import numpy as np
import ntpath
import yaml
# Parse the XML file

def iou(bbox1, bbox2):
    """
    args: 
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    
    output:
    iou: (float)
    """
    iou_c = 0.0
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    if poly2.intersects(poly1):
        poly3 = poly2.intersection(poly1)

        Ar1 = float(poly1.area)
        Ar2 = float(poly2.area)
        Ar_of_int = float(poly3.area)

        iou_c = Ar_of_int / (Ar1 + Ar2 - Ar_of_int)

    return iou_c

def union(bbox1, bbox2):
    """
    args: 
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    
    output:
    iou: (float)
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    poly3 = poly2.union(poly1)

    return Polygon(poly3)

def parse_xml_boxes(file_path):

    tree = ET.parse(file_path)
    polygons = []
    # Get the root element
    root = tree.getroot()
    # Extract and print 'pt' tags

    for elements in root.findall(".//polygon"):
        # print(elements)
        trap = []
        for pt_element in elements.findall("pt"):
            coord = []
            for points in pt_element:
                trap.append(float(points.text))
            # trap.append([coord[0], coord[1]])
        polygons.append(trap)

    return polygons


def get_bounding_boxes(text_file_path):

    df = pd.read_csv(text_file_path, sep=" ", names=['class_id', 'x', 'y', 'w', 'h'])
    motorcycle = df.loc[df['class_id']==3]
    rider = df.loc[df['class_id']==0]

    motorcycle_box = []
    rider_box = []
    for i in range(len(motorcycle)):
        motorcycle_box.append([motorcycle.iloc[i]['x'], motorcycle.iloc[i]['y'], motorcycle.iloc[i]['w'],motorcycle.iloc[i]['h']])
    for i in range(len(rider)):
        rider_box.append([rider.iloc[i]['x'], rider.iloc[i]['y'], rider.iloc[i]['w'], rider.iloc[i]['h']])



    return motorcycle_box, rider_box

def get_trap_box(trap_polygons, paired_box, size):
    max_iou = 0.0
    target_trap = []
    for trap in trap_polygons:
        motor_box = get_unnormed_boxes(paired_box[0], size)
        iou_c = iou([(trap[0], trap[1]), (trap[2], trap[3]) , (trap[4], trap[5]) , (trap[6], trap[7])], [(motor_box[0], motor_box[1]), (motor_box[2], motor_box[3]), (motor_box[4], motor_box[5]), (motor_box[6], motor_box[7])])
        if max_iou < iou_c:
            max_iou = iou_c
            target_trap = trap

    return target_trap

def get_unnormed_boxes(box, size):
    img_height = size[0]
    img_width = size[1]

    x, y, w, h = box[0], box[1], box[2], box[3]
    motor = [(x-w/2)*img_width, (y-h/2)*img_height, (x+w/2)*img_width, (y-h/2)*img_height, (x+w/2)*img_width, (y+h/2)*img_height, (x-w/2)*img_width, (y+h/2)*img_height]

    return motor

def convert_trap_points(motor_box, trap_box):
    trap_area = 0
    x_c = 0
    y_c = 0
    motorcycle = motor_box[0]
    x_left = 2000
    x_right = 0
    y_top = 2000
    y_bottom = 0
    for i in range(1, len(motor_box)):
        rider_box = motor_box[i]
        x_left = min(x_left, min(motorcycle[0], motor_box[i][0]))
        x_right = max(x_right, max(motorcycle[2], motor_box[i][2]))
        y_top = min(y_top, min(motorcycle[1], motor_box[i][1]))
        y_bottom = max(y_bottom, max(motorcycle[5], motor_box[i][5]))

    # for i in range(len(trap_box)):
    #     n = len(trap_box)-1
    #     x_i, y_i, x_i_1, y_i_1 = trap_box[i%n][0], trap_box[i%n][1], trap_box[(i+1)%n][0], trap_box[(i+1)%n][1]
    #     trap_area += x_i*y_i_1 - x_i_1*y_i
    #
    #     x_c += (x_i + x_i_1)*(x_i*y_i_1 - x_i_1*y_i)
    #     y_c += (y_i  + y_i_1)*(x_i*y_i_1 - x_i_1*y_i)
    #
    # x_c = x_c/(6*trap_area+0.000001)
    # y_c = y_c/(6*trap_area+0.000001)
    #
    # trap_area /= 2        # img_path = "/home/ur10/data/training_data_817Images/final_test_set M_R_H_NH_after_preprocessing/"
        # img_file =  files.replace('.txt', '.jpg')
        # img = cv.imread(img_file)
        # correctness_check(img, paired_boxes, test_trap_boxes)
    # o1, o2, o3, o4 =  trap_box[1][1] - motor_box[0][1], trap_box[2][1] - motor_box[1][1], trap_box[3][1] - motor_box[2][1], trap_box[0][1] - motor_box[3][1]
    # w = abs(motor_box[0][0] - motor_box[1][0])

    # return x_c, y_c, w, o1/2, o2/2, o3/2, o4/2

    return  x_left, y_top, x_right, y_top, x_right, y_bottom, x_left, y_bottom




def get_motor_rider_trap_intersections(files, threshold =0.02):
         file_name = ntpath.basename(files)
         xml_path  = "/home/ur10/data/training_data_817Images/Trapezium_instance_boxes/"
         img_path = "/home/ur10/data/training_data_817Images/final_test_set M_R_H_NH_after_preprocessing/"
         xml_file =  xml_path + file_name.replace('.txt', '.xml')
         img_file = img_path + file_name.replace('.txt', '.jpg')
         trap_polygons = parse_xml_boxes(xml_file)

         motorcyle_boxes, rider_boxes = get_bounding_boxes(files)
         paired_boxes = []
         trap_boxes = []
         test_trap_boxes = []
         img = cv.imread(img_file)
         size = img.shape
         for m_box in motorcyle_boxes:
            target_trap = []
            m_box_unnorm = get_unnormed_boxes(m_box, size)
            paired_box = [m_box]
            paired_box_unnorm = [m_box_unnorm]
            for r_box in rider_boxes:

                r_box_unnorm = get_unnormed_boxes(r_box, size)


                if (iou([(m_box_unnorm[0],m_box_unnorm[1]), (m_box_unnorm[2],m_box_unnorm[3]), (m_box_unnorm[4],m_box_unnorm[5]), (m_box_unnorm[6],m_box_unnorm[7])], [(r_box_unnorm[0],r_box_unnorm[1]), (r_box_unnorm[2],r_box_unnorm[3]), (r_box_unnorm[4],r_box_unnorm[5]), (r_box_unnorm[6],r_box_unnorm[7])]) > threshold) and len(paired_box) < 6:
                    paired_box.append(r_box)
                    paired_box_unnorm.append(r_box_unnorm)
            if len(trap_polygons) != 0:
                target_trap = get_trap_box(trap_polygons, paired_box, size)
            # print(target_trap)
            # if target_trap in trap_polygons: trap_polygons.remove(target_trap)
            if len(target_trap) != 0:
                target_trap_converted = convert_trap_points(paired_box_unnorm, target_trap)
                # target_trap_converted = convert_trap_points([[m_box_unnorm[0], m_box_unnorm[1]], [m_box_unnorm[2], m_box_unnorm[3]], [m_box_unnorm[4], m_box_unnorm[5]], [m_box_unnorm[6], m_box_unnorm[7]]], [[target_trap[0], target_trap[1]], [target_trap[2], target_trap[3]], [target_trap[4], target_trap[5]], [target_trap[6], target_trap[7]]])
                test_trap_boxes.append(target_trap)
                trap_boxes.append(target_trap_converted)
                paired_boxes.append(paired_box_unnorm) # CHnage this to paired_box for exactness to paper
         return trap_boxes, paired_boxes, test_trap_boxes # Changed the first return value from trap_boxes

def correctness_check(img, paired_boxes, trap_box):
        size = img.shape
        for i in range(len(paired_boxes)):

            color = list(np.random.random(size=3) * 256)
            motor_box = get_unnormed_boxes(paired_boxes[i][0], size)
            o1, o2, o3, o4 = trap_box[i][3], trap_box[i][4], trap_box[i][5], trap_box[i][6]
            # Uncomment this when you are considering offsets in the trap convert method trap_box[i] = [motor_box[0], motor_box[1] - o1/2, motor_box[2], motor_box[3]-o2/2, motor_box[4], motor_box[5]+o3/2, motor_box[6], motor_box[7]+o4/2]
            # trap_box[i] = [motor_box[0], trap_box[i][0]
            pts = np.array(
                [[trap_box[i][0], trap_box[i][1]], [trap_box[i][2], trap_box[i][3]], [trap_box[i][4], trap_box[i][5]],
                 [trap_box[i][6], trap_box[i][7]]], np.int32)
            cv.polylines(img, [pts], True, color, 5)
            for box in paired_boxes[i]:


                box = get_unnormed_boxes(box, size)
                cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 3)

        cv.imshow('Image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
def create_input_label(text_dir, data_dict):
    files = os.listdir(text_dir)
    sorted_files = sorted(files)
    empty_points = [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]] # Change the empty point[1] as well

    for files in glob.glob(text_dir + "/*.txt"):
        # Is files the real name?
        trap_boxes, paired_boxes, test_trap_boxes = get_motor_rider_trap_intersections(files)
        # Have a small visual check here to verify everything

        # img_path = "/home/ur10/data/training_data_817Images/final_test_set M_R_H_NH_after_preprocessing/"
        # img_file =  files.replace('.txt', '.jpg')
        # img = cv.imread(img_file)
        # correctness_check(img, paired_boxes, test_trap_boxes)
        # Check over

        for i in range(len(trap_boxes)):

            # continue
            if len(paired_boxes[i]) < 6:
                for j in range(6-len(paired_boxes[i])):
                    paired_boxes[i].append(empty_points[1])
            print('Adding data to dict')
            data_dict['input_data'].append(paired_boxes[i])
            # print(type(paired_boxes[i]))
            data_dict['output_labels'].append(trap_boxes[i])
            # print(type(trap_boxes[i]))

    return data_dict

def main():
    data_dict = {'input_data':[],
                 'output_labels': []
    }
    text_dir = "/home/ur10/data/training_data_817Images/final_test_set M_R_H_NH_after_preprocessing/"

    data_dict = create_input_label(text_dir, data_dict)

    df = pd.DataFrame(data_dict)
    text_file_path = 'data.csv'

    df.to_csv(text_file_path,',', index=False)

if __name__ == "__main__":
    main()