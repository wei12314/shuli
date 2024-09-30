import os
from PIL import Image

# convert
def convert(dw, dh, coord):
    
    # 计算中心点坐标和宽高
    # coord = [x, y, w, h]
    # l_x = coord[0], r_x = coord[0] + coord[2] 
    # mid_x = (l_x + r_x) / 2 -> coord[0] + coord[2] / 2.0
    x_center = coord[0] + coord[2] / 2.0
    y_center = coord[1] + coord[3] / 2.0
    width = coord[2]
    height = coord[3]
    
    # 归一化并保留六位小数
    x_center = round(x_center * dw, 6)
    width = round(width * dw, 6)
    y_center = round(y_center * dh, 6)
    height = round(height * dh, 6)
    
    return [x_center, y_center, width, height]

def txt_to_yolo(img_path, coordinates, output_path):
    img = Image.open(img_path)
    write_lines = []
    size = img.size
    dw = 1. / size[0]
    dh = 1. / size[1]
    for coord in coordinates:
        l = coord.split(' ')
        x, y, w, h = int(l[1]), int(l[2]), int(l[3]), int(l[4])
        coordinates = [x, y, w, h]
        res = convert(dw, dh, coordinates)
        res_s = [str(res[0]), str(res[1]), str(res[2]), str(res[3])]
        s = " ".join(res_s)
        line = f"0 {s}"
        write_lines.append(line)
    
    for line in write_lines:
        with open(output_path, 'a+', encoding='utf-8') as f:
            f.write(line + "\n")


if __name__ == "__main__":
    base_img_path = "compress_img"
    txt_base_path = "need_check_txt/shuidao"
    txt_to_yolo_base = "txt_to_yolo/shuidao"
    for txt_name in os.listdir(txt_base_path):
        coordinates = None
        img_name = txt_name.replace(".txt", ".jpg")
        img_path = os.path.join(base_img_path, img_name)
        txt_path = os.path.join(txt_base_path, txt_name)
        txt_to_yolo_path = os.path.join(txt_to_yolo_base, txt_name)
        with open(txt_path, 'r', encoding='utf-8') as f:
            coordinates = f.readlines()
        txt_to_yolo(img_path, coordinates, txt_to_yolo_path)
