import os
import time
import hmac
import hashlib
import base64
import urllib.parse
import requests
import json
import cv2
from PIL import Image, ImageEnhance

# "https://img.picui.cn/free/2024/09/26/66f526dbdf6ef.jpg"
'''output : [{"x":856,"y":874,"w":103,"h":114,"score":83},{"x":1067,"y":1500,"w":136,"h":174,"score":73}]'''
def get_from_lingmou(file_link, mode_type=205): # get img url return output obj
    app_key = 'b441c7e37b6aed09d242bcc382f076dc'
    app_secret = '7c419ba5a726f48e2e2a958396408c6b064cbddf254dfd8160fad650ce760bd0'

    timestamp = str(round(time.time() * 1000))
    accessSecret = app_secret
    secret_enc = accessSecret.encode('utf-8')
    string_to_sign = '{}\n{}'.format(timestamp, accessSecret)
    string_to_sign_enc = string_to_sign.encode('utf-8')
    hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    url = "https://openapi.zhunmou.com/ai/distinguish"

    payload = json.dumps({
    "file_link": file_link,
    "mode_type": mode_type,
    "on_mini_filter": 1,
    })
    headers = {
    "Access-Key": app_key,
    "Timestamp": timestamp,
    "Signature": sign
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    j = json.loads(response.text)
    res = j['results']
    return res

# transform img, make img like xingren
def transform_img(img_path, output_path=None):
    # 打开图片
    img_crop_1 = Image.open(img_path)

    # 调整亮度
    enhancer = ImageEnhance.Brightness(img_crop_1)
    img_bright = enhancer.enhance(1.2)  # 1.2 是示例值，您可以根据需要调整

    # 调整对比度
    enhancer = ImageEnhance.Contrast(img_bright)
    img_contrast = enhancer.enhance(1.6)  # 1.6 是示例值

    # 调整色调（饱和度）
    enhancer = ImageEnhance.Color(img_contrast)
    img_final = enhancer.enhance(1.2)  # 1.1 是示例值
    if output_path is not None:
        img_final.save(output_path)
        print(f"transform img write to: {output_path}")
    return img_final

def drawdot_img(coordinates, input_path, output_path):
    # 读取了一张图像到 img 中
    img = cv2.imread(input_path)
 
    color = (255,0,0)  # 使用蓝色
    thickness = -1  # 框的厚度
    # 逐个绘制
    for coord in coordinates:
        x, y, w, h = coord["x"], coord["y"], coord["w"], coord["h"]
        x2, y2 = x + w, y + h
              
        # 圆心坐标
        cx, cy = (x + x2) // 2, (y + y2) // 2
        
        # 绘制蓝色实心圆
        cv2.circle(img, center=(cx,cy), radius=5, color=color, thickness=thickness)
    # 将图片写入output_path
    print(f"dot_img write to: {output_path}")
    cv2.imwrite(output_path, img)

# draw from txt
def drawdot_txt_img(coordinates, input_path, output_path):
    # 读取了一张图像到 img 中
    img = cv2.imread(input_path)
 
    color = (255,0,0)  # 使用蓝色
    thickness = -1  # 框的厚度
    # 逐个绘制
    for coord in coordinates:
        l = coord.split(' ')
        x, y, w, h = int(l[1]), int(l[2]), int(l[3]), int(l[4])
        x2, y2 = x + w, y + h
              
        # 圆心坐标
        cx, cy = (x + x2) // 2, (y + y2) // 2
        
        # 绘制蓝色实心圆
        cv2.circle(img, center=(cx,cy), radius=5, color=color, thickness=thickness)
    # 将图片写入output_path
    print(f"dot_img write to: {output_path}")
    cv2.imwrite(output_path, img)

# 将压缩图片进行转换，并存储
def transform_save(img_path, img_transform_path):
    img_obj = transform_img(img_path, img_transform_path)

# 将转换的图片进行对应api处理，并在图片上标点
def circle_save(img_path, output_path, file_link, mode_type=205):
    coordinates = get_from_lingmou(file_link, mode_type)
    drawdot_img(coordinates, img_path, output_path)

# 将图片传给api，然后将返回结果存为txt文件
def api_result_save(file_link, output_txt_path, mode_type=205):
    coordinates = get_from_lingmou(file_link, mode_type)
    with open(output_txt_path, 'a+', encoding='utf-8') as f:
        for i, coord in enumerate(coordinates):
            x, y, w, h = coord['x'], coord['y'], coord['w'], coord['h']
            line = f"{i} {x} {y} {w} {h}"
            f.write(line + '\n')

# 将txt文件中的数据标注到图片上并且保存
def txt_circle_save(img_path, output_path, txt_path):
    coordinates = None
    with open(txt_path, 'r', encoding='utf-8') as f:
        coordinates = f.readlines()
    drawdot_txt_img(coordinates, img_path, output_path)


if __name__ == '__main__':
    base_url = "https://caiwei.obs.cn-north-4.myhuaweicloud.com/temp/test_crop/shuidao/"
    transform_base = "transform_img/shuidao"
    draw_base = "draw_circle/shuidao"
    base_img_path = "compress_img"
    txt_base_path = "need_check_txt/shuidao"
    output_img_base = "output_img_test/shuidao"

    # 对图像传给api并将其结果保存到txt文件中
    # for img_name in os.listdir(base_img_path):
    #     txt_name = img_name.replace('.jpg', '.txt')
    #     img_path = os.path.join(base_img_path, img_name)
    #     img_url = os.path.join(base_url, img_name)
    #     output_txt_path = os.path.join(txt_base_path, txt_name)
    #     api_result_save(img_url, output_txt_path)

    # 使用txt文件中的数据标注图像
    for txt_name in os.listdir(txt_base_path):
        img_name = txt_name.replace(".txt", ".jpg")
        img_path = os.path.join(base_img_path, img_name)
        out_img_path = os.path.join(output_img_base, img_name)
        txt_path = os.path.join(txt_base_path, txt_name)
        txt_circle_save(img_path, out_img_path, txt_path)

    # 对图像进行转换并保存
    # for img_name in os.listdir(base_img_path):
    #     # img_url = os.path.join(base_url, img_name)
    #     # circle_save_path = os.path.join(draw_base, img_name)
    #     img_path = os.path.join(base_img_path, img_name)
    #     img_transform_path = os.path.join(transform_base, img_name)
    #     transform_save(img_path, img_transform_path)
    
    # 在图像标识圆点
    # for img_name in os.listdir(base_img_path):
    #     img_url = os.path.join(base_url, img_name)
    #     img_path = os.path.join(base_img_path, img_name)
    #     circle_save_path = os.path.join(draw_base, img_name)
    #     circle_save(img_path, circle_save_path, img_url)
    #     break
    # res = get_from_lingmou(test_url)