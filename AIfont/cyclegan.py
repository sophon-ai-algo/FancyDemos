import cv2
import numpy as np
import sophon.sail as sail

import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
# from kaiti to sushi-style
# initialize an Engine instance using bmodel.
class CharsImageGenerator(object):
    """ 生成字符图像：背景为白色，字体为黑色 """
    # 数字和英文字母列表

    def __init__(self, plate_type):
        """ 一些参数的初始化
        :param plate_type: 需要生成的车牌类型
        """
        if plate_type == 'single_blue':
            # 字符为白色
            self.is_white_char = True
        elif plate_type in ['single_yellow', 'small_new_energy']:
            # 字符为黑字
            self.is_white_char = False
        self.plate_type = plate_type
        # 字符图片参数
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.font_ch = ImageFont.truetype(os.path.join(cur_dir, "kaiti.TTF"), 128, 0)  # 中文字体格式
        self.char_width = 128
        self.char_height = 128  # 字符高度
        self.chinese_original_width = 128
        self.chinese_original_height = 128
        self.bg_color = (255, 255, 255)
        self.fg_color = (0, 0, 0)
        #self.bg_img = np.array(Image.new("RGB", (self.plate_width, self.plate_height), self.bg_color))

    def generate_ch_char_image(self, char):
        """ 生成中文字符图片
        :param char: 待生成的中文字符
        """
        img = Image.new("RGB", (self.chinese_original_width, self.chinese_original_height), self.bg_color)
        ImageDraw.Draw(img).text((0, 3), char, self.fg_color, font=self.font_ch)
        if self.chinese_original_width != self.char_width:
            img = img.resize((self.char_width, self.plate_height))
        return np.array(img)
def main(model_path,input_kaiti):
    bmodel = sail.Engine(model_path, 0, sail.IOMode.SYSIO)
    # graph_name is just the net_name in conversion step.
    graph_name = bmodel.get_graph_names()[0]
    input_tensor_name = bmodel.get_input_names(graph_name)[0]

####you just need input a chinese char

    #input_kaiti = input()
    list_input = []
    for k in range(len(input_kaiti)):
        list_input.append(input_kaiti[k])
    #print(list_input)
    for j in range(len(list_input)):
        print(list_input)
        gen = CharsImageGenerator('single_yellow')
        char_zi = list_input[j]
        output = gen.generate_ch_char_image(char_zi)

        cv2.imwrite(str(j)+'.jpg',output)
    for i in range(len(input_kaiti)):
        img = cv2.imread(str(i)+'.jpg',cv2.IMREAD_GRAYSCALE)
        print(img)
        #img = cv2.resize(img,(128,128))

        img = np.expand_dims(img,axis =0)
        img = np.expand_dims(img,axis =0)
        #### change the shape of img from (128,128) to (1,1,128,128)
        input_data = {input_tensor_name: \
            img}

        # # do inference
        outputs = bmodel.process(graph_name, input_data)
        #print(outputs)
        key_1 = list(outputs.keys())[0]

        result = np.squeeze(outputs[str(key_1)])
        #result = np.trunc(result*255)
        result =(result +1)* 0.5
        result = result * 255
        result = result + 0.5
        result = np.clip(result,0,255)
        if i ==0:
            tem_result = result
        else:
            tem_result = np.hstack((tem_result,result))
        ###save output img
        cv2.imwrite('new__'+str(i)+'.jpg',result)

    cv2.imwrite('finalresult.jpg',tem_result)
    print(tem_result.shape)
    return tem_result
if __name__ == '__main__':
    input_kaiti = input()
    main('wangxizhi_2.bmodel',input_kaiti)

