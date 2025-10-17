# -*- encoding: utf-8 -*-
'''
@File    :   io.py
@Time    :   2025/10/18 02:41:29
@Author  :   crab 
@Version :   1.0
@Desc    :   最基本的一些io操作, 无关3D数据类型的处理
'''
import io
from PIL import Image

def save_gif_to_file(images:list[Image.Image], output_file):  
    with io.BytesIO() as writer:  
        images[0].save(  
            writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
        )  
        writer.seek(0)  
        with open(output_file, 'wb') as file:  
            file.write(writer.read())