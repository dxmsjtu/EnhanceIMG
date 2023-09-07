import cv2
import numpy as np

# 实现了多尺度图像增强和多尺度图像增强彩色恢复算法
def replaceZeroes(data):
    # 函数用于将输入数据中的零值替换为非零最小值，以避免在计算中出现除以零的情况。
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    # 单尺度Retinex算法的实现。它首先对输入图像进行高斯模糊处理，然后计算图像的对数域表示。
    # 接下来，它计算图像的反射成分，并对反射成分进行增强。最后，它将增强后的图像转换为8位无符号整数形式。
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    # 计算输入图像和经过高斯模糊处理后的图像的对数域表示。
    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    # 计算图像的反射成分。
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def MSR(img, scales):
    # 多尺度Retinex算法的实现。它根据不同的尺度对输入图像进行高斯模糊处理，然后计算每个尺度下的反射成分，并将它们加权求和，以得到最终的反射成分。

    # 定义了一个权重值，用于将不同尺度下的反射成分加权求和
    weight = 1 / 3.0
    # 获取尺度数组scales的大小，
    scales_size = len(scales)
    h, w = img.shape[:2]
    # 创建一个与输入图像大小相同的全零矩阵，用于存储反射成分。
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        # 对输入图像img 进行高斯模糊处理
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        # 将不同尺度下的反射成分进行累积
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)
    # 对累积的反射成分进行规范化，将其映射到0至255的范围内。
    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    # 得到最终的图像增强结果 log_uint8。
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def simple_color_balance(input_img, s1, s2):
    # 实现简单的颜色平衡
    h, w = input_img.shape[:2]
    temp_img = input_img.copy()
    one_dim_array = temp_img.flatten()
    sort_array = sorted(one_dim_array)

    per1 = int((h * w) * s1 / 100)
    minvalue = sort_array[per1]

    per2 = int((h * w) * s2 / 100)
    maxvalue = sort_array[(h * w) - 1 - per2]

    # 实施简单白平衡算法
    if (maxvalue <= minvalue):
        out_img = np.full(input_img.shape, maxvalue)
    else:
        scale = 255.0 / (maxvalue - minvalue)
        # out_img = np.where(temp_img < minvalue, 0)    # 防止像素溢出
        # out_img = np.where(out_img > maxvalue, 255)   # 防止像素溢出
        temp_img[temp_img<minvalue] = 0
        temp_img[temp_img>maxvalue] = 255
        out_img = scale * (temp_img - minvalue)        # 映射中间段的图像像素
        out_img = cv2.convertScaleAbs(out_img)
    return out_img

def MSRCR(img, scales, s1, s2):
    # 实现了多尺度Retinex颜色还原，可以增强图像的颜色和对比度，特别是在不同尺度下处理图像
    h, w = img.shape[:2]
    scles_size = len(scales)
    log_R = np.zeros((h, w), dtype=np.float32)
    # 计算输入图像的通道之和，以便后续颜色还原计算使用。
    img_sum = np.add(img[:,:,0],img[:,:,1],img[:,:,2])
    img_sum = replaceZeroes(img_sum)
    # 创建一个空列表，用于存储颜色还原后的灰度图像。
    gray_img = []

    # 迭代处理三个通道（R、G、B）。
    for j in range(3):
        img[:, :, j] = replaceZeroes(img[:, :, j])
        for i in range(0, scles_size):
            L_blur = cv2.GaussianBlur(img[:, :, j], (scales[i], scales[i]), 0)
            L_blur = replaceZeroes(L_blur)

            dst_img = cv2.log(img[:, :, j]/255.0)
            dst_Lblur = cv2.log(L_blur/255.0)
            dst_ixl = cv2.multiply(dst_img, dst_Lblur)
            # 计算Retinex反射成分，将当前通道的反射成分累加
            log_R += cv2.subtract(dst_img, dst_ixl)

        MSR = log_R / 3.0
        MSRCR = MSR * (cv2.log(125.0 * img[:, :, j]) - cv2.log(img_sum))
        gray = simple_color_balance(MSRCR, s1, s2)
        gray_img.append(gray)
    return gray_img

# 执行多尺度Retinex颜色恢复，并进行颜色平衡。
def MSRCP(img, scales, s1, s2):
    h, w = img.shape[:2]
    scales_size = len(scales)
    B_chan = img[:, :, 0]
    G_chan = img[:, :, 1]
    R_chan = img[:, :, 2]
    # 创建一个与输入图像大小相同的零矩阵，用于存储Retinex反射成分。
    log_R = np.zeros((h, w), dtype=np.float32)
    array_255 = np.full((h, w),255.0,dtype=np.float32)
    # 替换I_array中的零值，以避免在后续计算中出现除以零的情况。
    I_array = (B_chan + G_chan + R_chan) / 3.0
    I_array = replaceZeroes(I_array)

    for i in range(0, scales_size):
        L_blur = cv2.GaussianBlur(I_array, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_I = cv2.log(I_array/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_ixl = cv2.multiply(dst_I, dst_Lblur)
        log_R += cv2.subtract(dst_I, dst_ixl)
    # 除以3，得到平均的Retinex反射成分。
    MSR = log_R / 3.0
    Int1 = simple_color_balance(MSR, s1, s2)

    B_array = np.maximum(B_chan,G_chan,R_chan)
    A = np.minimum(array_255 / B_array, Int1/I_array)
    # 通过乘以调整系数A来调整红色通道。
    R_channel_out = A * R_chan
    G_channel_out = A * G_chan
    B_channel_out = A * B_chan
    # 三个通道合并为一个图像。
    MSRCP_Out_img = cv2.merge([B_channel_out, G_channel_out, R_channel_out])
    MSRCP_Out = cv2.convertScaleAbs(MSRCP_Out_img)

    return MSRCP_Out

