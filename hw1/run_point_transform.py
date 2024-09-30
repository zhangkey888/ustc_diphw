import cv2
import numpy as np
import gradio as gr
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换
def rbf_kernel(x1, x2, sigma):
    return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * sigma**2))


def f(x):
    alpha = 1000
    f=1. / (x ** 2 + alpha)
    return f
def point_guided_deformation(im, psrc, pdst, alpha=1000, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """


    h, w, _ = im.shape
    im1 = np.ones((h * w, 3), dtype=np.uint8) * 255
    I = im.reshape(h * w, 3)
    coordinates = []
    for i in range(h):  # 对于图像的每一行
        for j in range(w):  # 对于图像的每一列
            coordinates.append([j + 1, i + 1])  # 将坐标添加到列表中

    # 将坐标列表转换为NumPy数组
    X = np.array(coordinates)
    N = psrc.shape[0]
    if N == 1:
        A = 1 / alpha
        a = (psrc - pdst) / A
    else:
        A = f(squareform(pdist(pdst)))
        if A.ndim == 1:
            A = A.reshape(-1, 1)

        a = np.linalg.solve(A, (psrc - pdst))
    q = np.repeat(X, N, axis=0) - np.tile(pdst, (h * w, 1))
    p = f(np.linalg.norm(q, axis=1)).reshape(h * w, N)
    im_position = np.round(p @ a + X).astype(int)
    T1 = np.all(im_position > 0, axis=1) & (im_position[:, 0] <= w) & (im_position[:, 1] <= h)
    T2 = np.ravel_multi_index((im_position[T1, 1] - 1, im_position[T1, 0] - 1), (h, w))
    im1[T1] = I[T2]
    im1= im1.reshape(h, w, 3)

    return im1


# warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping

    # return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
