# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse#提供了一种简单而灵活的方式，让你能够轻松地编写用户友好的命令行界面，解析和处理命令行输入。
import csv#用于存储表格数据。在 CSV 文件中，每行代表表格中的一行，每个字段之间用逗号或其他分隔符进行分隔。
import os#提供了一系列与操作系统交互的功能。通过 os 模块，你可以执行各种与文件系统和进程管理相关的操作。
import platform#用于提供关于当前系统平台的信息。它允许你编写跨平台的代码，根据不同的操作系统或硬件架构采取不同的行为。
import sys#提供了与 Python 解释器及其环境交互的功能。通过 sys 模块，你可以访问和修改与解释器运行时相关的信息
from pathlib import Path#用于处理文件系统路径。它提供了一个面向对象的路径操作接口，使得路径的构建、解析和操作更加直观和方便

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box#导入用于在绘图或图像中进行标注、处理颜色相关信息，以及保存或可视化特定区域或框。

from models.common import DetectMultiBackend#导入一种用于检测多个后端（或者其他特定功能）的功能或类。
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams#导入可能用于处理图像和视频的加载和预处理。
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)#导入LOGGER：可能是一个用于记录日志的工具。
Profile：可能是与性能分析相关的类或函数。
check_file：可能是用于检查文件的存在性或其他属性的函数。
check_img_size：可能是用于检查图像尺寸的函数。
check_imshow：可能是用于检查图像显示的函数。
check_requirements：可能是用于检查依赖项的函数。
colorstr：可能是用于处理颜色字符串的函数。
cv2：可能是 OpenCV 库的接口。
increment_path：可能是用于生成唯一路径的函数。
non_max_suppression：可能是与非最大抑制相关的函数。
print_args：可能是用于打印程序参数的函数。
scale_boxes：可能是用于缩放边界框的函数。
strip_optimizer：可能是用于去除模型优化器的函数。
xyxy2xywh：可能是用于转换坐标表示的函数。
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL指定了 YOLOv5 模型权重文件的路径
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)指定了数据源的地址
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path数据集的地址
        imgsz=(640, 640),  # inference size (height, width)指定了推理时的图像大小，具体来说是高度和宽度。
        conf_thres=0.25,  # confidence threshold指定了置信度的阈值为0.25
        iou_thres=0.45,  # NMS IOU threshold在目标检测中使用的一个参数，用于控制非极大值抑制的策略
        max_det=1000,  # maximum detections per image指定了每张图片的最大检测次数
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu用于指定设备的执行，用于执行模型的推理
        view_img=False,  # show results用于是否显示检测结果图像
        save_txt=False,  # save results to *.txt指定了是否将检测结果保存为文本文件
        save_csv=False,  # save results in CSV format指定了是否将检测结果保存为CSV文件
        save_conf=False,  # save confidences in --save-txt labels指定了是否在文本文件中保存检测框的置信度信息
        save_crop=False,  # save cropped prediction boxes指定了是否保存裁剪的预测框。
        nosave=False,  # do not save images/videos指定了是否保存图像或视频（True不保存，False保存）
        classes=None,  # filter by class: --class 0, or --class 0 2 3指定了一个参数，用于根据类别进行过滤
        agnostic_nms=False,  # class-agnostic NMS    agnostic_nms=False: 这是一个参数，表示是否在运行模型推理后使用类别不可知的非最大抑制。在这个例子中，该参数被设置为 False，意味着不使用类别不可知的 NMS。
        augment=False,  # augmented inference这是一个参数，表示是否在运行模型推理时进行数据增强。在这个例子中，该参数被设置为 False，意味着不进行数据增强。
        visualize=False,  # visualize features   visualize=False: 这是一个参数，表示是否在运行模型推理时可视化模型的特征。在这个例子中，该参数被设置为 False，意味着不进行特征可视化。
        update=False,  # update all models  update=False: 这是一个参数，表示是否在运行时更新所有的模型。在这个例子中，该参数被设置为 False，意味着不进行模型更新。
        project=ROOT / 'runs/detect',  # save results to project/name指定了将结果保存到指定项目路径的参数。
        name='exp',  # save results to project/name表示设置保存检测结果的子目录名称。在这个例子中，name 被设置为 'exp'，这将是保存结果的子目录的名称。
        exist_ok=False,  # existing project/name ok, do not increment表示是否允许覆盖已经存在的目录或文件。在这个例子中，该参数被设置为 False，意味着不允许覆盖已经存在的目录或文件。
        line_thickness=3,  # bounding box thickness (pixels)这是一个参数，表示设置边界框的线条粗细为 3 个像素。这意味着在图像中绘制的边界框将具有 3 个像素的线宽
        hide_labels=False,  # hide labels表示是否在图像上隐藏目标的标签。在这个例子中，该参数被设置为 False，意味着不隐藏标签。
        hide_conf=False,  # hide confidences指定了是否隐藏置信度信息
        half=False,  # use FP16 half-precision inference表示是否在运行模型推理时使用 FP16（半精度）计算。在这个例子中，该参数被设置为 False，意味着不使用半精度推理。
        dnn=False,  # use OpenCV DNN for ONNX inference表示是否在运行 ONNX 模型推理时使用 OpenCV DNN。在这个例子中，该参数被设置为 False，意味着不使用 OpenCV DNN 进行推理。
        vid_stride=1,  # video frame-rate stride表示视频处理时使用的帧率跨度。在这个例子中，该参数被设置为 1，意味着对视频中的每一帧都进行处理。
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images根据条件设置变量 save_img，用于确定是否保存推理结果的图像。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)用于判断变量 source 是否代表一个图像或视频文件。
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))主要用于判断变量 source 是否代表一个 URL 地址。
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)用于判断变量 source 是否代表一个网络摄像头（webcam）。
    screenshot = source.lower().startswith('screen')判断变量 source 是否代表一个屏幕截图。
    if is_url and is_file:
        source = check_file(source)  # download用于检查文件是否存在，如果不存在，则尝试下载文件。

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run用于创建一个增量的目录路径。
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir创建一个目录，用于保存标签文件或其它相关数据

    # Load model
    device = select_device(device)择并设置深度学习模型的计算设备（CPU或者GPU）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)#创建一个模型对象
    stride, names, pt = model.stride, model.names, model.pt#定义模型的三个属性的值stride 是指在输入图像上进行目标检测时的步幅。names 包含了模型所能检测的目标类别的名称列表。pt 表示模型所使用的 PyTorch 框架的版本。
    imgsz = check_img_size(imgsz, s=stride)  # check image size确保输入图像的尺寸满足模型的要求

    # Dataloader
    bs = 1  # batch_size设置批量大小为 1。批量大小是深度学习中用于一次性处理的样本数量
    if webcam:
        view_img = check_imshow(warn=True)#检查当前环境是否支持显示图像，以便后续的任务中是否需要显示图像的处理结果
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)#该操作的目的是创建一个用于加载视频流的数据集对象，以便后续的模型推理
        bs = len(dataset)#返回数据集中的样本数量
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs#存储每个批次视频的路径和视频写入器

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup用于进行模型的预热（warmup）操作。
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)#确保输入数据（图像）的类型和位置与模型的要求相匹配
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32根据模型的设置将输入图像 im 转换为半精度浮点数（fp16）或单精度浮点数（fp32）格式。
            im /= 255  # 0 - 255 to 0.0 - 1.0#像素值从原始的 0 到 255 范围归一化到 0.0 到 1.0 范围
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim#输入图像 im 的形状，如果它是三维的，表示单张图像（例如，高度、宽度和通道数），则通过添加一个维度来扩展成四维，以符合模型对批量输入的要求。

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False#根据条件判断是否需要可视化（visualization），并创建一个路径用于保存可视化的结果。
            pred = model(im, augment=augment, visualize=visualize)#调用模型 model 对象对输入图像 im 进行推理，并将结果存储在变量 pred 中。

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)#过滤掉一些重叠的边界框，只保留概率最高的边界框

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))#创建了一个 Annotator 对象，用于在图像上绘制注释
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()#对检测结果 det 中的边界框坐标进行缩放操作，将其从模型输出的相对坐标（可能是在 0 到 1 范围内）映射到输入图像 im0 的实际像素坐标。

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            #在图像上添加注释，并在窗口中显示图像，以供用户观察。在 Linux 系统中，窗口的调整和显示可能需要进行特殊处理。
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        #根据条件保存图像或视频。当处理视频流时，它确保将每一帧都添加到正确的视频中，并在处理新视频时更新写入器。
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

   #根据条件是否保存文本文件或图像，输出保存结果
def parse_opt():
    parser = argparse.ArgumentParser()#解析命令行参数
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')#通过命令行传递 --weights 参数，并提供一个或多个模型权重文件的路径。如果用户没有提供该参数，将使用默认的模型权重路径。
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')#指定输入数据的来源
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')#指定数据集的配置文件的路径
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')#指定推理时图像的大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')#设定置信度阈值，低于阈值则会被过滤
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')#指定进行非极大值抑制时，两个边界框之间的IoU阈值。如果两个边界框的IoU高于该阈值，将执行NMS以过滤其中之一
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')#指定每张图片所能检测的最大次数
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#模型推理的设备
    parser.add_argument('--view-img', action='store_true', help='show results')#控制是否在模型推理时显示结果图像
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')#控制是否将模型推理的结果保存为文本文件
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')#是否将推理的结果保存为CSV文件
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')#是否将模型推理的置信度保存到文本文件
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')#是否保存模型推理的结果中检测到的目标区域的裁剪图像
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')#是否禁止保存模型推理的结果图像或视频。
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')#允许用户通过命令行传递一个或多个整数值，以指定要过滤的目标类别
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')#控制是否使用类别无关的非极大值抑制
    parser.add_argument('--augment', action='store_true', help='augmented inference')#控制是否在推理期间应用数据增强
    parser.add_argument('--visualize', action='store_true', help='visualize features')#控制是否在推理期间可视化模型的特征
    parser.add_argument('--update', action='store_true', help='update all models')#控制是否更新所有模型
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')#指定模型推理结果的保存目录
    parser.add_argument('--name', default='exp', help='save results to project/name')#指定模型推理结果保存目录的子目录名称
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')#控制是否在保存结果时允许使用现有的项目/名称，而不是递增命名
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')#指定绘制边界框时的线条粗细
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')#是否在图像上显示目标的标签
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')#是否隐藏置信度
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')#是否使用 FP16（半精度）推理
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')#是否使用opencv DNN进行推理
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')#设置步长
    opt = parser.parse_args()#
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand根据 opt.imgsz 的情况来调整图像的大小
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))#确保项目所需的软件包已经安装，并可能排除了一些不需要检查的软件包。
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
