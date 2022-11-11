# -*- coding: utf-8 -*-
import os.path
import time
import cv2
import sys
import numpy as np
import PyQt5
import PySide2
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
from PySide2.QtGui import QImage, QPixmap
from PySide2 import QtCore
from PyQt5.QtGui import QImageReader
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QFormLayout
from PySide2.QtCore import QFile, Signal
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QPushButton,QLineEdit,QLabel,QTextEdit
import threading
from deploy.utils.config import get_config
from deploy.python.predict_system import SystemPredictor
from deploy.utils.draw_bbox import draw_video
from concurrent.futures import ThreadPoolExecutor

camera_id = 0
OUTPUT = []
VIDEO_SRC_PATH = ""
IMAGE_SRC_PATH = ""
videoImageSet = set()
singalImageSet = set()
runtimeSet = set()
RuntimeOutput = []
thingsOfText = set()

threadpool = ThreadPoolExecutor(20)

class Data(PySide2.QtCore.QObject):
    # 定义信号
    mySignal = Signal()
    def __init__(self):
        super(Data, self).__init__()
    def run(self):
        self.mySignal.emit()

# 修改1 增加子窗口类
class Child(QWidget):
    signal = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("更新索引库")
        formlayout = QFormLayout()
        self.btn_1 = QPushButton('选择图片')
        self.line_1 = QLineEdit()
        self.label_1 = QLabel('图片名字：')
        self.line_2 = QLineEdit()
        self.btn_2 = QPushButton('添加')
        self.btn_1.clicked.connect(self.get_img)
        self.btn_2.clicked.connect(self.ok)
        formlayout.addRow(self.btn_1, self.line_1)
        formlayout.addRow(self.label_1, self.line_2)
        formlayout.addRow(self.btn_2)
        self.setLayout(formlayout)

    def get_img(self):
        img_path, filetype = PyQt5.QtWidgets.QFileDialog.getOpenFileName(self, '选择图片')
        if img_path == '':
            return
        self.line_1.setText(img_path)

    def ok(self):
        img_path = self.line_1.text()
        img_name = self.line_2.text()

        params = [img_path, img_name]
        if img_path == '' or img_name == '':
            reply = PyQt5.QtWidgets.QMessageBox.warning(self, "警告！", "图片或者名称为空！", PyQt5.QtWidgets.QMessageBox.Yes | PyQt5.QtWidgets.QMessageBox.No, PyQt5.QtWidgets.QMessageBox.Yes)
            return
        self.signal.emit(params)
        # self.close()

# 图片、视频 是否添加子窗口
class ChildGoOn(QWidget):
    signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("确认窗口")
        formlayout = QFormLayout()
        self.label_1 = QLabel('已识别:')
        self.line_1 = QTextEdit()
        self.btn_1 = QPushButton('继续识别')
        self.btn_2 = QPushButton('确认')

        self.btn_1.clicked.connect(self.goOnAdd)
        self.btn_2.clicked.connect(self.ok)

        formlayout.addRow(self.label_1, self.line_1)
        formlayout.addRow(self.btn_1)
        formlayout.addRow(self.btn_2)
        self.setLayout(formlayout)

    def ok(self):
        self.line_1.clear()
        self.close()

    def goOnAdd(self):
        # 获取一下当前QTextEdit框中的所有物品
        global thingsOfText
        thingsOfText = set(self.line_1.toPlainText().split())
        print("thingsOfText")
        print(thingsOfText)
        self.signal.emit()
        # self.close()


# 实时检测子窗口
class ChildRuntime(QWidget):
    # signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时检测")
        formlayout = QFormLayout()
        self.label_1 = QLabel('已识别:')
        self.line_1 = QTextEdit()

        self.btn_2 = QPushButton('确认')
        self.btn_2.clicked.connect(self.ok)
        formlayout.addRow(self.label_1, self.line_1)
        formlayout.addRow(self.btn_2)
        self.setLayout(formlayout)

    def ok(self):
        self.line_1.clear()
        self.close()


# 建立索引库子窗口
class ChildBuildG(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("建立索引库")

class StartWindow:
    def __init__(self):
        # 初始化UI
        data = Data()
        ui_file = QFile("uuui.ui")
        ui_file.open(QFile.ReadOnly)
        self.ui = QUiLoader().load(ui_file)

        self.CameraIsOpen = False   # 测试摄像头功能标志
        self.LabelIsBusy = False    # 打开视频忙碌标志
        self.RuntimeInfer = False   # 实时推理忙碌标志
        self.ImageIsBusy = False    # 检测图片忙碌标志
        self.is_infer_video = False # 视频识别忙碌标志

        self.cap = cv2.VideoCapture()
        self.out = None  # 录制视频输出对象
        self.target_path = None
        self.frame_height = None
        self.frame_width = None
        self.fps = None
        self.frameTime = 30   # ms

        self.goOnRecognize = False   # 视频是否继续识别
        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()

        # 绑定按钮
        self.ui.camera_test_btn.clicked.connect(self.camera_test)       # 测试打开摄像头
        self.ui.video_record_btn.clicked.connect(self.video_record)     # 打开摄像头录制一段视频并保存在本地
        self.ui.read_video_btn.clicked.connect(self.read_video)         # 读取指定路径的视频并打开
        self.ui.image_infer_btn.clicked.connect(self.infer_signal)      # 识别单张图片
        self.ui.bsk_dtc_btn.clicked.connect(self.runtime_open_camera)           # 实时推理
        self.ui.build_gallery_btn.clicked.connect(self.build_gallery)   # 建立索引库
        self.ui.update_gallery_btn.clicked.connect(self.update_gallery) # 更新索引库

        # 修改2 初始化子窗口，绑定信号槽
        self.child = Child()
        self.child.signal.connect(self.get_params_update_gallery)

        # 初始化单张图片识别子窗口 绑定信号槽
        self.ImageChild = ChildGoOn()
        self.ImageChild.signal.connect(self.get_image)

        # 初始化视频检测子窗口 绑定信号槽
        self.VideoChild = ChildGoOn()
        self.VideoChild.signal.connect(self.get_video)

        # 初始化实时检测子窗口
        self.RuntimeChild = ChildRuntime()


        PySide2.QtCore.QObject.connect(data, PySide2.QtCore.SIGNAL('mySignal()'), self.futureTest)
        # PySide2.QtCore.QObject.connect(data, PySide2.QtCore.SIGNAL('mySignal()'), self.show_real_vedio)
        # PySide2.QtCore.QObject.connect(data, PySide2.QtCore.SIGNAL('mySignal()'), self.show_vedio)

    def camera_test(self):
        if self.CameraIsOpen == False:
            self.ui.build_gallery_btn.setDisabled(True)
            self.ui.video_record_btn.setDisabled(True)
            self.ui.read_video_btn.setDisabled(True)
            self.ui.image_infer_btn.setDisabled(True)
            self.ui.bsk_dtc_btn.setDisabled(True)
            self.ui.update_gallery_btn.setDisabled(True)
            self.CameraIsOpen = True
            self.ui.camera_test_btn.setText("关闭摄像头")
            self.cap = cv2.VideoCapture(camera_id)
            # classfier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            # color = (0, 225, 0)  # 人脸框的颜色，采用rgb模型，这里表示g取255，为绿色框
            while True:
                if self.CameraIsOpen == False:
                    break
                # self.open()   # 平替一下

                success, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # PIL_img = frame
                # results = []
                # output = {'rec_docs': '测试', 'bbox': [10, 20, 50, 60], 'rec_scores': 0.8}
                # results.append(output)
                # draw_bbox_results(image=PIL_img, results=results,save_dir='./drawOutput/')

                # 人脸检测的trick
                '''
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3,
                                                       minSize=(32, 32))  # 利用分类器检测灰度图像中的人脸矩阵数，1.2和3分别为图片缩放比例和需要检测的有效点数
                if len(faceRects) > 0:  # 大于0则检测到人脸
                    for faceRect in faceRects:  # 可能检测到多个人脸，用for循环单独框出每一张人脸
                        x, y, w, h = faceRect  # 获取框的左上的坐标，框的长宽
                        # cv2.rectangle()完成画框的工作，这里外扩了10个像素以框出比人脸稍大一点的区域，从而得到相对完整一点的人脸图像；cv2.rectangle()函数的最后两个参数一个用于指定矩形边框的颜色，一个用于指定矩形边框线条的粗细程度。
                        cv2.rectangle(frame, (x - 10, y - 10), (x + w - 10, y + h - 10), color, 2)
                '''

                image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.video_label.setPixmap(QPixmap.fromImage(image))
                self.ui.FPS_label.setText("FPS:{:.2f}".format(self.cap.get(cv2.CAP_PROP_FPS)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cap.release()
                    self.ui.video_label.clear()
                    break
        else:
            self.ui.build_gallery_btn.setDisabled(False)
            self.ui.video_record_btn.setDisabled(False)
            self.ui.read_video_btn.setDisabled(False)
            self.ui.image_infer_btn.setDisabled(False)
            self.ui.bsk_dtc_btn.setDisabled(False)
            self.ui.update_gallery_btn.setDisabled(False)
            self.CameraIsOpen = False
            self.ui.camera_test_btn.setText("开启摄像头")
            self.ui.video_label.clear()
            self.cap.release()
            self.ui.FPS_label.setText("")


    def open(self):
        success, frame = self.cap.read()
        frame = cv2.flip(frame,1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1],frame.shape[0], QImage.Format_RGB888)
        self.ui.video_label.setPixmap(QPixmap.fromImage(image))

    def video_record(self):
        if self.LabelIsBusy == False:
            self.ui.camera_test_btn.setDisabled(True)
            self.ui.build_gallery_btn.setDisabled(True)
            self.ui.read_video_btn.setDisabled(True)
            self.ui.image_infer_btn.setDisabled(True)
            self.ui.bsk_dtc_btn.setDisabled(True)
            self.ui.update_gallery_btn.setDisabled(True)
            self.LabelIsBusy = True
            # 打开摄像头录制一段视频并保存在本地
            target_dir = QFileDialog.getExistingDirectory(self.ui, "选择文件夹", "/")
            self.ui.video_record_btn.setText("停止录制")
            print('write video to ', target_dir)
            name = 'video_{}.mp4'.format(time.strftime("%Y-%m-%d-%H-%M-%S"))
            self.target_path = target_dir + '/' +  name
            self.cap = cv2.VideoCapture(camera_id)
            # try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 获取视频的帧宽度
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的帧高度

            self.out = cv2.VideoWriter(self.target_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                  (width, height))  # 保存本地视频
            while True and self.LabelIsBusy==True:
                ret, frame = self.cap.read()
                # 按Q退出
                frame2 = cv2.flip(frame,1)
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                image = QImage(frame2.data, frame2.shape[1], frame2.shape[0], QImage.Format_RGB888)
                self.ui.video_label.setPixmap(QPixmap.fromImage(image))

                self.out.write(frame.astype(np.uint8))
                cv2.putText(frame, 'press Q to end record', (5, 50,), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.cap.release()
                    self.out.release()
                    msg_box = QMessageBox(QMessageBox.Information, '提示', '视频保存为' + self.target_path)
                    msg_box.exec_()
                    break
        else:
            self.ui.camera_test_btn.setDisabled(False)
            self.ui.build_gallery_btn.setDisabled(False)
            self.ui.read_video_btn.setDisabled(False)
            self.ui.image_infer_btn.setDisabled(False)
            self.ui.bsk_dtc_btn.setDisabled(False)
            self.ui.update_gallery_btn.setDisabled(False)
            self.LabelIsBusy = False
            self.ui.video_record_btn.setText("录制视频")
            self.cap.release()
            self.ui.video_label.clear()
            self.out.release()
            msg_box = QMessageBox(QMessageBox.Information, '提示', '视频保存为' + self.target_path)
            msg_box.exec_()


    # 视频识别
    def read_video(self):
        if self.is_infer_video == False:
            self.ui.camera_test_btn.setDisabled(True)
            self.ui.build_gallery_btn.setDisabled(True)
            self.ui.video_record_btn.setDisabled(True)
            self.ui.image_infer_btn.setDisabled(True)
            self.ui.bsk_dtc_btn.setDisabled(True)
            self.ui.update_gallery_btn.setDisabled(True)
            self.ui.read_video_btn.setText("关闭视频识别")
            self.is_infer_video = True
            global VIDEO_SRC_PATH  # 忘了加了md
            global videoImageSet
            self.VideoChild.line_1.clear()
            videoImageSet.clear()  # 每次开始识别之前都先清理一下上次识别的结果
            # 读取指定路径的视频并打开
            directory1 = QFileDialog.getOpenFileNames(self.ui, "选择文件", "/")
            print(directory1)
            src_path = directory1[0][0]
            VIDEO_SRC_PATH = src_path
            name = src_path.split(os.sep)[-1]

            self.cap = cv2.VideoCapture(src_path)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            config = get_config('deploy/configs/inference_general.yaml', show=False)
            system_predictor = SystemPredictor(config)  # 只创建一次对象
            flag = 1
            output = None
            SumOutput = []
            wait_time = (1 / fps) * 0.01
            while self.cap.isOpened():
                # 画面暂留一段时间
                time.sleep(wait_time)
                ret, frame = self.cap.read()
                # =====直接预测======
                if ret == True:
                    if flag == 1:
                        output = system_predictor.predict(frame)
                        SumOutput.extend(output)
                    flag ^= 1
                    # frame = cv2.flip(frame, 1)
                    frame = draw_video(frame, output)

                    # 这里修改为展示输出
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    self.ui.video_label.setPixmap(QPixmap.fromImage(image))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.cap.release()
                        self.ui.video_label.clear()
                        break
                else:
                    self.cap.release()
                    self.ui.video_label.clear()
                    break
            # 识别物品去重
            for i in range(len(SumOutput)):
                videoImageSet.add(SumOutput[i]["rec_docs"])
            # for i in range(len(imageSet)):
            #     self.VideoChild.line_1.append(imageSet.pop() + "\n")
            for every in iter(videoImageSet):
                self.VideoChild.line_1.append(every + "\n")
            self.VideoChild.show()
            self.ui.video_label.clear()
            self.cap.release()
        else:
            self.is_infer_video = False
            self.ui.camera_test_btn.setDisabled(False)
            self.ui.build_gallery_btn.setDisabled(False)
            self.ui.video_record_btn.setDisabled(False)
            self.ui.image_infer_btn.setDisabled(False)
            self.ui.bsk_dtc_btn.setDisabled(False)
            self.ui.update_gallery_btn.setDisabled(False)
            self.ui.read_video_btn.setText("视频识别")
            self.ui.video_label.clear()
            self.cap.release()


    def runImageCommand(self,img_name):
        command = r'python deploy/python/predict_system.py -m signal-img -c deploy/configs/inference_general.yaml -o Global.infer_imgs="{}" '.format(
            img_name)
        os.system(command)


    # def infer_signal(self):
    #     # 传入一张图片，然后将图片和预测结果都展示在界面
    #     dict = QFileDialog.getOpenFileNames(self.ui, '选择图片')
    #     img_path = dict[0][0]
    #     if img_path == '':
    #         return
    #     # 因为infer_img必须在deploy目录下，所以需要获取文件名，然后拼接
    #     img_name = os.path.basename(img_path)
    #     img_name = os.path.join('deploy/images/', img_name)
    #     threadpool.submit(self.runImageCommand,img_name)
    #     return

    def infer_signal(self):
        # 读取指定路径的图片并打开
        global singalImageSet,thingsOfText,IMAGE_SRC_PATH
        singalImageSet.clear()
        directory1 = QFileDialog.getOpenFileNames(self.ui, "选择文件", "/")
        # print(directory1)
        try:
            src_path = directory1[0][0]
            IMAGE_SRC_PATH = src_path
            # print(IMAGE_SRC_PATH)
            image = cv2.imread(src_path)
            name = src_path.split(os.sep)[-1]
        except Exception as e:
            print("the error is ", e)

        try:
            config = get_config('deploy/configs/inference_general.yaml', show=False)
            system_predictor = SystemPredictor(config)  # 只创建一次对象
            output = system_predictor.predict(image)
        except Exception as e:
            print("read image failed:", e)
            msg_box = QMessageBox(QMessageBox.Information, '错误', '读取图片失败')
            msg_box.exec_()
            return
        # OUTPUT = output
        try:
            print(output[0]["rec_docs"])
            image = draw_video(image, output)

            for i in range(len(output)):
                singalImageSet.add(output[i]["rec_docs"])
            for every in singalImageSet:
                self.ImageChild.line_1.append(every + "\n")
            self.ImageChild.show()
            threadpool.submit(self.showImage, image)

        except Exception as e:
            print("infer signal image failed:", e)
            msg_box = QMessageBox(QMessageBox.Information, '错误', '该图片没有识别出商品')
            msg_box.exec_()
            return


    def showImage(self,image):
        cv2.imshow("image", image)
        cv2.waitKey(0)


    def runtime_open_camera(self):
        global RuntimeOutput,runtimeSet
        if self.RuntimeInfer == False:
            self.ui.camera_test_btn.setDisabled(True)
            self.ui.video_record_btn.setDisabled(True)
            self.ui.read_video_btn.setDisabled(True)
            self.ui.image_infer_btn.setDisabled(True)
            self.ui.build_gallery_btn.setDisabled(True)
            self.ui.update_gallery_btn.setDisabled(True)
            self.RuntimeInfer = True
            self.ui.bsk_dtc_btn.setText("关闭实时推理")
            # 捕获摄像头并循环推理每一帧
            print(camera_id)
            self.cap = cv2.VideoCapture(camera_id)
            config = get_config('deploy/configs/inference_general.yaml', show=True)
            system_predictor = SystemPredictor(config)  # 只创建一次对象
            flag = 1
            output = None
            try:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    # 按Q退出
                    if flag:
                        output = system_predictor.predict(frame)
                        RuntimeOutput.extend(output)
                    flag ^= 1
                    # frame = cv2.flip(frame, 1)
                    frame = draw_video(frame, output)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    self.ui.video_label.setPixmap(QPixmap.fromImage(image))

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.cap.release()
                        self.ui.video_label.clear()
                        break

            except Exception as e:
                print("video record failed:", e)
                self.cap.release()
                msg_box = QMessageBox(QMessageBox.Information, '错误', '实时推理失败')
                msg_box.exec_()
                return
        else:
            self.ui.camera_test_btn.setDisabled(False)
            self.ui.video_record_btn.setDisabled(False)
            self.ui.read_video_btn.setDisabled(False)
            self.ui.image_infer_btn.setDisabled(False)
            self.ui.build_gallery_btn.setDisabled(False)
            self.ui.update_gallery_btn.setDisabled(False)
            self.RuntimeInfer = False
            self.ui.bsk_dtc_btn.setText("实时推理")
            self.ui.video_label.clear()
            self.cap.release()

            print("Runtimeoutput")
            print(RuntimeOutput)
            # 识别物品去重
            for i in range(len(RuntimeOutput)):
                runtimeSet.add(RuntimeOutput[i]["rec_docs"])
            for every in iter(runtimeSet):
                self.RuntimeChild.line_1.append(every + "\n")
            self.RuntimeChild.show()

            RuntimeOutput.clear()
            runtimeSet.clear()


    def futureTest(self):
        if self.future.result() == 1:
            self.ui.video_label.clear()
            self.gif.stop()
            msg_box = QMessageBox(QMessageBox.Information, '提示', '索引库建立完成')
            msg_box.exec_()


    def build_gallery(self):
        # 建立索引库
        # self.ui.video_label.setPixmap(QPixmap("123.jpg"))
        # 其他按钮变灰

        self.ui.camera_test_btn.setDisabled(True)
        self.ui.video_record_btn.setDisabled(True)
        self.ui.read_video_btn.setDisabled(True)
        self.ui.image_infer_btn.setDisabled(True)
        self.ui.bsk_dtc_btn.setDisabled(True)
        self.ui.update_gallery_btn.setDisabled(True)
        msg_box = QMessageBox(QMessageBox.Information, '提示', '索引库建立中...')
        msg_box.exec_()

        command = r'rmdir /s/q index'
        os.system(command)
        print("成功替换索引库文件夹")

        self.build_gallery_handle()

        self.ui.camera_test_btn.setDisabled(False)
        self.ui.video_record_btn.setDisabled(False)
        self.ui.read_video_btn.setDisabled(False)
        self.ui.image_infer_btn.setDisabled(False)
        self.ui.bsk_dtc_btn.setDisabled(False)
        self.ui.update_gallery_btn.setDisabled(False)


    def build_gallery_handle(self):
        # time.sleep(0.5)
        future = threadpool.submit(self.buildRun)
        # self.buildRun()
        # futureTemp = threadpool.submit(self.futureTest)
        if future.result() == 1:
            self.ui.video_label.clear()
            msg_box = QMessageBox(QMessageBox.Information, '提示', '索引库建立完成')
            msg_box.exec_()
        # msg_box = QMessageBox(QMessageBox.Information, '提示', '索引库建立完成')
        # msg_box.exec_()
        # future.result() 会阻塞
        # self.ui.video_label.clear()



    def buildRun(self):
        command = r'python deploy/python/build_gallery.py -c deploy/configs/build_general.yaml -o IndexProcess.data_file="deploy/gallery_label.txt" -o IndexProcess.index_dir="deploy/index/" '
        os.system(command)
        print('建立完成')
        return 1


    # 修改3 函数体修改
    def update_gallery(self):
        command = r'rmdir /s/q index'
        os.system(command)
        print("成功替换索引库文件夹")
        self.child.show()


    # 修改4 增加方法，获取子窗口参数
    def get_params_update_gallery(self, params):
        print('params:', params)
        self.update_gallery_img_path = params[0]
        self.update_gallery_img_name = params[1]

        if self.update_gallery_img_path == '' or self.update_gallery_img_name == '':
            reply = QMessageBox.warning(self.ui, "警告！", "未选择图片或者未设置名称，无法更新！", QMessageBox.Yes | QMessageBox.No,
                                        QMessageBox.Yes)
            return

        self.add_img(self.update_gallery_img_path, self.update_gallery_img_name)
        threadpool.submit(self.UpdateGallary)


    # 图像识别的槽函数
    def get_image(self):
        global thingsOfText,IMAGE_SRC_PATH
        # print(1111111111111)
        # print(IMAGE_SRC_PATH)
        img = cv2.imread(IMAGE_SRC_PATH)
        config = get_config('deploy/configs/inference_general.yaml', show=False)
        system_predictor = SystemPredictor(config)  # 只创建一次对象

        SumOutput = system_predictor.predict(img)
        img = draw_video(img, SumOutput)
        cv2.imshow("image",img)
        cv2.waitKey(0)

        # 识别物品去重
        for i in range(len(SumOutput)):
            if SumOutput[i]["rec_docs"] not in singalImageSet:
                self.ImageChild.line_1.append(SumOutput[i]["rec_docs"] + "\n")

        # 当前文本检查
        for every in singalImageSet:
            if every not in thingsOfText:
                thingsOfText.add(every)
                self.ImageChild.line_1.append(every + "\n")
        print("here is error")



    # 视频识别的槽函数
    def get_video(self):
        # self.goOnRecognize = params
        # if self.goOnRecognize is True:
        global thingsOfText
        cap = cv2.VideoCapture(VIDEO_SRC_PATH)
        config = get_config('deploy/configs/inference_general.yaml', show=False)
        # print("前")
        system_predictor = SystemPredictor(config)  # 只创建一次对象
        # print("后")
        flag = 1
        output = None
        SumOutput = []

        #  麻了 这竟然是False  没电了艹！！！！！
        # print(SRC_PATH)   # 这是空！！！
        # print(cap.isOpened())
        while cap.isOpened():
            ret, frame = cap.read()
            # =====直接预测======
            if ret == True:
                if flag == 1:
                    output = system_predictor.predict(frame)
                    SumOutput.extend(output)
                flag ^= 1
                frame = cv2.flip(frame, 1)
                frame = draw_video(frame, output)
                # 这里修改为展示输出
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.ui.video_label.setPixmap(QPixmap.fromImage(image))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    self.ui.video_label.clear()
                    break
            else:
                cap.release()
                self.ui.video_label.clear()
                # 识别物品去重
                for i in range(len(SumOutput)):
                    if SumOutput[i]["rec_docs"] not in videoImageSet:
                        self.VideoChild.line_1.append(SumOutput[i]["rec_docs"] + "\n")

                # 当前文本检查
                for every in videoImageSet:
                    if every not in thingsOfText:
                        thingsOfText.add(every)
                        self.VideoChild.line_1.append(every + "\n")
                break
        # print("here is error")
        cap.release()
        self.ui.video_label.clear()




    # 更新索引库创建子线程
    def UpdateGallary(self):
        command = r'python deploy/python/build_gallery.py -c deploy/configs/build_general.yaml -o IndexProcess.data_file="deploy/gallery_label.txt" -o IndexProcess.index_dir="deploy/index/" '
        os.system(command)
        print('更新完成')
        return 2


    # 修改5 增加方法，添加图片
    def add_img(self, img_path, img_name):
        from PIL import Image
        PIL_img = Image.open(img_path)

        # !!!!操他妈的，这段代码未知错误，debug差不多一个小时！！！！！！！！
        # basename=os.path.basename(PIL_img)

        # 后来发现写错参数了。。。。
        basename = os.path.basename(img_path)
        PIL_img.save(os.path.join('deploy/gallery', basename))

        with open('deploy/gallery_label.txt', 'a', encoding='utf-8') as f:
            f.write('\n' + 'gallery/' + basename + '\t' + img_name)

    # def update_gallery(self):
    #     # 更新索引库
    #     command = r'python deploy/python/build_gallery.py -c deploy/configs/build_general.yaml -o IndexProcess.data_file="deploy/gallery_update.txt" -o IndexProcess.index_dir="deploy/index/" '
    #     os.system(command)
    #     print('更新完成')


if __name__ == '__main__':
    QImageReader.supportedImageFormats()
    app = QApplication(sys.argv)
    app.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))
    startWindow = StartWindow()
    startWindow.ui.show()

    # child = Child()
    # child.show()
    sys.exit(app.exec_())

