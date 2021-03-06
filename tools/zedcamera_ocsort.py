import argparse
import os
import os.path as osp
import time
import cv2
import torch
import pdb

import sys
import numpy as np
import pyzed.sl as sl
import time

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer

from threading import Lock, Thread

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser

# zedcamera thread control signal
exit_signal = False
new_data = False

# Tread lock intialize
lock = Lock()

# image processing fromo zed api to numpy array type
def load_image_into_numpy_array(image):
    """_summary_

    Args:
        image (zedAPI image): image made from zedAPI

    Returns:
        numpy array: numpy image
    """
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_depth_into_numpy_array(depth):
    """_summary_

    Args:
        depth image (zedAPI image): depth image made from zedAPI

    Returns:
        numpy array: numpy depth image
    """
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

# zed image output size
width = 704
height = 416
confidence = 0.35

# make global dummy image variables to contain zed images
image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_np_global = np.zeros([width, height, 4], dtype=np.float)

# ZED image capture thread function (zed cuda 분리를 위한 thread 생성, 안하면 pythorch와 cuda context 차이로 cuda error 생김)
def capture_thread_func(svo_filepath=None):
    global image_np_global, depth_np_global, exit_signal, new_data, point_cloud
    get_cloud = True
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    input_type = sl.InputType()
    if svo_filepath is not None:
        input_type.set_from_svo_file(svo_filepath)

    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.svo_real_time_mode = False


    # Open the camera
    err = zed.open(init)
    while err != sl.ERROR_CODE.SUCCESS:
        err = zed.open(init)
        print(err)
        exit(1)
    """
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    """
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width / 2
    image_size.height = image_size.height / 2
    #image_size.width = image_size.width 
    #image_size.height = image_size.height 
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()
    

    while not exit_signal:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, resolution=image_size)
            #zed.retrieve_measure(depth_image_zed, sl.MEASURE.XYZRGBA, resolution=image_size)
            lock.acquire()
            image_np_global = load_image_into_numpy_array(image_zed)
            #depth_np_global = load_depth_into_numpy_array(depth_image_zed)
            #print("np global", image_np_global.shape)
            if get_cloud:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)
            new_data = True
            lock.release()

        time.sleep(0.01)

    zed.close()


    
class Predictor(object):
    def __init__(
        self,
        args,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        #self.test_size = (1920, 1080) 테스트 파일 이미지 크기는 yolox/data/data_augment.py의 preproc 함수를 통해서 만들어짐
        self.device = device
        self.fp16 = fp16
        self.hw = [[100,180],[50,90],[25,45]]
        self.strides = [8, 16, 32]
        if trt_file is not None:
            from torch2trt import TRTModule
        
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            self.model = model_trt
        
        else:
            self.model = exp.get_model().to(args.device)
            
        #self.rgb_means = (0.485, 0.456, 0.406)
        #self.std = (0.229, 0.224, 0.225)
        self.rgb_means = None
        self.std = None

    # yolox decoder function. If you use TRT module, this decoder should be implemented separately
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    
    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        #print(f"height {height} width {width}")
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        preproc_start = time.time()
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        preproc_time = time.time() - preproc_start
        with torch.no_grad():
            timer.tic()
            mdl_start = time.time()
            outputs = self.model(img)
            if args.trt:
                self.decoder = self.decode_outputs
                outputs = self.decoder(outputs, dtype=outputs.type())
            mdl_time = time.time() - mdl_start
            pos_start = time.time()
            #print(outputs.shape)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            if outputs is not None:
                print(outputs[0].shape, len(outputs))
            pos_time = time.time() - pos_start
            print(f"preprocess time : {preproc_time} model time : {mdl_time} postprocess time : {pos_time}")
        return outputs, img_info
    
def depthflow_demo(predictor, vis_folder, current_time, args):
    global image_np_global, depth_np_global, new_data, exit_signal, point_cloud
    # 타임스텝 및 세이브 폴더
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    
    tracker = OCSort(det_thresh=args.track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte)
    timer = Timer()
    frame_id = 0
    results = []
    
    # Start the capture thread with the ZED input
    capture_thread = Thread(target=capture_thread_func, daemon=True)
    capture_thread.start()

    key = ' '
    prevTime = time.time()
    tottime = 0
    iter = 0
    fps = 0
    while key != 113 :
        # frame 분석 로거
        """
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        """
        img_time = pred_time = oc_time = 0
        if new_data:
            img_start = time.time()
            lock.acquire()
            image_ocv = np.copy(image_np_global)
            # depth_image_zed = np.copy(depth_np_global)
            new_data = False
            lock.release()
            img_time = time.time() - img_start
            pred_start = time.time()
            outputs, img_info = predictor.inference(image_ocv, timer)
            pred_time = time.time() - pred_start
            oc_start = time.time()
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                        )
                # timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], point_cloud, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            oc_time = time.time() - oc_start
        else:
            online_im = image_np_global
            #time.sleep(0.01)
        #print(f'img_time : {img_time} pred_time : {pred_time} oc_time : {oc_time}')
        curTime = time.time()
        tottime += (curTime - prevTime)
        iter += 1
        avgtime = tottime / iter
        fps = 1./avgtime
        #print(fps)
        prevTime = curTime
        
        cv2.imshow("image", online_im)
        key = cv2.waitKey(10)
    cv2.destroyAllWindows()
    # capture_thread.join()
    print("\nFINISH")
    
def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    
    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        # model.head.decode_in_inference = False
        # decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
        decoder = None
    else:
        trt_file = None
        decoder = None

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
        
    predictor = Predictor(args, exp, trt_file, decoder, args.device, args.fp16)
    # logger.info("Model Summary: {}".format(get_model_info(predictor.model, exp.test_size)))
    model = predictor.model
    model.eval()
    
    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)

        if args.fp16:
            model = model.half()  # to FP16

    current_time = time.localtime()
    if args.demo_type == "depthcam":
        depthflow_demo(predictor, vis_folder, current_time, args)
    else:
        print("choose proper demo_type. ex) depthcam ")
        exit(1)
        
if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)