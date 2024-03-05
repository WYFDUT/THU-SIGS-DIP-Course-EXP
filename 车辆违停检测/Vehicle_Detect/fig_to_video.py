import os
import cv2
from moviepy.editor import *
from PIL import Image


class VideoShow:
    def __init__(self, no_video, src_path, video_save_path, fps, no_reshape):
        """
        利用OpenCV将单帧图片拼接为视频类
        :param no_video: 是否需要输出视频
        :param src_path: 图片存放的根目录
        :param video_save_path: 视频保存的路径
        :param fps: 输出的视频帧数
        :param no_reshape: 输入图片是否需要重设大小
        """
        self.no_video = no_video
        self.src_path = src_path
        self.no_reshape = no_reshape
        self.img_path = [name for name in os.listdir(self.src_path)]
        self.img_path.sort()
        self.init_image = cv2.imread(os.path.join(self.src_path, self.img_path[0]))
        self.image_height = self.init_image.shape[0]
        self.image_width = self.init_image.shape[1]
        self.video_save_path = os.path.join(video_save_path, ('test'+'.mp4'))
        # print(self.image_height, self.image_width)
        # 设置视频写入器
        # self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，
        # 第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        self.video_write = cv2.VideoWriter(self.video_save_path, self.fourcc, fps, (self.image_width, self.image_height))
        # 临时存放图片的数组
        self.img_array = []

    def figure_to_video(self):
        """
        将图片拼接为视频
        :return: 视频保存地址
        """
        print(self.image_height, self.image_width)
        for i in range(len(self.img_path)):
            filename = 'test'+str(i)+'.png'
            #print(filename)
            if not self.no_reshape:
                self.figure_reshape(os.path.join(self.src_path, filename), os.path.join(self.src_path, filename))
            img = cv2.imread(os.path.join(self.src_path, filename))
            self.img_array.append(img)
        for i in range(len(self.img_path)):
            #self.img_array[i] = cv2.resize(self.img_array[i], (self.image_width, self.image_height))
            # print(self.img_array[i].shape[0], self.img_array[i].shape[1])
            self.video_write.write(self.img_array[i])
        self.video_writer_close()
        return self.video_save_path

    @staticmethod
    def figure_reshape(img_path, out_path):
        """
        图片大小重设(静态方法)
        """
        img = Image.open(img_path)
        region = img.crop((350, 50, 900, 500))
        region.save(out_path)

    def video_writer_close(self):
        """
        视频写入器关闭
        """
        self.video_write.release()


class VideoClip:
    def __init__(self, gt_video_path, lane_video_path, output_path):
        """
        利用MoviePy实现视频剪切拼接类
        :param gt_video_path: Ground Truth视频存放地址
        :param lane_video_path: 车道线检测结果视频存放地址
        :param output_path: 输出视频路径
        """
        self.gt_video_path = gt_video_path
        self.lane_video_path = lane_video_path
        self.output_path = output_path

    def video_clip(self):
        """
        视频剪切拼接
        """
        clip1 = VideoFileClip(self.gt_video_path)
        size = (int(clip1.size[0] / 20.0) * 10, int(clip1.size[1] / 20.0) * 10)
        clip2 = VideoFileClip(self.lane_video_path).resize(size).set_position(
            (clip1.size[0] - size[0], 0))
        clip_final = CompositeVideoClip([clip1, clip2])
        clip_final.write_videofile(self.output_path)
        clip1.close()
        clip2.close()
        clip_final.close()


if __name__ == "__main__":
    video_class = VideoShow(no_video=False, src_path='/root/wyf/airs/fig_results/mydata/unetformer_lsk_s/seq2/Labels',
                            video_save_path='/root/wyf/airs/fig_results/mydata/unetformer_lsk_s/video',
                            fps=30, no_reshape=True)
    video_class.figure_to_video()