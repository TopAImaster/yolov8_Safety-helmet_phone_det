import cv2

def trim_video(input_file, output_file, start_frame, end_frame):
    # 打开输入视频文件
    input_video = cv2.VideoCapture(input_file)

    # 检查视频文件是否成功打开
    if not input_video.isOpened():
        print("无法打开输入视频文件")
        return

    # 获取视频的相关信息
    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置截取的起始和结束帧
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))

    # 创建输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    # 读取并写入所需的帧范围
    frame_count = 0
    while frame_count <= end_frame:
        ret, frame = input_video.read()
        if not ret:
            break

        # 如果帧在所需范围内，则写入输出视频文件
        if start_frame <= frame_count <= end_frame:
            output_video.write(frame)

        frame_count += 1

    # 释放资源
    input_video.release()
    output_video.release()

    print("截取完成")

# 示例用法
input_file = "1.mp4"
output_file = "output_v.mp4"
start_frame = 100  # 起始帧
end_frame = 150   # 结束帧

trim_video(input_file, output_file, start_frame, end_frame)