# 说明

## 关于代码
1. app.py 为入口
2. image_segment.py 为图片分割逻辑;
3. video_segment.py 为视频分割逻辑;
4. video_process.py 类似与MVC模式的Model层，用于管理物品标记等数据。

## 关于素材
1. 图片和视频素材需要放到 images 目录下

## 其他依赖
1. ffmpeg命令，用于拆图片、合视频

## 后续计划
1. 该代码为原型预览之目的，UI交互体验较差；
2. 整合包内代码做了优化，为收集反馈之用；如果正面反馈较多，将逐步开源；