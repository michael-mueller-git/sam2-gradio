import os
import cv2
import torch
import numpy as np
import gradio as gr
from image_segment import image_inference
from video_segment import video_interfrence, InterferenceFrame
from video_process import count_video_frame_total, get_video_frame, SegmentItemContainer, SegmentItem, ImageFrame
from remove_video_background import remove_background_execute
from glob import glob


wdir = os.path.dirname(__file__)
os.chdir(wdir)
# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]


image_examples = sorted(list(glob(os.path.join(os.curdir, 'images', '*.jpg'))))
video_examples = sorted(list(glob(os.path.join(os.curdir, 'images', '*.mp4'))))



# ---- Video Global Variables ----
current_origin_frame = None
# ---- End ----


# ---- Video Global Variables ----
# current video file
current_video_file = None
# item container
item_container = SegmentItemContainer.instance()
# ---- End ----

current_bgr_video_file = None

def add_mark(frame):
    if frame is None:
        return None
    result = frame.frame_data.copy()
    marker_size = 25
    marker_thickness = 3
    marker_default_width = 1200
    width = result.shape[0]
    ratio = width / marker_default_width
    marker_final_size = int(marker_size * ratio)
    if marker_final_size < 3:
        marker_final_size = 3
    marker_final_thickness = int(marker_thickness * ratio)
    if marker_final_thickness < 2:
        marker_final_thickness = 2
    for (x, y, label) in frame.point_set:
        cv2.drawMarker(result, (x, y), colors[label], markerType=markers[label], markerSize=marker_final_size, thickness=marker_final_thickness)
    return result


def process_origin_image(img):
    global current_origin_frame
    current_origin_frame = ImageFrame(img, 0)
    return [None, None, 'foreground']


def new_existing_items_dropdown(choices = []):
    return gr.Dropdown(label = 'Select item', info = 'Enter new item name -> Enter to add new item, or drop down to select item', choices = choices, type = 'value', allow_custom_value=True)


with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            '''# Gradio-WebUI for Segment Anything Model 2 (SAM 2) 🚀'''
        )
    with gr.Row():
        device = gr.Dropdown(choices = ['cuda', 'cpu'], type='value', value='cuda', label='Select Equipment', visible = False)
    
    with gr.Tab(label='Background Remove'):
        with gr.Row(equal_height=True):
            with gr.Column():
                bgr_input_video = gr.Video(label='Source video', value=current_bgr_video_file)
        with gr.Row():
            bgr_commit_btn = gr.Button("Remove Background")
        with gr.Row():
            bgr_output_video = gr.Video(format='mp4', label='Output Video', interactive=False)


    with gr.Tab(label='Image Segmentation'):
        with gr.Row(equal_height=True):
            with gr.Column():
                input_image = gr.Image(type="numpy", label='Picture to be segment', format='png', image_mode='RGB')
            with gr.Column():
                with gr.Tab(label='Overlay'):
                    output_image = gr.Image(type='numpy', format='png', image_mode='RGB')
                with gr.Tab(label='Foreground only'):
                    output_mask = gr.Image(type='numpy', format='png', image_mode='RGBA')
        with gr.Row(equal_height = True):
            with gr.Column():
                with gr.Row():
                    gr.Markdown('Select a sample image or upload an image.')
                    undo_point_btn = gr.Button('Undo Marker Points')
                    remove_points_btn = gr.Button('Remove marker points')
                
                bg_radio = gr.Radio(['foreground', 'background'], label='Object Type')

                image_ex = gr.Examples(
                    examples=image_examples,
                    examples_per_page=20,
                    inputs=input_image,
                    fn=process_origin_image,
                    outputs=[output_image, output_mask, bg_radio],
                    run_on_click = True
                )
                image_commit_btn = gr.Button("Partitioning of objects")
            with gr.Column():
                gr.Image(visible = False)


    with gr.Tab(label='Video Segmentation'):
        with gr.Row(equal_height=True):
            with gr.Column():
                with gr.Row():
                    with gr.Column(variant = 'panel'):
                        gr.Markdown('### Step 1: Select/upload source video')
                        input_video = gr.Video(label='Source video', value=current_video_file)
                        gr.Examples(
                            label = 'Source Video',
                            examples=video_examples,
                            inputs=input_video,
                            examples_per_page=20,
                        )
                with gr.Row():
                    with gr.Column(variant = 'panel'):
                        gr.Markdown('### Step 3: Submit labelling results for video segmentation')
                        output_video = gr.Video(format='mp4', label='Output Video', interactive=False)
                        button_video = gr.Button(value = 'Video Segmentation')

            with gr.Column():
                gr.Markdown('### Step 2: Editing Segmentation Items')
                with gr.Row():
                    with gr.Column(variant='panel'):
                        maximum = count_video_frame_total(current_video_file) if current_video_file is not None else 0
                        origin_frame = gr.Image(label='Preview', type='numpy', interactive=False,value=get_video_frame(current_video_file, 0) if current_video_file is not None else None)
                        origin_slider = gr.Slider(label='Select video frame', maximum = maximum, value = 0, step=1)

                with gr.Row():
                    with gr.Column(variant='panel'):
                        gr.Markdown('Drag the upper slider to select the source frame containing the target item, create a new item or select an existing item from the ‘Select Item’ drop-down box, and click ‘Add Item’ to add it to the ‘Item Frame Preview')
                    with gr.Column(variant='panel'):
                        existing_items = new_existing_items_dropdown()
                        existing_item_btn = gr.Button('Add items')

                with gr.Row(equal_height=True):
                    with gr.Column(variant='panel'):

                        with gr.Row(equal_height=True): 
                            with gr.Column(variant='panel'):
                                item_frame_preview = gr.Image(label='Preview', interactive = False, sources=[])
                                origin_frame_preview = gr.Image(label='Original preview', interactive = False, sources=[], visible = False)
                                with gr.Row(equal_height=True):
                                    video_mark_radio = gr.Radio(['foreground', 'background'], label='Marker type', value='foreground', type='index', interactive=True)
                                with gr.Row(equal_height=True):
                                    item_frame_slider = gr.Slider(label='Select item frame', scale=5)
                                    item_origin_frame = gr.Number(label='Frame Number', value=0, scale=1, min_width=10, interactive=False)
                                with gr.Row():
                                    gr.Markdown('Undo recently added marker points')
                                    undo_vedio_button = gr.Button(value='Undo Marker')

                        with gr.Row(equal_height=True): 
                            with gr.Column(variant='panel'):
                                item_seg_preview = gr.Image(label='Item Split Preview', interactive = False)
                                with gr.Row():
                                    gr.Markdown('Click to view split preview results')
                                    item_seg_btn = gr.Button(value='Generate Preview')

                        with gr.Tab(label='current item'):
                            with gr.Row():
                                item_id = gr.Number(label='Item ID', value=10, scale=1, min_width=10, interactive=False)
                                item_name = gr.TextArea(label='Name of item', value='', lines=1, max_lines=1, scale=1, min_width=20, interactive=False)
                            with gr.Row():
                                item_delete_button = gr.Button('Deleting items')

                        with gr.Row(equal_height=True):
                            def update_current_item(it_id, it_name):
                                global item_container
                                item_container.select_by_item_id(it_id)
                                cur_item = item_container.current_item()
                                cur_frame = None
                                if cur_item is not None:
                                    cur_frame = cur_item.current_frame()
                                frame_preview = None
                                frame_data = None
                                if cur_frame and cur_frame.frame_data is not None:
                                    frame_data = cur_frame.frame_data.copy()
                                if cur_frame:
                                    frame_preview = gr.Image(label='Item frame preview', interactive = True, value=add_mark(cur_frame), sources=[])
                                else:
                                    frame_preview = gr.Image(label='Item frame preview', interactive = False)
                                    
                                frame_slider = None
                                if cur_item:
                                    maximum = len(cur_item) - 1
                                    if maximum == 0:
                                        maximum = 1
                                    frame_slider = gr.Slider(value=cur_item.current_index, minimum = 0, maximum = maximum, label='Select item frame', scale=5, interactive = True)
                                else:
                                    frame_slider = gr.Slider(value=0, minimum = 0, maximum = 1, label='Select item frame', scale=5, interactive = False)
                                origin_frame = 0
                                if cur_frame:
                                    origin_frame = cur_frame.origin_index

                                seg_preview = None
                                if cur_frame:
                                    seg_preview = cur_frame.preview_data
                                
                                origin_frame_data = None
                                if cur_frame and cur_frame.frame_data is not None:
                                    origin_frame_data = cur_frame.frame_data

                                return [frame_preview, origin_frame_data, frame_slider, origin_frame, seg_preview, it_id, it_name]
                            all_preview_image_widgets = [item_frame_preview, origin_frame_preview, item_frame_slider, item_origin_frame, item_seg_preview, item_id, item_name]
                            all_items_ex = gr.Examples(
                                label='Selection of marked items',
                                examples = [[-1, '']],
                                inputs = [item_id, item_name],
                                fn = update_current_item,
                                run_on_click = True,
                                outputs = all_preview_image_widgets,
                            )


    
    
    # 图片分割事件处理函数

    gr.Image.input(input_image, process_origin_image, input_image, [output_image, output_mask, bg_radio])
    

    def add_mark_point(point_type, event: gr.SelectData):
        global current_origin_frame
        label = 1
        if point_type == 'foreground':
            label = 1
        elif point_type == 'background':
            label = 0
        
        if current_origin_frame is None:
            return None
        
        current_origin_frame.add(*event.index, label)
        return add_mark(current_origin_frame)

    gr.Image.select(input_image, add_mark_point, inputs = [bg_radio], outputs = [input_image])


    def undo_last_point():
        global current_origin_frame
        if current_origin_frame is None:
            return None
        current_origin_frame.pop()
        return add_mark(current_origin_frame)
    gr.Button.click(undo_point_btn, undo_last_point, inputs=None, outputs=input_image)


    def remove_all_points():
        global current_origin_frame
        if current_origin_frame is None:
            return None
        current_origin_frame.clear()
        return current_origin_frame.frame_data, 'foreground'

    gr.Button.click(remove_points_btn, remove_all_points, inputs=None, outputs=[input_image, bg_radio])

    
    def do_image_interference(device):
        global current_origin_frame
        if current_origin_frame is None:
            gr.Warning('Please select the image first', duration = 3)
            return None, None
        points = []
        if current_origin_frame is not None:
            for x, y, label in current_origin_frame.point_set:
                points.append(((x, y), label))
        
        return image_inference(device, current_origin_frame.frame_data, points)

    gr.Button.click(image_commit_btn, do_image_interference, inputs=[device], outputs=[output_image, output_mask])
    
    # 视频分割事件处理函数


    def change_video(path):
        global current_video_file
        global item_container
        if path != current_video_file:
            item_container.clear()
        current_video_file = path
        maximum = count_video_frame_total(current_video_file)
        slider = gr.Slider(minimum=0, maximum=maximum, label='Preview of the source frame', value=0, step=1, interactive=True)
        return get_video_frame(current_video_file, 0), slider, *update_all_preview_widgets(True)


    def change_origin_preview(value):
        global current_video_file
        return get_video_frame(current_video_file, value - 1)
                

    def update_all_preview_widgets(update_all):
        global item_container
        results = []
        cur_item = item_container.current_item()
        
        if update_all:
            all_items = [(item.name, item.item_id) for item in item_container]
            item_names =  new_existing_items_dropdown(all_items)
            example_all_items = [(item[1], item[0]) for item in all_items]
            results = [item_names, gr.Dataset(samples = example_all_items)]

        if cur_item is None:
            slider = gr.Slider(value=0, minimum = 0, maximum = 1, label='Select item frame', scale=5, interactive = False)
            results.extend([None, None, slider, 0, None, 0, ''])
        else:
            results.extend(update_current_item(cur_item.item_id, cur_item.name))
        return results


    def attach_existing_item(item_name, frame_data, frame_index):
        if item_name is None:
            gr.Warning('There is no selected item, please add an item and select it first.', duration = 3)
            return update_all_preview_widgets(True) 

        item_container.select_by_item_name(item_name)
        frame = ImageFrame(frame_data = frame_data, origin_index = frame_index)
        cur_item = item_container.current_item()
        if cur_item is not None:
            cur_item.add_frame(frame)
        else:
            gr.Warning('There is no item currently selected, no item frame can be added.', duration = 3)
        return update_all_preview_widgets(True)


    def delete_current_item():
        global item_container
        item_container.remove_current()
        return update_all_preview_widgets(True)

    
    def change_current_frame(value):
        global item_container
        cur_item = item_container.current_item()
        result = [None, 0, None]
        if not cur_item:
            return result
        cur_item.select_frame(value)
        
        if not cur_item:
            return result
        
        cur_frame = cur_item.current_frame()
        return add_mark(cur_frame), cur_frame.frame_data, cur_frame.origin_index, cur_frame.preview_data


    def clear_current_frame():
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            return update_all_preview_widgets(False)
        
        cur_item.remove_current()
        return update_all_preview_widgets(False)


    def get_video_points(img, point_type, evt: gr.SelectData):
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            return img
        
        cur_frame = cur_item.current_frame()
        if not cur_frame:
            return img
        
        p_type = 0
        if point_type is not None:
            p_type = point_type
        cur_frame.add(*evt.index, 1 - p_type)
        return add_mark(cur_frame)


    def undo_video_points(img):
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            return img
        
        cur_frame = cur_item.current_frame()
        if not cur_frame:
            return img
        cur_frame.pop()
        return add_mark(cur_frame)
    
    
    def run_sample_inference(device):
        global item_container
        cur_item = item_container.current_item()
        if not cur_item:
            gr.Warning('No item selected, please set item', duration = 3)
            return None
        cur_frame = cur_item.current_frame()
        if not cur_frame:
            gr.Warning('No preview frame selected, please select a preview frame', duration = 3)
            return None
        if cur_frame.preview_data:
            return cur_frame.preview_data
        point_data = []
        for x, y, label in cur_frame.point_set:
            point_data.append(((x, y), label))
        result, _ = image_inference(device, cur_frame.frame_data, point_data)
        cur_frame.preview_data = result
        return result
    
    
    def segment_video(video_path):
        global item_container
        output_path = 'output'
        frames = []
        width = 0
        height = 0
        for item in item_container:
            for frame in item:
                data = InterferenceFrame()
                data.origin_frame_id = frame.origin_index
                data.item_id = item.item_id
                data.point_set = frame.point_set
                if width == 0 and frame.frame_data is not None:
                    width = frame.frame_data.shape[0]
                    height = frame.frame_data.shape[1]
                frames.append(data)
        return video_interfrence(video_path, output_path, frames, width, height)


    def add_or_select_item(value, frame_data, frame_index):
        global item_container        
        result = value
        item_index = value
        update_choices = False

        if isinstance(value, int):
            item_index = value
            item_container.select_item(item_index)
        elif isinstance(value, str) and value:
            update_all = True
            if not item_container.exists(value):
                new_item = SegmentItem.create(value)
                item_index = new_item.item_id
                item_container.add_item(new_item)
            else:
                item_container.select_by_item_name(value)
            choices = [(item.name, item.item_id) for item in item_container]
            result = new_existing_items_dropdown(choices)
        return result


    all_preview_widgets = [existing_items, all_items_ex.dataset, *all_preview_image_widgets]
    gr.Video.change(input_video, change_video, inputs=input_video, outputs=[origin_frame, origin_slider, *all_preview_widgets])
    gr.Slider.change(origin_slider, change_origin_preview, inputs=origin_slider, outputs=origin_frame)
    gr.Button.click(existing_item_btn, attach_existing_item, [existing_items, origin_frame, origin_slider], all_preview_widgets)
    gr.Dropdown.input(existing_items, add_or_select_item, [existing_items, origin_frame, origin_slider], existing_items)
    gr.Slider.change(item_frame_slider, change_current_frame,
        inputs=[item_frame_slider], 
        outputs=[item_frame_preview, 
            origin_frame_preview, 
            item_origin_frame, 
            item_seg_preview])
    gr.Button.click(item_delete_button, delete_current_item, None, all_preview_widgets)
    gr.Image.clear(item_frame_preview, clear_current_frame, None, all_preview_image_widgets)
    gr.Image.select(item_frame_preview, get_video_points, [item_frame_preview, video_mark_radio], item_frame_preview)
    gr.Button.click(undo_vedio_button, undo_video_points, item_frame_preview, item_frame_preview)
    gr.Button.click(item_seg_btn, run_sample_inference, inputs=[device], outputs=[item_seg_preview])
    gr.Button.click(button_video, segment_video, input_video, output_video)
    

    gr.Button.click(bgr_commit_btn, remove_background_execute, bgr_input_video, output_video)


if __name__ == '__main__':
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, debug=True, quiet=False)
