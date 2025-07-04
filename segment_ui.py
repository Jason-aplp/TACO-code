import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QWidget, QFileDialog, QSpinBox, QGroupBox, QRadioButton)
from PyQt5.QtCore import QTimer
from sam2.build_sam import build_sam2_video_predictor
from skimage import measure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # Optimize figure rendering
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        super(MplCanvas, self).__init__(self.fig)


class SAM2AnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Video Annotation Tool")
        self.setGeometry(200, 200, 2000, 1500)
        
        # Optimize Matplotlib settings
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 1.0
        plt.rcParams['agg.path.chunksize'] = 10000
        
        # Initialize variables
        self.data_dir = ""
        self.frame_names = []
        self.current_frame_idx = 0
        self.inference_state = None
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ann_obj_id = 0  # Starting from 0 instead of 1
        self.video_segments = {}
        self.point_mode = "positive"  # Default mode for point adding
        self.add_point_mode = False   # Flag for adding points mode
        
        # Add storage for each object ID's prompts
        self.object_prompts = {
            0: {
                0: {'points': [], 'labels': [], 'box': None}
            }
        }  # {obj_id: {'points': [], 'labels': [], 'box': None}}
        self.segmented_objects = set()

        # Frame cache
        self.frame_cache = {}
        self.max_cache_size = 100  # Maximum number of cached frames
        
        # Animation variables
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animation_step)
        self.animation_playing = False
        self._animation_last_update = 0
        self.animation_interval = 1  # Show every Nth frame (default: show every frame)
        self.animation_counter = 0
        
        # Setup UI
        self.init_ui()
        
        # Setup SAM2 model
        self.setup_sam2()
        
    def setup_sam2(self):
        try:
            # Ask user for the checkpoint file location if not found
            sam2_checkpoint = "/home/yuliu/Projects/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
            while not os.path.exists(sam2_checkpoint):
                sam2_checkpoint = QFileDialog.getOpenFileName(
                    self, "Select SAM2 Checkpoint File", "", "Model Files (*.pt)")[0]
                if not sam2_checkpoint:  # User canceled
                    self.status_label.setText("SAM2 model loading canceled. Please restart application.")
                    return
            
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
            self.status_label.setText("SAM2 model loaded successfully")
        except Exception as e:
            self.status_label.setText(f"Error loading SAM2 model: {str(e)}")
    
    def init_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left panel for controls - increase width
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        # Input file path section
        path_group = QGroupBox("File Path Settings")
        path_layout = QVBoxLayout()
        
        self.path_input = QLineEdit()
        self.path_button = QPushButton("Browse...")
        self.path_button.clicked.connect(self.browse_directory)
        
        path_layout.addWidget(QLabel("Input Directory:"))
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.path_button)
        
        self.save_path_input = QLineEdit()
        path_layout.addWidget(QLabel("Save Subfolder:"))
        path_layout.addWidget(self.save_path_input)
        
        path_group.setLayout(path_layout)
        left_layout.addWidget(path_group)
        
        # Frame navigation section
        frame_group = QGroupBox("Frame Navigation")
        frame_layout = QVBoxLayout()
        
        frame_buttons = QHBoxLayout()
        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.prev_frame)
        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.next_frame)
        frame_buttons.addWidget(self.prev_button)
        frame_buttons.addWidget(self.next_button)
        
        frame_jump = QHBoxLayout()
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(0)
        self.frame_spinbox.setMaximum(0)  # Will be updated when loading data
        self.frame_spinbox.valueChanged.connect(self.jump_to_frame)
        self.jump_button = QPushButton("Jump")
        self.jump_button.clicked.connect(self.jump_to_frame)
        frame_jump.addWidget(QLabel("Frame:"))
        frame_jump.addWidget(self.frame_spinbox)
        frame_jump.addWidget(self.jump_button)
        
        # Animation controls
        animation_layout = QHBoxLayout()
        self.animation_button = QPushButton("Animation")
        self.animation_button.clicked.connect(self.toggle_animation)
        
        # Change FPS to interval (show every Nth frame)
        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setMinimum(1)
        self.interval_spinbox.setMaximum(30)
        self.interval_spinbox.setValue(1)  # Default: show every frame
        self.interval_spinbox.valueChanged.connect(self.update_interval)
        
        animation_layout.addWidget(self.animation_button)
        animation_layout.addWidget(QLabel("Interval:"))
        animation_layout.addWidget(self.interval_spinbox)
        
        frame_layout.addLayout(frame_buttons)
        frame_layout.addLayout(frame_jump)
        frame_layout.addLayout(animation_layout)
        frame_group.setLayout(frame_layout)
        left_layout.addWidget(frame_group)
        
        # Annotation tools section
        annotation_group = QGroupBox("Annotation Tools")
        annotation_layout = QVBoxLayout()
        
        # Object ID input
        obj_id_layout = QHBoxLayout()
        obj_id_layout.addWidget(QLabel("Object ID:"))
        self.obj_id_spinbox = QSpinBox()
        self.obj_id_spinbox.setMinimum(0)  # Starting from 0 instead of 1
        self.obj_id_spinbox.setMaximum(100)
        self.obj_id_spinbox.setValue(0)  # Default to 0
        self.obj_id_spinbox.valueChanged.connect(self.update_object_id)
        obj_id_layout.addWidget(self.obj_id_spinbox)
        annotation_layout.addLayout(obj_id_layout)
        
        # Point selection mode
        point_mode_layout = QHBoxLayout()
        self.pos_point_radio = QRadioButton("Positive Points")
        self.neg_point_radio = QRadioButton("Negative Points")
        self.pos_point_radio.setChecked(True)
        self.pos_point_radio.toggled.connect(self.set_point_mode)
        point_mode_layout.addWidget(self.pos_point_radio)
        point_mode_layout.addWidget(self.neg_point_radio)
        annotation_layout.addLayout(point_mode_layout)
        
        # Add point button
        self.add_point_button = QPushButton("Add Point")
        self.add_point_button.clicked.connect(self.enable_point_selection)
        annotation_layout.addWidget(self.add_point_button)
        
        # Add annotation buttons
        self.add_bbox_button = QPushButton("Add Bounding Box")
        self.add_bbox_button.clicked.connect(self.enable_bbox_selection)
        annotation_layout.addWidget(self.add_bbox_button)
        
        self.clear_annotations_button = QPushButton("Clear Annotations")
        self.clear_annotations_button.clicked.connect(self.clear_annotations)
        annotation_layout.addWidget(self.clear_annotations_button)
        
        # Reset state button
        self.reset_state_button = QPushButton("Reset State")
        self.reset_state_button.clicked.connect(self.reset_state)
        annotation_layout.addWidget(self.reset_state_button)
        
        annotation_group.setLayout(annotation_layout)
        left_layout.addWidget(annotation_group)
        
        # Segmentation section
        segment_group = QGroupBox("Segmentation")
        segment_layout = QVBoxLayout()
        
        self.segment_current_button = QPushButton("Segment Current Frame")
        self.segment_current_button.clicked.connect(self.segment_current_frame)
        segment_layout.addWidget(self.segment_current_button)
        
        self.segment_all_button = QPushButton("Segment All Frames")
        self.segment_all_button.clicked.connect(self.segment_all_frames)
        segment_layout.addWidget(self.segment_all_button)
        
        segment_group.setLayout(segment_layout)
        left_layout.addWidget(segment_group)
        
        # Save section
        save_group = QGroupBox("Save Results")
        save_layout = QVBoxLayout()
        
        self.save_button = QPushButton("Save Masks")
        self.save_button.clicked.connect(self.save_masks)
        save_layout.addWidget(self.save_button)
        
        save_group.setLayout(save_layout)
        left_layout.addWidget(save_group)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)  # Allow word wrap for status messages
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(600)  # Increase width from 300 to 400
        
        # Right panel for image display
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.canvas)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize UI state
        self.update_ui_state()
    
    def update_ui_state(self):
        has_data = len(self.frame_names) > 0
        
        # Update frame navigation controls
        self.prev_button.setEnabled(has_data and self.current_frame_idx > 0 and not self.animation_playing)
        self.next_button.setEnabled(has_data and self.current_frame_idx < len(self.frame_names) - 1 and not self.animation_playing)
        self.frame_spinbox.setEnabled(has_data and not self.animation_playing)
        self.jump_button.setEnabled(has_data and not self.animation_playing)
        self.animation_button.setEnabled(has_data)
        self.interval_spinbox.setEnabled(has_data)
        
        # Update animation button text
        if self.animation_playing:
            self.animation_button.setText("Stop")
        else:
            self.animation_button.setText("Animation")
        
        # Update annotation controls
        annotation_enabled = has_data and self.inference_state is not None and not self.animation_playing
        self.add_point_button.setEnabled(annotation_enabled)
        self.add_bbox_button.setEnabled(annotation_enabled)
        self.pos_point_radio.setEnabled(annotation_enabled)
        self.neg_point_radio.setEnabled(annotation_enabled)
        self.obj_id_spinbox.setEnabled(annotation_enabled)
        has_annotations = False
        if self.ann_obj_id in self.object_prompts:
            if self.current_frame_idx in self.object_prompts[self.ann_obj_id]:
                has_annotations = len(self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points']) > 0 or self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] is not None
        self.clear_annotations_button.setEnabled(annotation_enabled and has_annotations)
        self.reset_state_button.setEnabled(annotation_enabled)
        
        # Update segmentation controls
        self.segment_current_button.setEnabled(annotation_enabled and has_annotations)
        self.segment_all_button.setEnabled(annotation_enabled and has_annotations)
        
        # Update save controls
        has_segments = len(self.video_segments) > 0
        self.save_button.setEnabled(has_segments and not self.animation_playing)
    
    def update_object_id(self):
        self.ann_obj_id = self.obj_id_spinbox.value()
        if self.ann_obj_id not in self.object_prompts:
            self.object_prompts[self.ann_obj_id] = {}
            self.object_prompts[self.ann_obj_id][self.current_frame_idx] = {}
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points'] = []
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['labels'] = []
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] = None
        self.status_label.setText(f"Object ID set to {self.ann_obj_id}")
        self.display_current_frame()  # Refresh display to show masks for the selected object ID
    
    def set_point_mode(self):
        if self.pos_point_radio.isChecked():
            self.point_mode = "positive"
        else:
            self.point_mode = "negative"
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Video Frames")
        if directory:
            self.path_input.setText(directory)
            self.load_frames(directory)
    
    def load_frames(self, directory):
        try:
            self.data_dir = directory
            # Get only image files
            self.frame_names = [
                p for p in os.listdir(directory)
                if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
            ]
            
            if not self.frame_names:
                self.status_label.setText("No image files found in the selected directory")
                return
            
            # Sort frames by number
            self.frame_names.sort()
            
            # Initialize segmentation state
            self.current_frame_idx = 0
            self.frame_spinbox.setMaximum(len(self.frame_names) - 1)
            self.frame_spinbox.setValue(0)
            
            # Clear frame cache
            self.frame_cache = {}
            
            # Initialize inference state
            if self.predictor:
                self.inference_state = self.predictor.init_state(video_path=directory)
                self.predictor.reset_state(self.inference_state)
            self.ann_obj_id = 0  # Start from 0
            self.obj_id_spinbox.setValue(0)
            self.object_prompts = {0: {0: {'points': [], 'labels': [], 'box': None}}}
            self.segmented_objects = set()
            self.video_segments = {}
            
            # Preload first few frames to cache
            self.preload_frames(0, min(10, len(self.frame_names)))
            
            # Display first frame
            self.display_current_frame()
            self.status_label.setText(f"Loaded {len(self.frame_names)} frames")
            
        except Exception as e:
            self.status_label.setText(f"Error loading frames: {str(e)}")
        
        self.update_ui_state()
    
    def preload_frames(self, start_idx, end_idx):
        """Preload specified range of frames to cache"""
        for idx in range(start_idx, min(end_idx, len(self.frame_names))):
            if idx not in self.frame_cache:
                try:
                    frame_path = os.path.join(self.data_dir, self.frame_names[idx])
                    img = Image.open(frame_path)
                    self.frame_cache[idx] = img
                    
                    # Limit cache size
                    if len(self.frame_cache) > self.max_cache_size:
                        # Remove frames furthest from current frame
                        cache_keys = list(self.frame_cache.keys())
                        cache_keys.sort(key=lambda x: abs(x - self.current_frame_idx))
                        del self.frame_cache[cache_keys[-1]]
                except Exception as e:
                    print(f"Error preloading frame {idx}: {e}")
    
    def get_frame_from_cache(self, idx):
        """Get frame from cache, load if not present"""
        if idx not in self.frame_cache:
            try:
                frame_path = os.path.join(self.data_dir, self.frame_names[idx])
                img = Image.open(frame_path)
                self.frame_cache[idx] = img
                
                # Limit cache size
                if len(self.frame_cache) > self.max_cache_size:
                    cache_keys = list(self.frame_cache.keys())
                    cache_keys.sort(key=lambda x: abs(x - self.current_frame_idx))
                    del self.frame_cache[cache_keys[-1]]
            except Exception as e:
                print(f"Error loading frame {idx}: {e}")
                return None
        
        return self.frame_cache[idx]
    
    def display_current_frame(self):
        if not self.frame_names or self.current_frame_idx >= len(self.frame_names):
            return
        
        # Get frame from cache or load it
        img = self.get_frame_from_cache(self.current_frame_idx)
        if img is None:
            return
        
        # Clear axis and display current frame
        self.canvas.axes.clear()
        
        # Disable axis ticks to improve performance
        self.canvas.axes.set_xticks([])
        self.canvas.axes.set_yticks([])
        
        # Display image
        self.canvas.axes.imshow(img)
        self.canvas.axes.set_title(f"Frame {self.current_frame_idx}: {self.frame_names[self.current_frame_idx]}", fontsize=10)
        
        # Optimize display in animation mode
        if self.animation_playing:
            # Only show mask in animation mode, not points or box
            if self.current_frame_idx in self.video_segments:
                for obj_id, mask in self.video_segments[self.current_frame_idx].items():
                    self.show_mask(mask, self.canvas.axes, obj_id=obj_id)
        else:
            # Show all elements in non-animation mode
            for obj_id, prompts in self.object_prompts.items():
                if self.current_frame_idx not in prompts:
                    continue
                else:
                    prompts = prompts[self.current_frame_idx]
                points, labels, box = prompts['points'], prompts['labels'], prompts['box']
            
                if points:
                    points_array = np.array(points)
                    labels_array = np.array(labels)
                    self.show_points(points_array, labels_array, self.canvas.axes)
            
                if box is not None:
                    self.show_box(box, self.canvas.axes)
            
            if self.current_frame_idx in self.video_segments:
                for obj_id, mask in self.video_segments[self.current_frame_idx].items():
                    self.show_mask(mask, self.canvas.axes, obj_id=obj_id)
        
        # Draw image
        self.canvas.draw()
        
        # Preload next frames
        if not self.animation_playing:
            next_idx = self.current_frame_idx + 1
            if next_idx < len(self.frame_names) and next_idx not in self.frame_cache:
                self.preload_frames(next_idx, next_idx + 3)
    
    def prev_frame(self):
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.frame_spinbox.setValue(self.current_frame_idx)
            self.display_current_frame()
            self.update_ui_state()
    
    def next_frame(self):
        if self.current_frame_idx < len(self.frame_names) - 1:
            self.current_frame_idx += 1
            self.frame_spinbox.setValue(self.current_frame_idx)
            self.display_current_frame()
            self.update_ui_state()
    
    def jump_to_frame(self):
        frame_idx = self.frame_spinbox.value()
        if 0 <= frame_idx < len(self.frame_names):
            self.current_frame_idx = frame_idx
            self.display_current_frame()
            self.update_ui_state()
    
    def toggle_animation(self):
        if self.animation_playing:
            # Stop animation
            self.animation_timer.stop()
            self.animation_playing = False
        else:
            # Preload some frames to cache before starting
            for i in range(self.current_frame_idx, min(self.current_frame_idx + 15, len(self.frame_names))):
                self.get_frame_from_cache(i)
            
            # Start animation
            self.animation_timer.start(30)  # Fixed 30ms timeout for consistent UI updates
            self.animation_playing = True
            self._animation_last_update = time.time() * 1000
            self.animation_counter = 0
        
        self.update_ui_state()
    
    def update_interval(self):
        # Update animation interval (show every Nth frame)
        self.animation_interval = self.interval_spinbox.value()
    
    def animation_step(self):
        # Skip frames based on interval setting
        self.animation_counter += 1
        
        # Move to next frame or loop
        if self.current_frame_idx < len(self.frame_names) - 1:
            self.current_frame_idx += 1
        else:
            # Loop back to first frame
            self.current_frame_idx = 0
        
        if self.current_frame_idx % self.animation_interval != 0:
            return
        # Preload next few frames
        next_idx = (self.current_frame_idx + 1) % len(self.frame_names)
        if next_idx not in self.frame_cache:
            QApplication.processEvents()  # Allow UI updates
            self.get_frame_from_cache(next_idx)
        
        # Update spinbox without triggering signals
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(self.current_frame_idx)
        self.frame_spinbox.blockSignals(False)
        
        # Display frame
        self.display_current_frame()
    
    def enable_point_selection(self):
        self.add_point_mode = True
        self.bbox_selection_active = False
        point_type = "positive" if self.point_mode == "positive" else "negative"
        self.status_label.setText(f"Click on the image to add a {point_type} point")
    
    def on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None or not self.inference_state or self.animation_playing:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        if self.current_frame_idx not in self.object_prompts[self.ann_obj_id]:
            self.object_prompts[self.ann_obj_id][self.current_frame_idx] = {'points': [], 'labels': [], 'box': None}
        if hasattr(self, 'bbox_selection_active') and self.bbox_selection_active:
            # Bounding box selection mode
            if not hasattr(self, 'bbox_start'):
                # First click - start of bounding box
                self.bbox_start = (x, y)
                self.status_label.setText(f"Bounding box started at ({x}, {y}). Click again to complete.")
            else:
                # Second click - end of bounding box
                x1, y1 = self.bbox_start
                x2, y2 = x, y
                self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
                self.bbox_selection_active = False
                delattr(self, 'bbox_start')
                self.status_label.setText(f"Bounding box added: {self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box']}")
                self.display_current_frame()
        elif self.add_point_mode:
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points'].append([x, y])
            label = 1 if self.point_mode == "positive" else 0
            self.object_prompts[self.ann_obj_id][self.current_frame_idx]['labels'].append(label)
            point_type = "positive" if label == 1 else "negative"
            self.status_label.setText(f"Added {point_type} point at ({x}, {y})")
            self.add_point_mode = False  # Turn off point mode after adding a point
            self.display_current_frame()
        
        self.update_ui_state()
    
    def enable_bbox_selection(self):
        self.bbox_selection_active = True
        self.add_point_mode = False
        self.status_label.setText("Click and drag to define a bounding box")
    
    def clear_annotations(self):
        self.object_prompts[self.ann_obj_id][self.current_frame_idx]['points'] = []
        self.object_prompts[self.ann_obj_id][self.current_frame_idx]['labels'] = []
        self.object_prompts[self.ann_obj_id][self.current_frame_idx]['box'] = None
        self.status_label.setText("Annotations cleared")
        self.display_current_frame()
        self.update_ui_state()
    
    def reset_state(self):
        if not self.inference_state or not self.predictor:
            return
        
        try:
            self.predictor.reset_state(self.inference_state)
            self.ann_obj_id = 0  # Start from 0
            self.obj_id_spinbox.setValue(0)
            self.object_prompts = {
                0: {
                    0: {'points': [], 'labels': [], 'box': None}
                }
            }
            self.segmented_objects = set()
            self.video_segments = {}
            self.status_label.setText("State reset successfully")
            self.current_frame_idx = 0
            self.frame_spinbox.setValue(0)
            self.display_current_frame()
        except Exception as e:
            self.status_label.setText(f"Error resetting state: {str(e)}")
        
        self.update_ui_state()
    
    def segment_current_frame(self):
        if not self.inference_state or not self.object_prompts:
            return
        
        try:
            for obj_id, prompts in self.object_prompts.items():
                if self.current_frame_idx not in prompts:
                    continue
                else:
                    prompts = prompts[self.current_frame_idx]
                self.segmented_objects.add(obj_id)
                # Apply segmentation
                points_array = np.array(prompts['points'], dtype=np.float32) if prompts['points'] else None
                labels_array = np.array(prompts['labels'], dtype=np.int32) if prompts['labels'] else None
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame_idx,
                    obj_id=obj_id,
                    points=points_array,
                    labels=labels_array,
                    box=prompts['box'],
                )
            
                # Store and display the result
                mask = (out_mask_logits[obj_id] > 0.0).cpu().numpy()
                if self.current_frame_idx not in self.video_segments:
                    self.video_segments[self.current_frame_idx] = {}
                self.video_segments[self.current_frame_idx][obj_id] = mask
            
            self.display_current_frame()
            self.status_label.setText(f"Segmentation completed in frame {self.current_frame_idx}")
            
        except Exception as e:
            self.status_label.setText(f"Segmentation error: {str(e)}")
        
        self.update_ui_state()
    
    def segment_all_frames(self):
        if not self.inference_state or not self.object_prompts:
            return
        
        self.segment_current_frame()
        
        # Then propagate to all frames
        self.status_label.setText("Propagating segmentation to all frames...")
        QApplication.processEvents()  # Update UI
        
        self.video_segments = {}  # Reset segments
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            # Update UI occasionally to show progress
            if out_frame_idx % 10 == 0:
                self.status_label.setText(f"Processed frame {out_frame_idx}/{len(self.frame_names)}")
                QApplication.processEvents()
        
        self.status_label.setText(f"Segmentation completed for all {len(self.frame_names)} frames")
        self.display_current_frame()
            
        self.update_ui_state()
    
    def save_masks(self):
        if not self.video_segments:
            return
        
        try:
            # Get save path
            subfolder = self.save_path_input.text().strip()
            if not subfolder:
                subfolder = "mask"
            
            # Create save directory in parent of data_dir
            parent_dir = os.path.dirname(os.path.normpath(self.data_dir))
            save_dir = os.path.join(self.data_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)
            vis_dir = save_dir.replace(subfolder, "masks_vis")
            os.makedirs(vis_dir, exist_ok=True)
            pallete = np.concatenate([
                np.array([[0, 0, 0]]),
                np.random.randint(0, 256, (100, 3), dtype=np.uint8)
            ], axis=0)
            
            # Save masks as numpy arrays
            saved_count = 0
            for frame_idx, obj_masks in self.video_segments.items():
                frame_name = os.path.splitext(self.frame_names[frame_idx])[0]
                masks = []
                for obj_id, mask in obj_masks.items():
                    masks.append(mask)
                mask_img = np.zeros_like(masks[0][0]).astype(np.uint8)
                print(mask_img.shape)
                print(masks[0][0].max(), masks[0][0].min(), masks[0][0].sum())
                # breakpoint()
                mask_img[np.where(masks[0][0] == True)] = 255
                print(mask_img.dtype)
                mask_img = ((mask_img - mask_img.min()) / (mask_img.max() - mask_img.min()) * 255).astype(np.uint8)
                print(mask_img.max())
                save_path = os.path.join(save_dir, f"{frame_name}.png")
                Image.fromarray(mask_img).save(save_path)
                print(save_path)
                masks = np.stack(masks, axis=0) # (num_masks, H, W)
                # save_path = os.path.join(save_dir, f"{frame_name}.npy")
                # np.save(save_path, masks)

                # visualize the masks
                masks = np.concatenate([np.zeros((1, *masks.shape[1:])), masks], axis=0)
                masks = np.argmax(masks, axis=0).squeeze()
                vis_mask = pallete[masks]
                vis_path = os.path.join(vis_dir, f"{frame_name}.png")
                Image.fromarray(vis_mask.astype(np.uint8)).save(vis_path)

                saved_count += 1
                
                # Periodically update UI to show progress
                if saved_count % 50 == 0:
                    self.status_label.setText(f"Saved {saved_count} masks...")
                    QApplication.processEvents()
            
            self.status_label.setText(f"Saved {saved_count} masks to {save_dir}")
            
        except Exception as e:
            self.status_label.setText(f"Error saving masks: {str(e)}")
    
    # Helper functions to display masks, points, and boxes
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id % 10  # Ensure we don't exceed colormap range
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    def show_points(self, coords, labels, ax, marker_size=200):
        if len(coords) == 0:
            return
        
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        
        if len(pos_points) > 0:
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                       s=marker_size, edgecolor='white', linewidth=1.25)
        
        if len(neg_points) > 0:
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                       s=marker_size, edgecolor='white', linewidth=1.25)
    
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
    def closeEvent(self, event):
        # Stop animation timer when closing the application
        if self.animation_timer.isActive():
            self.animation_timer.stop()
        # Clean up resources
        self.frame_cache.clear()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = SAM2AnnotationTool()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()