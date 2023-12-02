#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import numpy as np

def calculate_color_histogram(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_color_distributions(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def calculate_video_difference(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    hist1 = None
    hist2 = None
    frame_count = 0
    total_correlation = 0.0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        if hist1 is None:
            hist1 = calculate_color_histogram(frame1)
        
        hist2 = calculate_color_histogram(frame2)
        
        correlation = compare_color_distributions(hist1, hist2)
        total_correlation += correlation
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    if frame_count > 0:
        average_correlation = total_correlation / frame_count
        # Map correlation to a percentage scale (0% to 100%)
        difference_percentage = (1 - average_correlation) * 100
        return difference_percentage
    else:
        return None

def main():
    video1_path = 'demo_1.mp4'
    video2_path = 'car_far.mp4'
    
    difference_percentage = calculate_video_difference(video1_path, video2_path)
    
    if difference_percentage is not None:
        print(f"Color distribution difference between videos: {difference_percentage:.2f}%")
    else:
        print("Error: Videos have no frames or cannot be read.")

if __name__ == "__main__":
    main()


# In[11]:


import cv2
import numpy as np

def calculate_color_histogram(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_color_distributions(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def calculate_video_difference(video_path1, video_path2):
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    hist1 = None
    hist2 = None
    frame_count = 0
    total_correlation = 0.0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        if hist1 is None:
            hist1 = calculate_color_histogram(frame1)
        
        hist2 = calculate_color_histogram(frame2)
        
        correlation = compare_color_distributions(hist1, hist2)
        total_correlation += correlation
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    if frame_count > 0:
        average_correlation = total_correlation / frame_count
        # Map correlation to a percentage scale (0% to 100%)
        difference_percentage = (1 - average_correlation) * 100
        return difference_percentage
    else:
        return None

def main():
    video1_path = 'demo_1.mp4'
    video2_path = 'car_camera.mp4'
    
    difference_percentage = calculate_video_difference(video1_path, video2_path)
    
    if difference_percentage is not None:
        print(f"Color distribution difference between videos: {difference_percentage:.2f}")
    else:
        print("Error: Videos have no frames or cannot be read.")

if __name__ == "__main__":
    main()


# In[14]:


import cv2 # comparing watermarked video and pirated video

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return width, height

def get_video_bitrate(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get video codec and frames per second
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get bitrate in bps (bits per second)
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))
    
    cap.release()
    
    return codec, fps, bitrate

def main():
    video1_path = 'demo_1.mp4'
    video2_path = 'car_crop.mp4'
    
    width_video1, height_video1 = get_video_resolution(video1_path)
    width_video2, height_video2 = get_video_resolution(video2_path)
    
    codec_video1, fps_video1, bitrate_video1 = get_video_bitrate(video1_path)
    codec_video2, fps_video2, bitrate_video2 = get_video_bitrate(video2_path)
    
    print(f"Resolution of Video 1: {width_video1}x{height_video1}")
    print(f"Resolution of Video 2: {width_video2}x{height_video2}")
    
    print(f"Bitrate of Video 1: {bitrate_video1} bps")
    print(f"Bitrate of Video 2: {bitrate_video2} bps")
    
    if width_video1 > width_video2 or height_video1 > height_video2:
        print("Video 1 has higher resolution.")
    elif width_video1 < width_video2 or height_video1 < height_video2:
        print("Video 2 has higher resolution.")
    else:
        print("Both videos have the same resolution.")
    
    if bitrate_video1 > bitrate_video2:
        print("Video 1 has higher bitrate.")
    elif bitrate_video1 < bitrate_video2:
        print("Video 2 has higher bitrate.")
    else:
        print("Both videos have the same bitrate.")

if __name__ == "__main__":
    main()


# In[ ]:




