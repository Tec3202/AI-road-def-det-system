import streamlit as st
import module1
import streamlit.components.v1 as components 
import cv2 
import numpy as np 
from ultralytics import YOLO
import streamlit_option_menu as option_menu
from PIL import Image, ImageDraw
import io
import tempfile
import imageio.v2 as imageio
from moviepy.editor import ImageSequenceClip
import os
import shutil
# from ultralytics.yolo.utils.plotting import Annotator
from cv2 import cvtColor
import os
from helper import *

# page configuration
st.set_page_config(page_icon=':motorway:',page_title='road inspection system')



#Importing the model
model = YOLO('best.pt')
def bgr2rgb(image):
    return image[:, :, ::-1]




# video processing function
def process_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Set a default value for fps if it is 0 or None

    # Create a list to store the processed frames
    processed_frames = []

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform the prediction on the frame
        prediction = model.predict(frame)
        frame_with_bbox = prediction[0].plot()

        # Convert the frame to PIL Image and store in the list
        processed_frames.append(Image.fromarray(frame_with_bbox))

    cap.release()

    # Create the output video file path
    video_path_output = "output.mp4"

    # Save the processed frames as individual images
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(processed_frames):
            frame.save(f"{temp_dir}/frame_{i}.png")

        # Create a video clip from the processed frames
        video_clip_path = 'clip.mp4' #video_clip_path = f"{temp_dir}/clip.mp4"
        os.system(f"ffmpeg -framerate {fps} -i {temp_dir}/frame_%d.png -c:v libx264 -pix_fmt yuv420p {video_clip_path}")

        # Rename the video clip with the desired output path
        shutil.copy2(video_clip_path, video_path_output)

    if os.path.exists(video_path_output):
       return video_path_output
    else:
        print(f"{video_path_output} is not there")




# Main function       
def main():

    with open("styles.css", "r") as source_style:
        st.markdown(f"<style>{source_style.read()}</style>", 
             unsafe_allow_html = True)
    
    # default title goes here
        
    Header = st.container()
    js_code = """
        const elements = window.parent.document.getElementsByTagName('footer');
        elements[0].innerHTML = "Mexico x Omdena VIT Bhopal Local Chapter " + new Date().getFullYear();
        """
    st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)
            
    
    ##MainMenu
    
    with st.sidebar:
        selected = option_menu.option_menu(
            "Main Menu",
            options=[           
                "Predict Defects"
            ],
        )
    
    st.sidebar.markdown('---')
        
    ##HOME page  
    if selected == "Predict Defects": 
        
        st.sidebar.subheader('Settings')
        
        options = st.sidebar.radio(
            'Options:', ('Image', 'Video'), index=0)
        
        st.sidebar.markdown("---")
         # Image
        if options == 'Image':
            # re-define page title
            p_title = "Road Defect Detection System[Image]" 
            st.subheader(p_title) 
            upload_img_file = st.sidebar.file_uploader(
                'Upload Image', type=['jpg', 'jpeg', 'png'])
            if upload_img_file is not None:
                # call run button
                p_button = st.sidebar.button('Run')
                if p_button != True:
                    st.subheader('original image')
                    st.image(module1.display_image(upload_img_file),caption='original image')
                    st.info('Click run button to start the detection process')
                else:
                    with st.spinner('Defect detection in progress kindly wait...'):
                        file_bytes = np.asarray(bytearray(upload_img_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, 1)
    
                        prediction = model.predict(img)
                        res_plotted = prediction[0].plot()
                        image_pil = Image.fromarray(res_plotted)
                        image_bytes = io.BytesIO()
                        image_pil.save(image_bytes, format='PNG')

                        st.image(image_bytes, caption='Predicted Image', use_column_width=True)
                        st.success('done!')
            
                
        if options == 'Video':
            # re-define page title
            p_title = "Road Defect Detection System[Video]"
            st.subheader(p_title) 
            upload_vid_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv']
                )
            if upload_vid_file is not None:
            # Save the uploaded video file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(upload_vid_file.read())
                vid = open(temp_file.name,'rb')
                vid_bytes = vid.read()

                st.sidebar.text('Input Video')
                st.sidebar.video(vid_bytes)
                

                # Process the video frames and get the output video file path
                with st.spinner('video frames processing in progress kindly wait...'):
                     video_path_output = process_video(temp_file.name)

                # Display the processed video using the st.video function
                if os.path.exists(video_path_output):
                    print(video_path_output," is there")
                
                st.video(video_path_output,format='video/mp4')


                # Remove the temporary files
                temp_file.close()
                os.remove(video_path_output)

         if options == 'YouTube Video':
                 conf = float(st.sidebar.slider(
                     "Select Model Confidence", 25, 100, 40)) / 100
                 play_youtube_video(conf, model)
        

         if options == 'Stored Video':
                 conf = float(st.sidebar.slider(
                     "Select Model Confidence", 25, 100, 40)) / 100
                 play_stored_video(conf, model)

                
            


             
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    
