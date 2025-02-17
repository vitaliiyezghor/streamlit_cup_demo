import streamlit as st
from streamlit_webrtc import webrtc_streamer
from streamlit_webrtc.config import VideoHTMLAttributes
import av


flip = st.checkbox("Flip")


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:] if flip else img

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", 
                video_frame_callback=video_frame_callback, 
                video_html_attrs=VideoHTMLAttributes(
                    autoPlay=True, 
                    controls=True, 
                    style={"width": "100%"},
                    playsInline=True
                )
)