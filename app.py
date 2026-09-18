import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.title("🎯 Real-Time Object Detection App")
st.write("Image ya video upload karo, YOLOv8 usme objects detect karega!")

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Sidebar mein settings
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

option = st.radio("Kya upload karna chahte ho?", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Image choose karo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        if st.button("Detect karo"):
            with st.spinner("Detecting..."):
                results = model(np.array(image), conf=confidence)
                annotated_image = results[0].plot()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                st.image(annotated_image, caption="Detected Objects", use_container_width=True)

                # Detected objects ki list dikhao
                detected_classes = results[0].boxes.cls.tolist()
                if detected_classes:
                    class_names = [model.names[int(c)] for c in detected_classes]
                    from collections import Counter
                    counts = Counter(class_names)
                    st.subheader("Detected Objects:")
                    for obj, count in counts.items():
                        st.write(f"- **{obj}**: {count}")
                else:
                    st.write("Koi object detect nahi hua.")

elif option == "Video":
    uploaded_video = st.file_uploader("Video choose karo", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        st.video(uploaded_video)

        if st.button("Detect karo"):
            with st.spinner("Processing video... (thoda time lagega)"):
                results = model(tfile.name, save=True, conf=confidence)
                st.success("Detection complete! Video process ho gayi.")
                st.write("Note: Colab mein output video dikhane ke liye extra steps chahiye - GitHub/local pe better chalega.")
