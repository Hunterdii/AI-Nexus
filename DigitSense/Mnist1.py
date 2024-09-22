
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
import requests
import streamlit_lottie as st_lottie
import json

st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🧮", layout="wide")

# # Load Lottie animation
# def load_lottie_url(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# # Lottie Animation
# lottie_url = "https://lottie.host/d037f966-769e-40c3-88c4-7c418130cd05/ipnsCTF8x2.json"
# lottie_animation = load_lottie_url(lottie_url)

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Path to the Lottie file
lottie_filepath = "mnist_classification/lottie.json"  # Make sure the path is correct

# Load Lottie animation
lottie_animation = load_lottie_file(lottie_filepath)

# CSS for centering
st.markdown(
    """
    <style>
    .center-lottie {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, 
    unsafe_allow_html=True
)


with st.sidebar:
# Center Lottie animation
    st.markdown('<div class="center-lottie">', unsafe_allow_html=True)
    st_lottie.st_lottie(lottie_animation, height=200, width=300, key="lottie_animation")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='color: #007bff;' align='center'>Explore the App!</h2>", unsafe_allow_html=True)
    st.markdown("**About the Model:** This MNIST digit recognizer uses a convolutional neural network trained on handwritten digits.")

    st.markdown(""" 
        <style>
            .feature-hover {
                position: relative;
                display: inline-block;
                color: #007bff;
                cursor: pointer;
            }

            .feature-hover .tooltip-text {
                visibility: hidden;
                width: 200px;
                background-color: #333;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 100%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }

            .feature-hover:hover .tooltip-text {
                visibility: visible;
                opacity: 1;
            }
        </style>

        <ul>
            <li>
                <div class="feature-hover">Accurate Predictions
                    <span class="tooltip-text">Our model accurately identifies digits with high precision.</span>
                </div>
            </li>
            <li>
                <div class="feature-hover">Interactive Drawing
                    <span class="tooltip-text">Draw digits directly on the canvas for recognition.</span>
                </div>
            </li>
        </ul>
    """, unsafe_allow_html=True)

    # Contact information
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Samarth**](https://www.linkedin.com/in/samarth-garge-77b352248?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) | [**Hunterdii**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# Load model
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("mnist_classification/model.h5")
    return model

model = load_my_model()

st.markdown("""
    <style>
    .centered-box {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(""" 
    <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
    🧮 MNIST Digit Recognition
    </h1>
    <style>
    @keyframes glow {
        0% {
            text-shadow: 0 0 10px #9b59b6, 0 0 20px #007bff, 0 0 30px #007bff, 0 0 40px #9b59b6;
        }
        100% {
            text-shadow: 0 0 20px #8e44ad, 0 0 30px #007bff, 0 0 40px #007bff, 0 0 50px #8e44ad;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .centered-box {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
""", unsafe_allow_html=True)

st.header("Upload an image or draw a digit to get predictions!")

upload_or_draw = st.selectbox("Choose Input Method:", ["Upload an Image", "Draw on Canvas"])

img_to_predict = None 

if upload_or_draw == "Upload an Image":
    st.markdown('<div class="centered-box">', unsafe_allow_html=True)
    image_file = st.file_uploader("Upload an image (jpg, png, jpeg, etc.)", type=["jpg", "jpeg", "png"], key="file_uploader")
    st.markdown('</div>', unsafe_allow_html=True)  # Close centered box

    if image_file is not None:
        image = Image.open(image_file).convert("L")  # grayscale
        img = np.array(image)
        img = cv2.resize(img, (28, 28))  # Resize
        img = np.expand_dims(img, axis=-1)  
        img = img / 255.0 
        img_to_predict = img.reshape(1, 28, 28, 1)  
        
        st.markdown('<div class="centered-box">', unsafe_allow_html=True)
        st.image(img.reshape(28, 28), width=120)  
        st.markdown('</div>', unsafe_allow_html=True)

elif upload_or_draw == "Draw on Canvas":
    st.subheader("Draw a digit:")
    SIZE = 256  # Canvas size for easier drawing
    mode = st.checkbox("Draw (or Delete)?", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=8,  # Adjust pen width
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw" if mode else "transform",
        key='canvas'
    )

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
        img = np.expand_dims(img, axis=-1) 
        img = img / 255.0  
        img_to_predict = img.reshape(1, 28, 28, 1)  
        
        st.markdown('<div class="centered-box">', unsafe_allow_html=True)
        st.image(img.reshape(28, 28), width=120)  
        st.markdown('</div>', unsafe_allow_html=True)

# Classification button
if st.button("Classify Image"):
    if img_to_predict is not None:
        with st.spinner('🔍 Classifying image...'):
            try:
                predictions = model.predict(img_to_predict)
                predicted_class = np.argmax(predictions, axis=-1)
                confidence = np.max(predictions)

                confidence_threshold = 0.60
                if confidence < confidence_threshold:
                    result = f"Prediction: Not confident (Confidence: {confidence*100:.2f}%)"
                else:
                    result = f"Prediction: Digit {predicted_class[0]} with {confidence*100:.2f}% confidence"

                st.success(result)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.error("No image or drawing to classify.")

st.markdown(""" 
### **MNIST Digits:**
- **0:** Zero
- **1:** One
- **2:** Two
- **3:** Three
- **4:** Four
- **5:** Five
- **6:** Six
- **7:** Seven
- **8:** Eight
- **9:** Nine
""")

# Performance Data 
st.markdown(""" 
### **Model Performance:**
- **Accuracy:** 89.5%
- **Precision:** 95.7%
""")


# import numpy as np
# import tensorflow as tf
# import streamlit as st
# from PIL import Image
# from streamlit_drawable_canvas import st_canvas
# import cv2

# # Streamlit page configuration
# st.set_page_config(page_title="MNIST Digit Recognition", page_icon="🧮", layout="wide")

# # Sidebar with unique elements
# with st.sidebar:
#     st.markdown("<h2 style='color: #007bff;'>Explore the App!</h2>", unsafe_allow_html=True)
#     st.markdown("**About the Model:** This MNIST digit recognizer uses a convolutional neural network trained on handwritten digits.")

#     # Features section with hover effect
#     st.markdown(""" 
#         <style>
#             .feature-hover {
#                 position: relative;
#                 display: inline-block;
#                 color: #007bff;
#                 cursor: pointer;
#             }

#             .feature-hover .tooltip-text {
#                 visibility: hidden;
#                 width: 200px;
#                 background-color: #333;
#                 color: #fff;
#                 text-align: center;
#                 border-radius: 6px;
#                 padding: 5px;
#                 position: absolute;
#                 z-index: 1;
#                 bottom: 100%;
#                 left: 50%;
#                 margin-left: -100px;
#                 opacity: 0;
#                 transition: opacity 0.3s;
#             }

#             .feature-hover:hover .tooltip-text {
#                 visibility: visible;
#                 opacity: 1;
#             }
#         </style>

#         <ul>
#             <li>
#                 <div class="feature-hover">Accurate Predictions
#                     <span class="tooltip-text">Our model accurately identifies digits with high precision.</span>
#                 </div>
#             </li>
#             <li>
#                 <div class="feature-hover">Interactive Drawing
#                     <span class="tooltip-text">Draw digits directly on the canvas for recognition.</span>
#                 </div>
#             </li>
#         </ul>
#     """, unsafe_allow_html=True)

#     # Contact information
#     st.markdown("<hr>", unsafe_allow_html=True)
#     st.markdown('Contact us at: [**Hunterdii**](https://www.linkedin.com/in/het-patel-8b110525a/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')

# # Load model
# @st.cache_resource
# def load_my_model():
#     model = tf.keras.models.load_model("model.h5")
#     print("Model loaded successfully.")
#     return model

# model = load_my_model()

# # Main title with cool text effect
# st.markdown(""" 
#     <h1 style="text-align:center; color: #007bff; font-family: 'Courier New', Courier, monospace; animation: glow 2s ease-in-out infinite alternate;">
#     🧮 MNIST Digit Recognition
#     </h1>
#     <style>
#     @keyframes glow {
#         0% {
#             text-shadow: 0 0 10px #9b59b6, 0 0 20px #007bff, 0 0 30px #007bff, 0 0 40px #9b59b6;
#         }
#         100% {
#             text-shadow: 0 0 20px #8e44ad, 0 0 30px #007bff, 0 0 40px #007bff, 0 0 50px #8e44ad;
#         }
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.header("Upload an image or draw a digit to get predictions!")

# # Toggle for upload or canvas
# upload_or_draw = st.selectbox("Choose Input Method:", ["Upload an Image", "Draw on Canvas"])

# # Image upload or drawing section
# img_to_predict = None

# # Layout to center the canvas and other elements
# col1, col2, col3 = st.columns([1, 2, 1])

# with col2:  # This will center the content between columns
#     if upload_or_draw == "Upload an Image":
#         # Image upload section
#         image_file = st.file_uploader("Upload an image (28x28 PNG)", type=["png"], key="file_uploader")

#         if image_file is not None:
#             st.write("You have uploaded an image.")
#             # Resize the uploaded image to 28x28 and preprocess
#             image = Image.open(image_file).convert("L")  # Convert to grayscale
#             img = np.array(image)
#             img = cv2.resize(img, (28, 28))  # Resize to 28x28
#             img = np.expand_dims(img, axis=-1)  # Add channel dimension
#             img = img / 255.0  # Normalize pixel values
#             img_to_predict = img.reshape(1, 28, 28, 1)  # Shape it for the model
            
#             # Display the smaller version of the image (processed)
#             st.write(f"Processed Image Shape: {img_to_predict.shape}")
#             st.image(img.reshape(28, 28), width=120)  # Set width to make it smaller

#     elif upload_or_draw == "Draw on Canvas":
#         # Canvas for drawing digits
#         st.subheader("Draw a digit:")
#         SIZE = 160  # Medium canvas size
#         mode = st.checkbox("Draw (or Delete)?", True)
#         canvas_result = st_canvas(
#             fill_color='#000000',
#             stroke_width=10,  # Adjust pen width
#             stroke_color='#FFFFFF',
#             background_color='#000000',
#             width=SIZE,
#             height=SIZE,
#             drawing_mode="freedraw" if mode else "transform",
#             key='canvas'
#         )

#         if canvas_result.image_data is not None:
#             st.write("You are drawing on the canvas.")
#             # Resize the canvas image to 28x28 and preprocess it
#             img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (if needed)
#             img = np.expand_dims(img, axis=-1)  # Add channel dimension
#             img = img / 255.0  # Normalize pixel values to range [0, 1]
#             img_to_predict = img.reshape(1, 28, 28, 1)  # Ensure correct shape for the model (batch size, 28x28, 1 channel)
            
#             # Display the smaller version of the image (processed)
#             st.write(f"Processed Image Shape: {img_to_predict.shape}")
#             st.image(img.reshape(28, 28), width=120)  # Set width to make it smaller

# # Classification button
# if st.button("Classify Image"):
#     if img_to_predict is not None:
#         with st.spinner('🔍 Classifying image...'):
#             try:
#                 predictions = model.predict(img_to_predict)
#                 predicted_class = np.argmax(predictions, axis=-1)
#                 confidence = np.max(predictions)

#                 # Threshold and result display
#                 confidence_threshold = 0.60
#                 if confidence < confidence_threshold:
#                     result = f"Prediction: Not confident (Confidence: {confidence*100:.2f}%)"
#                 else:
#                     result = f"Prediction: Digit {predicted_class[0]} with {confidence*100:.2f}% confidence"

#                 st.success(result)
#             except Exception as e:
#                 st.error(f"Error during prediction: {e}")

# # Additional MNIST Information
# st.markdown(""" 
# ### **MNIST Digits:**
# - **0:** Zero
# - **1:** One
# - **2:** Two
# - **3:** Three
# - **4:** Four
# - **5:** Five
# - **6:** Six
# - **7:** Seven
# - **8:** Eight
# - **9:** Nine
# """)

# # Performance Data (if applicable)
# st.markdown(""" 
# ### **Model Performance:**
# - **Accuracy:** 98.5%
# - **Precision:** 98.7%
# """)
