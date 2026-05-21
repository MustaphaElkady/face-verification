from pathlib import Path
import sys

import streamlit as st

# Add /src to the Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.FaceVerificationService import FaceVerificationService
from helpers.image import bytes_to_cv2_image
from src.core import settings


st.set_page_config(
    page_title="Face Verification App",
    page_icon="📸",
    layout="centered"
)

st.title("📸 Face Verification Tester")
st.write("Upload an ID card and a selfie to test the similarity score.")


@st.cache_resource
def load_service():
    return FaceVerificationService()


service = load_service()

col1, col2 = st.columns(2)

with col1:
    id_file = st.file_uploader(
        "Upload ID Image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if id_file is not None:
        st.image(id_file, caption="ID Image", width="stretch")

with col2:
    selfie_file = st.file_uploader(
        "Upload Selfie Image",
        type=["jpg", "jpeg", "png", "webp"]
    )
    if selfie_file is not None:
        st.image(selfie_file, caption="Selfie Image", width="stretch")

if st.button("Verify Faces"):
    if id_file is None or selfie_file is None:
        st.warning("Please upload both the ID image and the selfie.")
    else:
        with st.spinner("Analyzing faces..."):
            try:
                id_image = bytes_to_cv2_image(id_file.getvalue())
                selfie_image = bytes_to_cv2_image(selfie_file.getvalue())

                if id_image is None:
                    st.error("Could not decode the ID image.")
                elif selfie_image is None:
                    st.error("Could not decode the selfie image.")
                else:
                    result = service.verify_faces(id_image, selfie_image)

                    st.markdown("---")
                    st.subheader("Results")

                    similarity = result["similarity"] if isinstance(result, dict) else result.similarity
                    verified = result["verified"] if isinstance(result, dict) else result.verified

                    if verified:
                        st.success(f"✅ Faces Match! Similarity: {similarity}")
                    else:
                        st.error(f"❌ Faces Do Not Match. Similarity: {similarity}")

                    st.info(f"Current threshold: {settings.similarity_threshold}")

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")