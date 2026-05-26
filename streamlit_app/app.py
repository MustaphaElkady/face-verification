from pathlib import Path
import sys

# Add the project root and src/ to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from services.FaceVerificationService import FaceVerificationService
from helpers.image import bytes_to_cv2_image
from core.config import settings
from core.exceptions import (
    ImageValidationError,
    QualityCheckError,
    NoFaceDetectedError,
    DuplicateImageError,
)

st.set_page_config(
    page_title="Face Verification v2",
    page_icon="🪪",
    layout="centered",
)

st.title("🪪 Face Verification — v2")

st.caption(
    f"Backend: **{settings.model_backend}/{settings.model_name}** | "
    f"Device requested: **{'GPU' if settings.use_gpu else 'CPU'}** | "
    f"Threshold: **{settings.similarity_threshold}** | "
    f"Gray zone: **{settings.manual_review_threshold} – {settings.similarity_threshold}**"
)


@st.cache_resource(show_spinner="Loading face model…")
def load_service(use_gpu: bool, backend: str, model_name: str):
    """
    Cache key includes use_gpu/backend/model_name.
    So when you restart Streamlit with USE_GPU=true/false,
    it loads the correct device mode.
    """
    return FaceVerificationService()


col1, col2 = st.columns(2)

with col1:
    id_file = st.file_uploader(
        "Upload ID Photo",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if id_file:
        st.image(id_file, caption="ID Image", use_container_width=True)

with col2:
    selfie_file = st.file_uploader(
        "Upload Selfie",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if selfie_file:
        st.image(selfie_file, caption="Selfie", use_container_width=True)


if st.button("🔍 Verify Faces", use_container_width=True, type="primary"):
    if not id_file or not selfie_file:
        st.warning("Please upload both the ID photo and the selfie.")
    else:
        with st.spinner("Analysing…"):
            try:
                service = load_service(
                    settings.use_gpu,
                    settings.model_backend,
                    settings.model_name,
                )

                id_img = bytes_to_cv2_image(
                    id_file.getvalue(),
                    label="ID image",
                )

                selfie_img = bytes_to_cv2_image(
                    selfie_file.getvalue(),
                    label="selfie",
                )

                result = service.verify_faces(id_img, selfie_img)

                st.divider()
                st.subheader("Result")

                verdict = result["verdict"]
                similarity = result["similarity"]

                if verdict == "VERIFIED":
                    st.success(f"✅ **VERIFIED** — Similarity: `{similarity}`")
                elif verdict == "MANUAL_REVIEW":
                    st.warning(
                        f"⚠️ **MANUAL REVIEW REQUIRED** — Similarity: `{similarity}`\n\n"
                        "Score is in the gray zone. Route to a human reviewer."
                    )
                else:
                    st.error(f"❌ **REJECTED** — Similarity: `{similarity}`")

                st.progress(
                    min(max(float(similarity), 0.0), 1.0),
                    text=f"Similarity score: {similarity}",
                )

                for warning in result.get("warnings", []):
                    st.info(f"ℹ️ {warning}")

                with st.expander("Details"):
                    cols = st.columns(2)

                    cols[0].metric("ID det_score", result["id_det_score"])
                    cols[1].metric("Selfie det_score", result["selfie_det_score"])

                    cols[0].metric("Threshold", result["threshold"])
                    cols[1].metric(
                        "Manual review threshold",
                        result["manual_review_threshold"],
                    )

                    st.caption(f"Model: `{result['model']}`")

                with st.expander("Quality metrics"):
                    st.json(result.get("quality_metrics", {}))

            except (ImageValidationError, NoFaceDetectedError) as e:
                st.error(f"📷 {e}")

            except QualityCheckError as e:
                st.error(f"🔍 Quality check failed: {e}")
                if e.metrics:
                    st.json(e.metrics)

            except DuplicateImageError as e:
                st.error(f"⚠️ {e}")

            except Exception as e:
                st.error(f"Unexpected error: {e}")