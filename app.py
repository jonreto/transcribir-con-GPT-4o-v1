import streamlit as st
import openai
import os
import math
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

# --- Transcription Logic ---
def transcribe_audio(client, file_path, model, prompt):
    """
    Splits audio file into chunks and transcribes them using OpenAI API.
    Uses st.status to show progress.
    """
    status_ui = st.status(f"Processing file: '{os.path.basename(file_path)}'...", expanded=True)
    
    final_transcript = ""
    temp_chunk_files = []
    
    try:
        # --- Configuration ---
        RESPONSE_FORMAT = "text"
        TARGET_CHUNK_SIZE_MB = 2.9
        CHUNK_EXPORT_FORMAT = "mp3"
        FALLBACK_CHUNK_DURATION_S = 60
        BYTES_PER_MB = 1024 * 1024
        TARGET_CHUNK_SIZE_BYTES = TARGET_CHUNK_SIZE_MB * BYTES_PER_MB

        # 1. Load Audio
        status_ui.write("Loading audio file with pydub...")
        audio = AudioSegment.from_file(file_path)
        duration_s = len(audio) / 1000.0
        status_ui.write(f"Audio loaded successfully. Duration: {duration_s:.2f} seconds.")

        # 2. Estimate Bitrate and Calculate Chunk Duration
        file_size_bytes = os.path.getsize(file_path)
        if duration_s > 0:
             estimated_bitrate_bps = (file_size_bytes * 8) / duration_s
             ideal_chunk_duration_s = (TARGET_CHUNK_SIZE_BYTES * 8) / estimated_bitrate_bps
             chunk_duration_s = max(1.0, ideal_chunk_duration_s * 0.95)
             status_ui.write(f"Estimated bitrate: {estimated_bitrate_bps/1000:.1f} kbps. Target chunk duration: ~{chunk_duration_s:.1f}s for < {TARGET_CHUNK_SIZE_MB}MB chunks.")
        else:
             status_ui.write(f"Warning: Audio duration is zero. Using fallback duration: {FALLBACK_CHUNK_DURATION_S}s.")
             chunk_duration_s = FALLBACK_CHUNK_DURATION_S

        chunk_duration_ms = int(chunk_duration_s * 1000)

        # 3. Calculate number of chunks
        num_chunks = math.ceil(len(audio) / chunk_duration_ms)
        if num_chunks == 0 and len(audio) > 0:
             num_chunks = 1
        status_ui.write(f"Splitting into {num_chunks} chunks.")

        progress_bar = st.progress(0)

        # 4. Split, Export, and Transcribe Chunks
        for i in range(num_chunks):
            start_ms = i * chunk_duration_ms
            end_ms = min((i + 1) * chunk_duration_ms, len(audio))
            if start_ms >= end_ms:
                continue

            chunk = audio[start_ms:end_ms]
            chunk_filename = f"temp_chunk_{i}.{CHUNK_EXPORT_FORMAT}"
            temp_chunk_files.append(chunk_filename)

            status_ui.update(label=f"Processing chunk {i+1}/{num_chunks} ({start_ms/1000:.2f}s to {end_ms/1000:.2f}s)...")

            try:
                # Export chunk
                status_ui.write(f"  Exporting chunk to '{chunk_filename}'...")
                chunk.export(chunk_filename, format=CHUNK_EXPORT_FORMAT)
                chunk_size_bytes = os.path.getsize(chunk_filename)

                if chunk_size_bytes > (25 * BYTES_PER_MB):
                    st.error(f"FATAL ERROR: Chunk size {chunk_size_bytes / BYTES_PER_MB:.2f} MB exceeds the 25MB API limit! Stopping.")
                    break

                if chunk_size_bytes == 0:
                    status_ui.write("  Warning: Exported chunk is empty. Skipping.")
                    continue

                # Transcribe chunk
                status_ui.write(f"  Sending chunk to OpenAI API ({model})...")
                with open(chunk_filename, "rb") as chunk_audio_file:
                    transcription_response = client.audio.transcriptions.create(
                        model=model,
                        file=chunk_audio_file,
                        response_format=RESPONSE_FORMAT,
                        prompt=prompt if prompt else None
                    )
                
                chunk_transcript = transcription_response
                final_transcript += chunk_transcript + " "
                status_ui.write(f"  Transcription received for chunk {i+1}.")

            except openai.APIError as e:
                st.error(f"OpenAI API error for chunk {i+1}: {e}")
                status_ui.write(f"  Skipping chunk {i+1} due to API error.")
            except Exception as e:
                st.error(f"An unexpected error occurred processing chunk {i+1}: {e}")
                status_ui.write(f"  Skipping chunk {i+1} due to an unexpected error.")

            progress_bar.progress((i + 1) / num_chunks)

        status_ui.update(label="Transcription complete!", state="complete", expanded=False)

    except CouldntDecodeError:
        st.error(f"ERROR: pydub failed to load or decode the audio file: '{os.path.basename(file_path)}'. Please ensure it's a valid, uncorrupted audio file.")
        status_ui.update(label="Error processing audio.", state="error")
        return ""
    except Exception as e:
        st.error(f"An error occurred during the main transcription process: {e}")
        status_ui.update(label="An unexpected error occurred.", state="error")
        return ""
    finally:
        # 6. Clean up temporary files
        cleaned_count = 0
        for temp_file in temp_chunk_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    cleaned_count += 1
            except OSError as e:
                st.warning(f"Could not remove temporary file '{temp_file}': {e}")
        if cleaned_count > 0:
            st.info(f"Cleaned up {cleaned_count} temporary chunk files.")

    return final_transcript.strip()


# --- Streamlit UI ---
st.set_page_config(page_title="Audio Transcription", page_icon="🎙️", layout="wide")
st.title("Audio Transcription with OpenAI")

st.markdown("""
This app transcribes audio files using OpenAI's transcription models.
It handles large files by automatically splitting them into smaller chunks.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    st.subheader("1. Upload Audio File")
    uploaded_file = st.file_uploader(
        "Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm",
        type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm']
    )

    st.subheader("2. Transcription Settings")
    model_id = st.selectbox(
        "Model",
        ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"],
        index=0,
        help="The model names from the original script are included. If they don't work, `whisper-1` is the standard model for transcription."
    )
    prompt = st.text_area("Prompt (optional)", help="A prompt to guide the model's style or to provide context.")


# --- Main App Logic ---
if uploaded_file is None:
    st.info("Please upload an audio file to get started.")
else:
    st.audio(uploaded_file, format=uploaded_file.type)
    
    if st.button("Transcribe Audio", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar to proceed.")
        else:
            temp_file_path = None
            try:
                client = openai.OpenAI(api_key=api_key)
                
                # Save uploaded file to a temporary path
                temp_dir = "temp_uploads"
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                final_transcript = transcribe_audio(client, temp_file_path, model_id, prompt)

                if final_transcript:
                    st.subheader("Final Transcription")
                    st.text_area("", final_transcript, height=400)
                    st.download_button(
                        label="Download Transcript",
                        data=final_transcript,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_transcript.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up the uploaded file
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    st.info(f"Cleaned up temporary uploaded file: {os.path.basename(temp_file_path)}") 