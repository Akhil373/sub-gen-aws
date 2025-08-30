from faster_whisper import WhisperModel, BatchedInferencePipeline
import yt_dlp
import subprocess
from typing import Optional
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
import os
import time
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import zipfile
import tempfile


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)


def youtube_download_video(VIDEO_URL, DOWNLOAD_DIR, output_template):
    URLS = [VIDEO_URL]
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[height<=1080]+bestaudio/best",
        "merge_output_format": "mp4",
        "verbose": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading from YouTube: {URLS[0]}")
            info = ydl.extract_info(URLS[0], download=True)
            if not info:
                return "Error downloading youtube video"

            final_filepath = None
            if "requested_downloads" in info and info["requested_downloads"]:
                final_filepath = info["requested_downloads"][0]["filepath"]
            elif "_filename" in info:
                final_filepath = info["_filename"]
            else:
                print(
                    "Warning: yt-dlp did not provide a clear filepath. Attempting to construct."
                )
                if "title" in info and "ext" in info:
                    guessed_filename = f"{info['title']}.{info['ext']}"
                    guessed_path = os.path.join(DOWNLOAD_DIR, guessed_filename)
                    if os.path.exists(guessed_path):
                        final_filepath = guessed_path
                    else:
                        print(
                            f"Could not determine downloaded file path for {URLS[0]}."
                        )

        except Exception as e:
            print(f"An error occurred during YouTube download: {e}")
            final_filepath = None
        finally:
            return final_filepath


def local_audio_file(DOWNLOAD_DIR, AUDIO_FILE):
    try:
        potential_path = os.path.join(DOWNLOAD_DIR, AUDIO_FILE)
        if os.path.exists(potential_path):
            final_filepath = potential_path
            print(f"Using local file: {final_filepath}")
        elif os.path.exists(AUDIO_FILE):
            final_filepath = AUDIO_FILE
            print(f"Using local file: {final_filepath}")
        else:
            print(f"Local file not found at '{potential_path}' or as '{AUDIO_FILE}'")
            final_filepath = None
    except Exception as e:
        final_path = None
        print(f"Error finding file:{e}")
    finally:
        return final_filepath


def create_subtitle_chunks(segments, max_words=8, max_duration=5.0):
    subtitle_chunks = []

    for segment in segments:
        if hasattr(segment, "words") and segment.words:
            current_chunk = []
            chunk_start = segment.words[0].start

            for i, word in enumerate(segment.words):
                current_chunk.append(word.word)

                if (
                    len(current_chunk) >= max_words
                    or word.end - chunk_start >= max_duration
                ):
                    text = "".join(current_chunk).strip()
                    subtitle_chunks.append(
                        {"start": chunk_start, "end": word.end, "text": text}
                    )

                    current_chunk = []
                    if i + 1 < len(segment.words):
                        chunk_start = segment.words[i + 1].start

            if current_chunk:
                text = "".join(current_chunk).strip()
                subtitle_chunks.append(
                    {"start": chunk_start, "end": segment.words[-1].end, "text": text}
                )
        else:
            subtitle_chunks.append(
                {"start": segment.start, "end": segment.end, "text": segment.text}
            )

    return subtitle_chunks


def format_time(seconds):
    seconds -= 0.2
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"


def add_subtitles(media_path):
    base, ext = os.path.splitext(os.path.basename(media_path))
    dir_path = os.path.dirname(media_path)

    final_output = os.path.join(dir_path, f"{base}_subtitled.mp4")
    subtitle_file = os.path.join(dir_path, f"{base}.srt")

    if not os.path.exists(subtitle_file):
        print(f"Error: Subtitle file not found at {subtitle_file}")
        return

    video_formats = [".mp4", ".webm", ".mpeg"]

    try:
        if ext.lower() in video_formats:
            print("Found video file.")

            temp_output = os.path.join(dir_path, f"{base}_temp.mp4")
            cmd = [
                "ffmpeg",
                "-i",
                media_path,
                "-i",
                subtitle_file,
                "-c",
                "copy",
                "-c:s",
                "mov_text",
                temp_output,
                "-y",
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            if ext.lower() == ".mp4":
                os.remove(media_path)
                os.rename(temp_output, media_path)
            else:
                os.rename(temp_output, final_output)
        else:
            print("Found audio file.")
            temp_video = os.path.join(dir_path, f"{base}_temp.mp4")
            cmd1 = [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720:r=5",
                "-i",
                media_path,
                "-c:a",
                "copy",
                "-shortest",
                temp_video,
                "-y",
            ]
            subprocess.run(cmd1, check=True, capture_output=True)

            cmd2 = [
                "ffmpeg",
                "-i",
                temp_video,
                "-i",
                subtitle_file,
                "-c",
                "copy",
                "-c:s",
                "mov_text",
                final_output,
                "-y",
            ]
            subprocess.run(cmd2, check=True, capture_output=True)
            os.remove(temp_video)

        return final_output

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")


@app.get("/test")
async def test_endpoint():
    return {"message": "FastAPI is working!"}


@app.post("/generate-subtitles")
async def generate_subtitles(
    file: Optional[UploadFile] = File(None), youtube_url: Optional[str] = Form(None)
):
    upload_dir = "/tmp/audio"
    os.makedirs(upload_dir, exist_ok=True)

    if file:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        final_filepath = file_path
        print(f"Uploaded file saved to {final_filepath}")
    elif youtube_url:
        output_template = os.path.join(upload_dir, "%(title)s.%(ext)s")
        final_filepath = youtube_download_video(
            youtube_url, upload_dir, output_template
        )
    else:
        raise HTTPException(
            status_code=400, detail="You must provide either a file or youtube URL."
        )

    if final_filepath and os.path.exists(final_filepath):
        print(f"Processing audio file: {final_filepath}")
        print(f"File size: {os.path.getsize(final_filepath) / 1024 / 1024:.2f} MB")

        base_name = os.path.basename(final_filepath)
        file_name_without_extension, _ = os.path.splitext(base_name)

        FILE_NAME_FOR_TXT = file_name_without_extension
        model_size = "small"

        print(f"\nLoading Whisper model: {model_size}...")
        try:
            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root="/app/models",
            )
            batched_model = BatchedInferencePipeline(model=model)
            print("Model loaded successfully.")

            print("\nStarting transcription...")
            start_time = time.time()

            segments, info = batched_model.transcribe(
                final_filepath, batch_size=8, beam_size=5, word_timestamps=True
            )

            os.makedirs(upload_dir, exist_ok=True)
            transcript_filename = os.path.join(upload_dir, f"{FILE_NAME_FOR_TXT}.srt")

            subtitle_chunks = create_subtitle_chunks(
                segments, max_words=12, max_duration=4.0
            )

            full_transcript_text = []
            for chunk in subtitle_chunks:
                start_time_formatted = format_time(chunk["start"])
                end_time_formatted = format_time(chunk["end"])

                line = (
                    f"{start_time_formatted} --> {end_time_formatted}\n{chunk['text']}"
                )
                full_transcript_text.append(line)

            with open(transcript_filename, "w", encoding="utf-8") as f:
                count = 1
                for line in full_transcript_text:
                    f.write(f"{count}\n{line}\n\n")
                    count += 1

            end_time = time.time()
            processed_time = end_time - start_time

            print(f"\nTranscription complete and saved to {transcript_filename}.")
            print(f"Processed in {processed_time:.2f} seconds")

            video_output = Path(final_filepath).resolve()
            subtitle_output = Path(transcript_filename).resolve()

            files_to_send = [video_output, subtitle_output]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in files_to_send:
                        zf.write(f, arcname=f.name)
                tmp_path = tmp.name

            return FileResponse(
                tmp_path, media_type="application/zip", filename="subtitles.zip"
            )

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        finally:
            if "model" in locals():
                del model
            if "batched_model" in locals():
                del batched_model
            print("Model resources released.")
            import gc

            gc.collect()

    else:
        raise HTTPException(status_code=400, detail="Failed to process the file.")
