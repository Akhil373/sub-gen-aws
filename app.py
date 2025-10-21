import asyncio
import glob
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import aioboto3
import boto3
import yt_dlp
from botocore.client import BaseClient, Config
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fast-subs.vercel.app"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

s3_resource: BaseClient = boto3.client(
    "s3", region_name="eu-north-1", config=Config(signature_version="s3v4")
)
bucket_name = "subtitle-generator-project"
lambda_client = boto3.client("lambda", region_name="eu-north-1")
dynamodb = boto3.resource("dynamodb", region_name="eu-north-1")
status_table = dynamodb.Table("SubtitleJobStatus")


def format_time(seconds):
    seconds -= 0.2
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"


def youtube_download_video(VIDEO_URL, DOWNLOAD_DIR, output_template):
    URLS = [VIDEO_URL]
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    ydl_opts = {
        "outtmpl": output_template,
        "format": "bestvideo[height<=1080]+bestaudio/best",
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
        "merge_output_format": "mp4",
        "verbose": True,
        "cookiefile": "youtube_cookies.txt",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading from YouTube: {URLS[0]}")
            info = ydl.extract_info(URLS[0], download=True)
            if not info:
                return None

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
                        return None

            return final_filepath

        except Exception as e:
            print(f"An error occurred during YouTube download: {e}")
            return None


def clean_files(path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        print("Log: Cleaned all files")
    except Exception as err:
        print("Error clearing files: ", err)


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix.lower()


async def upload_chunks_to_s3(chunk_files: list, job_id: str, file_extension: str):
    async with aioboto3.Session().client("s3") as s3:
        tasks = []

        for index, chunk_file in enumerate(chunk_files):
            s3_key = f"chunks/{job_id}/chunk_{index:03d}{file_extension}"
            task = s3.upload_file(chunk_file, bucket_name, s3_key)
            tasks.append(task)

        await asyncio.gather(*tasks)


def split_file_into_chunks(
    filepath: str, job_id: str, chunk_duration: int = 60
) -> List[str]:
    original_ext = get_file_extension(filepath)
    output_pattern = f"/tmp/audio/{job_id}/chunk_%03d{original_ext}"
    try:
        os.makedirs(f"/tmp/audio/{job_id}", exist_ok=True)
        cmd = [
            "ffmpeg",
            "-i",
            filepath,
            "-f",
            "segment",
            "-segment_time",
            str(chunk_duration),
            "-reset_timestamps",
            "1",
            "-c",
            "copy",
            output_pattern,
        ]
        subprocess.run(cmd, check=True)

        chunk_files = sorted(glob.glob(f"/tmp/audio/{job_id}/chunk_*{original_ext}"))
        return chunk_files
    except Exception as e:
        raise HTTPException(
            status_code=400, details=f"Error whe splitting file: {e}"
        ) from e


def trigger_lambda(job_id: str, total_chunks: int):
    for chunk_index in range(total_chunks):
        response = lambda_client.invoke(
            FunctionName="subtitle-gen-lambda",
            InvocationType="Event",
            Payload=json.dumps(
                {
                    "job_id": str(job_id),
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "s3_key": f"chunks/{job_id}/chunk_{chunk_index:03d}.mp4",
                    "bucket_name": bucket_name,
                }
            ),
        )
    return response


def add_subtitles(media_path, subtitle_file):
    base, ext = os.path.splitext(os.path.basename(media_path))
    dir_path = os.path.dirname(media_path)

    final_output = os.path.join(dir_path, f"{base}_subtitled.mp4")

    if not os.path.exists(subtitle_file):
        print(f"Error: Subtitle file not found at {subtitle_file}")
        return None

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


def combine_segments_with_timing(all_segments: list, total_chunks: int) -> list:
    chunk_duration = 60
    combined = []

    for chunk_index in range(total_chunks):
        chunk_segments = [
            s for s in all_segments if s.get("chunk_index") == chunk_index
        ]

        time_offset = chunk_index * chunk_duration
        for segment in chunk_segments:
            adjusted_segment = segment.copy()
            adjusted_segment["start"] += time_offset
            adjusted_segment["end"] += time_offset
            combined.append(adjusted_segment)

        combined.sort(key=lambda x: x["start"])
        return combined


def create_final_srt_file(segments: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_time(segment["start"])
            end_time = format_time(segment["end"])
            text = segment["text"]

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

    print(f"Created final SRT file with {len(segments)} segments")


def download_original_video(job_id: str, filename: str) -> str:
    original_key = f"originals/{job_id}/original_{filename}"
    local_path = f"/tmp/audio/{job_id}_original_{filename}"

    s3_resource.download_file(bucket_name, original_key, local_path)
    return local_path


def assemble_final_video(job_id: str, total_chunks: int, filename: str):
    try:
        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": -1,
                "status": "ASSEMBLING",
                "message": "Combining transcriptions and creating final video",
            }
        )

        all_segments = []
        for chunk_index in range(total_chunks):
            s3_key = f"transcriptions/{job_id}/chunk_{chunk_index:03d}.json"
            local_json_path = f"/tmp/audio/{job_id}_chunk_{chunk_index}.json"

            try:
                s3_resource.download_file(bucket_name, s3_key, local_json_path)
                with open(local_json_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    all_segments.extend(chunk_data["segments"])
                print(f"Downloaded chunk {chunk_index}")
            except Exception as e:
                print(f"Error downloading chunk {chunk_index}: {e}")
                continue

        if not all_segments:
            raise Exception("No transcription data found for any chunks")

        combined_segments = combine_segments_with_timing(all_segments, total_chunks)

        srt_filename = f"/tmp/{job_id}_final.srt"
        create_final_srt_file(combined_segments, srt_filename)

        original_video_path = download_original_video(job_id, filename)

        final_video_path = add_subtitles(original_video_path, srt_filename)
        final_key = f"results/{job_id}/final_subtitled_video.mp4"
        s3_resource.upload_file(final_video_path, bucket_name, final_key)

        # 6. Update status to completed
        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": -1,
                "status": "COMPLETED",
                "result_key": final_key,
                "message": "Final video with subtitles created successfully",
            }
        )

        print(f"SUCCESS: Final assembly completed for job {job_id}")

    except Exception as e:
        error_msg = f"Final assembly failed: {str(e)}"
        print(error_msg)
        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": -1,
                "status": "FAILED",
                "message": error_msg,
            }
        )


def assemble_final_video(job_id: str, total_chunks: int):
    try:
        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": -1,
                "status": "ASSEMBLING",
                "message": "Combining transcriptions and creating final video",
            }
        )

        resp = status_table.get_item(
            Key={"job_id": job_id},
            ProjectionExpression="original_filename",
        )
        if "Item" not in resp or not resp["Item"].get("original_filename"):
            raise RuntimeError("original_filename not found in DynamoDB")
        original_basename = os.path.splitext(resp["Item"]["original_filename"])[0]

        all_segments = []
        for chunk_index in range(total_chunks):
            s3_key = f"transcriptions/{job_id}/chunk_{chunk_index:03d}.json"
            local_json_path = f"/tmp/audio/{job_id}_chunk_{chunk_index}.json"
            try:
                s3_resource.download_file(bucket_name, s3_key, local_json_path)
                with open(local_json_path, encoding="utf-8") as f:
                    all_segments.extend(json.load(f)["segments"])
            except Exception as e:
                print(f"Warning: could not load chunk {chunk_index}: {e}")
                continue

        if not all_segments:
            raise RuntimeError("No transcription segments found")

        combined_segments = combine_segments_with_timing(all_segments, total_chunks)

        srt_path = f"/tmp/audio/{job_id}_final.srt"
        create_final_srt_file(combined_segments, srt_path)

        original_video_path = download_original_video(
            job_id, resp["Item"]["original_filename"]
        )

        final_video_local = add_subtitles(original_video_path, srt_path)
        if not final_video_local:
            raise RuntimeError("Subtitle burn-in failed")

        final_s3_key = f"results/{job_id}/{original_basename}_subtitled.mp4"
        s3_resource.upload_file(final_video_local, bucket_name, final_s3_key)

        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": -1,
                "status": "COMPLETED",
                "result_key": final_s3_key,
                "message": "Final video with subtitles created successfully",
            }
        )
        print(f"SUCCESS: final video {final_s3_key} assembled for job {job_id}")

    except Exception as e:
        error_msg = f"Final assembly failed: {str(e)}"
        print(error_msg)
        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": -1,
                "status": "FAILED",
                "message": error_msg,
            }
        )


@app.get("/test")
async def test_endpoint():
    return {"message": "FastAPI is working!"}


@app.post("/generate-subtitles")
async def generate_subtitles(
    file: Optional[UploadFile] = File(None), youtube_url: Optional[str] = Form(None)
):
    upload_dir = "/tmp/audio"
    os.makedirs(upload_dir, exist_ok=True)
    job_id: UUID = uuid.uuid4()
    filepath: str | None = None

    if file:
        filepath = Path(upload_dir) / file.filename
        with open(filepath, "wb", encoding="utf-8") as f:
            shutil.copyfileobj(file.file, f)

        print(f"File saved successfully in {filepath}")
    elif youtube_url:
        output_template = os.path.join(upload_dir, "%(title)s.%(ext)s")
        filepath = youtube_download_video(youtube_url, upload_dir, output_template)

        if filepath is None or not os.path.exists(filepath):
            raise HTTPException(
                status_code=400, detail="Failed to download YouTube video"
            )

    else:
        raise HTTPException(
            status_code=400, detail="You must provide either a file or youtube URL."
        )

    job_id_str: str = str(job_id)

    original_key = f"originals/{job_id_str}/original_{os.path.basename(filepath)}"
    s3_resource.upload_file(filepath, bucket_name, original_key)

    status_table.put_item(
        Item={
            "job_id": job_id_str,
            "chunk_index": -1,
            "status": "STARTED",
            "original_filename": os.path.basename(filepath),
            "total_chunks": 0,
        }
    )
    chunk_files = split_file_into_chunks(filepath, job_id_str)
    file_extension = get_file_extension(filepath)
    await upload_chunks_to_s3(chunk_files, job_id, file_extension)

    try:
        response = trigger_lambda(job_id, len(chunk_files))
        print("Lambda invoked successfully:", response)

    except Exception as e:
        print(f"Error invoking Lambda: {e}")
        status_table.put_item(
            Item={"job_id": job_id_str, "status": "FAILED", "message": str(e)}
        )
        raise HTTPException(
            status_code=500, detail=f"Job initialization failed: {str(e)}"
        ) from e

    try:
        clean_files(upload_dir)
        print("cleaned up temp files")
    except Exception as e:
        print("Error clearning on tmp files: " + e)

    status_table.update_item(
        Key={"job_id": job_id_str},
        UpdateExpression="SET total_chunks = :tc",
        ExpressionAttributeValues={":tc": len(chunk_files)},
    )
    return {"job_id": job_id, "status": "started"}


@app.get("/job-status/{job_id}")
def check_job_status(job_id: str):
    response = status_table.query(
        KeyConditionExpression=boto3.dynamodb.conditions.Key("job_id").eq(job_id)
    )
    if "Items" not in response or len(response["Items"]) == 0:
        raise HTTPException(status_code=404, detail="Job ID not found")

    items = response["Items"]
    overall_job = None
    chunk_statuses = []

    for item in items:
        if item.get("chunk_index") == -1:
            overall_job = item
        else:
            chunk_statuses.append(item)

    if not overall_job:
        raise HTTPException(status_code=404, detail="Job not found")

    total_chunks = overall_job.get("total_chunks", 0)
    completed_chunks = sum(
        1 for chunk in chunk_statuses if chunk.get("status") == "COMPLETED_CHUNK"
    )
    failed_chunks = sum(
        1 for chunk in chunk_statuses if chunk.get("status") == "FAILED_CHUNK"
    )

    response_body = {
        "job_id": job_id,
        "status": overall_job.get("status", "PROCESSING"),
        "progress": f"{completed_chunks}/{total_chunks}",
        "completed_chunks": completed_chunks,
        "total_chunks": total_chunks,
        "failed_chunks": failed_chunks,
    }

    if completed_chunks >= total_chunks and overall_job.get("status") not in [
        "ASSEMBLING",
        "COMPLETED",
    ]:
        import threading

        thread = threading.Thread(
            target=assemble_final_video, args=(job_id, total_chunks)
        )
        thread.daemon = True
        thread.start()

        response_body["status"] = "ASSEMBLING"

    elif overall_job.get("status") == "COMPLETED":
        result_key = overall_job.get("result_key")
        try:
            presigned_url = s3_resource.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": result_key},
                ExpiresIn=3600,
            )
            response_body["download_url"] = presigned_url
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return JSONResponse(response_body)
