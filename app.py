import json
import os
import shutil
from typing import Optional
import logging
import uuid
from uuid import UUID

import yt_dlp
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import boto3
from boto3.s3.transfer import S3UploadFailedError
from botocore.client import Config, BaseClient

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
    "s3",
    region_name='eu-north-1',
    config=Config(signature_version='s3v4')
)
bucket_name = "subtitle-generator-project"
lambda_client = boto3.client("lambda", region_name="eu-north-1")
dynamodb = boto3.resource('dynamodb', region_name="eu-north-1")
status_table = dynamodb.Table('SubtitleJobStatus')


def youtube_download_video(VIDEO_URL, DOWNLOAD_DIR, output_template):
    URLS = [VIDEO_URL]
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    ydl_opts = {
        'outtmpl': output_template,
        'format': 'bestvideo[height<=1080]+bestaudio/best',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        },
        'merge_output_format': 'mp4',
        'verbose': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading from YouTube: {URLS[0]}")
            info = ydl.extract_info(URLS[0], download=True)
            if not info:
                return None

            final_filepath = None
            if 'requested_downloads' in info and info['requested_downloads']:
                final_filepath = info['requested_downloads'][0]['filepath']
            elif '_filename' in info:
                final_filepath = info['_filename']
            else:
                print("Warning: yt-dlp did not provide a clear filepath. Attempting to construct.")
                if 'title' in info and 'ext' in info:
                    guessed_filename = f"{info['title']}.{info['ext']}"
                    guessed_path = os.path.join(DOWNLOAD_DIR, guessed_filename)
                    if os.path.exists(guessed_path):
                        final_filepath = guessed_path
                    else:
                        print(f"Could not determine downloaded file path for {URLS[0]}.")
                        return None

            return final_filepath

        except Exception as e:
            print(f"An error occurred during YouTube download: {e}")
            return None

def clean_files(path, zip_file, video_path):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        if os.path.exists(zip_file):
            os.remove(zip_file)
        if os.path.exists(video_path):
            os.remove(video_path)
        # if os.path.exists(srt_path):
        #     os.remove(srt_path)
        print("Log: Cleaned all files")
    except Exception as err:
        print("Error clearing files: ", err)

async def store_in_s3(file: UploadFile):
    job_id = uuid.uuid4()
    s3_key = f"uploads/{job_id}/{file.filename}"
    try:
        await file.seek(0)
        s3_resource.upload_fileobj(file.file, bucket_name, s3_key)
        print(f"Uploaded {file.filename} to {bucket_name}")
        return job_id, s3_key
    except S3UploadFailedError as err:
        raise HTTPException(status_code=400, detail=f"Error when uploading to S3: {err}")


@app.get('/test')
async def test_endpoint():
    return {"message": "FastAPI is working!"}


@app.post('/generate-subtitles')
async def generate_subtitles(
        file: Optional[UploadFile] = File(None),
        youtube_url: Optional[str] = Form(None)
):
    upload_dir = '/tmp/audio'
    os.makedirs(upload_dir, exist_ok=True)
    job_id: str | None = None
    s3_key: str | None = None

    if file:
        job_id, s3_key = await store_in_s3(file)

    elif youtube_url:
        output_template = os.path.join(upload_dir, "%(title)s.%(ext)s")
        filepath = youtube_download_video(youtube_url, upload_dir, output_template)

        if filepath is None or not os.path.exists(filepath):
            raise HTTPException(status_code=400, detail="Failed to download YouTube video")

        filename = os.path.basename(filepath)
        job_id: str = str(uuid.uuid4())
        s3_key: str = f"uploads/{job_id}/{filename}"
        s3_resource.upload_file(filepath, bucket_name, s3_key)
    else:
        raise HTTPException(status_code=400, detail="You must provide either a file or youtube URL.")

    job_id_str: str = str(job_id)
    payload = {
        's3_key': s3_key,
        'job_id': job_id_str,
        'bucket_name': bucket_name
    }
    try:
        response = lambda_client.invoke(
            FunctionName='lambda_handler',
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
        print("Lambda invoked successfully:", response)
        return JSONResponse(
            status_code=202,
            content={"message": "Processing started", "job_id": job_id_str}
        )

    except Exception as e:
        print(f'Error invoking Lambda: {e}')
        status_table.put_item(Item={
            'job_id': job_id,
            'status': 'FAILED',
            'message': str(e)
        })
        raise HTTPException(status_code=500, detail=f"Job initialization failed: {str(e)}")


@app.get('/job-status/{job_id}')
def check_job_status(job_id: str):
    response = status_table.get_item(Key={'job_id': job_id})

    if 'Item' not in response:
        raise HTTPException(status_code=404, detail="Job ID not found")

    item = response['Item']
    curr_status = item['status']

    response_body = {
        "job_id": job_id,
        "status": curr_status,
    }

    if curr_status == 'COMPLETED':
        result_key = item['result_key']
        try:
            presigned_url = s3_resource.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': result_key},
                ExpiresIn=3600
            )
            response_body["download_url"] = presigned_url
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return JSONResponse(response_body)
