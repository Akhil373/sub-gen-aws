import os
import json
import shutil
from pathlib import Path
import tempfile
import zipfile
import traceback
import time
from faster_whisper import WhisperModel, BatchedInferencePipeline
import subprocess
import boto3
from botocore.client import BaseClient, Config

dynamodb = boto3.resource('dynamodb', region_name="eu-north-1")
status_table = dynamodb.Table('SubtitleJobStatus')

s3_resource: BaseClient = boto3.client(
    "s3",
    region_name='eu-north-1',
    config=Config(signature_version='s3v4')
)

def create_subtitle_chunks(segments, max_words=8, max_duration=5.0):
    subtitle_chunks = []

    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            current_chunk = []
            chunk_start = segment.words[0].start

            for i, word in enumerate(segment.words):
                current_chunk.append(word.word)

                if (len(current_chunk) >= max_words or
                        word.end - chunk_start >= max_duration):

                    text = ''.join(current_chunk).strip()
                    subtitle_chunks.append({
                        'start': chunk_start,
                        'end': word.end,
                        'text': text
                    })

                    current_chunk = []
                    if i + 1 < len(segment.words):
                        chunk_start = segment.words[i + 1].start

            if current_chunk:
                text = ''.join(current_chunk).strip()
                subtitle_chunks.append({
                    'start': chunk_start,
                    'end': segment.words[-1].end,
                    'text': text
                })
        else:
            subtitle_chunks.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text
            })

    return subtitle_chunks

def format_time(seconds):
    seconds -= 0.2
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remainder = seconds % 60
    milliseconds = int((seconds_remainder - int(seconds_remainder)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{int(seconds_remainder):02d},{milliseconds:03d}"

def add_subtitles(media_path, subtitle_file):
    base, ext = os.path.splitext(os.path.basename(media_path))
    dir_path = os.path.dirname(media_path)

    final_output = os.path.join(dir_path, f"{base}_subtitled.mp4")

    if not os.path.exists(subtitle_file):
        print(f"Error: Subtitle file not found at {subtitle_file}")
        return None

    video_formats = ['.mp4', '.webm', '.mpeg']

    try:
        if ext.lower() in video_formats:
            print('Found video file.')

            temp_output = os.path.join(dir_path, f"{base}_temp.mp4")
            cmd = ['ffmpeg', '-i', media_path, '-i', subtitle_file, '-c', 'copy', '-c:s', 'mov_text', temp_output, '-y']

            subprocess.run(cmd, check=True, capture_output=True)

            if ext.lower() == ".mp4":
                os.remove(media_path)
                os.rename(temp_output, media_path)
            else:
                os.rename(temp_output, final_output)
        else:
            print('Found audio file.')
            temp_video = os.path.join(dir_path, f"{base}_temp.mp4")
            cmd1 = ['ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720:r=5',
                    '-i', media_path, '-c:a', 'copy', '-shortest', temp_video, '-y']
            subprocess.run(cmd1, check=True, capture_output=True)

            cmd2 = ['ffmpeg', '-i', temp_video, '-i', subtitle_file, '-c',
                    'copy', '-c:s', 'mov_text', final_output, '-y']
            subprocess.run(cmd2, check=True, capture_output=True)
            os.remove(temp_video)

        return final_output

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")

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


def lambda_handler(event, context):
    # current_path = os.environ['PATH']
    # os.environ['PATH'] = os.environ['LAMBDA_TASK_ROOT'] + ':' + current_path

    payload = event if 'body' not in event else json.loads(event['body'])
    s3_key: str = payload['s3_key']
    job_id: str = payload['job_id']
    bucket_name: str = payload['bucket_name']

    final_filepath: str = f"/tmp/{job_id}_video.mp4"
    download_dir: str = '/tmp/audio'
    tmp_path = None

    try:
        status_table.put_item(Item={
            'job_id': job_id,
            'status': 'PROCESSING',
            'message': 'Downloading file and starting transcription...'
        })
    except Exception as e:
        print(f"Failed to write START status to DynamoDB. Error: {e}")

    try:
        s3_resource.download_file(bucket_name, s3_key, final_filepath)
        if final_filepath and os.path.exists(final_filepath):
            print(f"Processing audio file: {final_filepath}")
            print(f"File size: {os.path.getsize(final_filepath) / 1024 / 1024:.2f} MB")

            base_name = os.path.basename(final_filepath)
            file_name_without_extension, _ = os.path.splitext(base_name)

            FILE_NAME_FOR_TXT = file_name_without_extension
            model_size = "small"

            print(f"\nLoading Whisper model: {model_size}...")
            model_dir = "/tmp/models"
            os.makedirs(model_dir, exist_ok=True)

            model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8",
                download_root=model_dir
            )
            batched_model = BatchedInferencePipeline(model=model)
            print("Model loaded successfully.")

            print("\nStarting transcription...")
            start_time = time.time()

            segments, info = batched_model.transcribe(
                final_filepath,
                batch_size=4,
                beam_size=5,
                word_timestamps=True
            )

            os.makedirs(download_dir, exist_ok=True)
            transcript_filename = os.path.join(download_dir, f"{FILE_NAME_FOR_TXT}.srt")

            subtitle_chunks = create_subtitle_chunks(segments, max_words=12, max_duration=4.0)

            full_transcript_text = []
            for chunk in subtitle_chunks:
                start_time_formatted = format_time(chunk['start'])
                end_time_formatted = format_time(chunk['end'])

                line = f"{start_time_formatted} --> {end_time_formatted}\n{chunk['text']}"
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

            add_subtitles(video_output, subtitle_output)

            files_to_send = [video_output, subtitle_output]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in files_to_send:
                        zf.write(f, arcname=f.name)
                tmp_path = tmp.name

            result_key = f"results/{job_id}/subtitled_files.zip"
            s3_resource.upload_file(tmp_path, bucket_name, result_key)

            status_table.put_item(Item={
                'job_id': job_id,
                'status': 'COMPLETED',
                'result_key': f"results/{job_id}/subtitled_files.zip",
                'message': 'Subtitles generated successfully.'
            })

            print(f"SUCCESS: Job {job_id} completed.")
            return {
                'statusCode': 200,
                'body': json.dumps('Processing completed successfully.')
            }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        stack_trace = traceback.format_exc()
        print(f"ERROR processing job {job_id}: {error_msg}")
        print(f"Stack Trace:\n{stack_trace}")

        try:
            status_table.put_item(Item={
                'job_id': job_id,
                'status': 'FAILED',
                'message': error_msg
            })
        except Exception as db_e:

            print(
                f"DOUBLE FAULT: Also failed to save ERROR status to DynamoDB. Original error: {error_msg}. Dynamo error: {db_e}")
        raise e

    finally:
        if 'model' in locals():
            del model
        if 'batched_model' in locals():
            del batched_model
        print("Model resources released.")
        clean_files(download_dir, tmp_path, final_filepath)
        import gc
        gc.collect()
