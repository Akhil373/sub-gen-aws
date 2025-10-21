import json
import os
import shutil
import traceback

import boto3
from botocore.client import BaseClient, Config
from faster_whisper import BatchedInferencePipeline, WhisperModel

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"

dynamodb = boto3.resource("dynamodb", region_name="eu-north-1")
status_table = dynamodb.Table("SubtitleJobStatus")

s3_resource: BaseClient = boto3.client(
    "s3", region_name="eu-north-1", config=Config(signature_version="s3v4")
)


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


def clean_files(path, zip_file, video_path):
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path)
        if zip_file and os.path.exists(zip_file):
            os.remove(zip_file)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        print("Log: Cleaned all files")
    except Exception as err:
        print("Error clearing files: ", err)


def lambda_handler(event, context):
    payload = event if "body" not in event else json.loads(event["body"])
    s3_key: str = payload["s3_key"]
    job_id: str = payload["job_id"]
    bucket_name: str = payload["bucket_name"]
    total_chunks: int = payload["total_chunks"]
    chunk_index: int = payload["chunk_index"]

    original_filename: str = s3_key.split("/")[-1]
    final_filepath: str = f"/tmp/{original_filename}"
    download_dir: str = "/tmp/audio"

    transcript_filename = None

    try:
        status_table.put_item(
            Item={
                "job_id": job_id,
                "chunk_index": chunk_index,
                "status": "PROCESSING CHUNK",
                "message": f"Processing chunk {chunk_index}/{total_chunks}",
                "total_chunks": total_chunks,
            }
        )
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
                model_size, device="cpu", compute_type="int8", download_root=model_dir
            )
            batched_model = BatchedInferencePipeline(model=model)
            print("Model loaded successfully.")

            print("\nStarting transcription...")
            # start_time = time.time()

            segments, info = batched_model.transcribe(
                final_filepath, batch_size=4, beam_size=5, word_timestamps=True
            )

            os.makedirs(download_dir, exist_ok=True)
            transcript_filename = os.path.join(download_dir, f"{FILE_NAME_FOR_TXT}.srt")

            transcription_data = {
                "job_id": job_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "original_file": original_filename,
                "segments": [],
            }

            subtitle_chunks = create_subtitle_chunks(
                segments, max_words=12, max_duration=4.0
            )

            for segment in subtitle_chunks:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                }

                if hasattr(segment, "words") and segment.words:
                    segment_data["words"] = []
                    for word in segment.words:
                        segment_data["words"].append(
                            {"word": word.word, "start": word.start, "end": word.end}
                        )

                transcription_data["segments"].append(segment_data)

            # end_time = time.time()
            # processed_time = end_time - start_time

            # print(f"\nTranscription complete and saved to {transcript_filename}.")
            # print(f"Processed in {processed_time:.2f} seconds")

            json_filename = os.path.join(download_dir, f"{FILE_NAME_FOR_TXT}.json")
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(transcription_data, f, indent=2, ensure_ascii=False)

            result_key = f"transcriptions/{job_id}/chunk_{chunk_index:03d}.json"
            s3_resource.upload_file(json_filename, bucket_name, result_key)

            status_table.put_item(
                Item={
                    "job_id": job_id,
                    "chunk_index": chunk_index,
                    "status": "COMPLETED_CHUNK",
                    "result_key": result_key,
                    "message": f"Chunk {chunk_index} processed successfully",
                }
            )

            print(f"SUCCESS: Job {job_id} completed.")
            return {
                "statusCode": 200,
                "body": json.dumps("Processing completed successfully."),
            }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        stack_trace = traceback.format_exc()
        print(f"ERROR processing job {job_id}: {error_msg}")
        print(f"Stack Trace:\n{stack_trace}")

        try:
            status_table.put_item(
                Item={"job_id": job_id, "status": "FAILED", "message": error_msg}
            )
        except Exception as db_e:
            print(
                f"DOUBLE FAULT: Also failed to save ERROR status to DynamoDB. Original error: {error_msg}. Dynamo error: {db_e}"
            )
        raise e

    finally:
        if "model" in locals():
            del model
        if "batched_model" in locals():
            del batched_model
        clean_files(download_dir, None, final_filepath)
        print("Model resources released.")
        import gc

        gc.collect()
