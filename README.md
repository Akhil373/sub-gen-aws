# Automated Subtitle Generation Pipeline

Serverless, fault-tolerant service that produces time-synchronized subtitles for arbitrary online video.  
Built on AWS Free-Tier components; no persistent servers required.

## Core Flow
1. User submits URL or file → FastAPI (t3.micro EC2)
2. yt-dlp downloads → S3 originals/{job_id}/
3. FFmpeg splits into 3-min overlapping chunks → S3 chunks/{job_id}/
4. One SQS message per chunk → Lambda (3 GB, 10 GB /tmp)
5. Lambda runs faster-whisper (tiny) → JSON → S3 transcriptions/{job_id}/
6. EC2 merges JSON → .srt, burns into video with FFmpeg → S3 results/{job_id}/
7. Presigned, time-limited HTTPS download URL returned

## Stack
- *Compute*: Lambda (container image, ECR), EC2 t3.micro (orchestration only)  
- *Storage*: S3 with 24-h lifecycle  
- *Queue*: SQS (5-min visibility, DLQ)  
- *State*: DynamoDB (job_id PK, chunk status, timestamps)  
- *IAM*: least-privilege roles, path-scoped S3/SQS/DynamoDB rights  
- *Observability*: CloudWatch metrics & logs, CloudTrail audit  

## Cost (Free-Tier)
Under free tier.
