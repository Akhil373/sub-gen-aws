FROM public.ecr.aws/lambda/python:3.13

RUN dnf update -y && \
    dnf install -y wget tar xz && \
    dnf clean all


RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xf ffmpeg-release-amd64-static.tar.xz && \
    cp ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    cp ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf ffmpeg-*

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD [ "lambda_function.lambda_handler" ]
