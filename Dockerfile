FROM public.ecr.aws/lambda/python:3.9

ENV TF_CPP_MIN_LOG_LEVEL=2

RUN yum update -y && yum install amazon-linux-extras -y && PYTHON=python2 amazon-linux-extras install epel -y && yum update -y && yum install git-lfs -y

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://huggingface.co/plsakr/vit-garbage-classification-v2 plsakr/vit-garbage-classification-v2

COPY . .

CMD [ "handler.predict"]
