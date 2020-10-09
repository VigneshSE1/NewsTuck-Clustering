FROM python:3.8
ADD api.py /
ADD requirements.txt /
RUN pip install -r requirements.txt
CMD [ "python", "./api.py" ]