FROM python:3.6
ADD newspaper /newspaper
ADD app.py /
ADD requirements.txt /
RUN pip install -r requirements.txt
EXPOSE 80
CMD [ "python", "./app.py" ]