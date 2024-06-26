FROM h4ckermike/inference.petals:main
WORKDIR /usr/src/app/


ADD pip.freeze petals/pip.freeze
RUN pip install --no-cache-dir -r petals/pip.freeze
ADD pip2.freeze petals/pip2.freeze
RUN pip install --no-cache-dir -r petals/pip2.freeze

ADD requirements.txt ./


RUN pip install --no-cache-dir -r ./requirements.txt --upgrade pip


#RUN transformers==4.38.2
#ADD . /usr/src/app/


#pip install git+https://github.com/huggingface/transformers
ADD config.py /usr/src/app/
ADD  utils.py /usr/src/app/


CMD gunicorn app:app --bind 0.0.0.0:5000 --worker-class gthread --threads 100 --timeout 1000
