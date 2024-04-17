FROM h4ckermike/inference.petals:main
WORKDIR /usr/src/app/

#pip install git+https://github.com/huggingface/transformers
ADD config.py /usr/src/app/
ADD  utils.py /usr/src/app/