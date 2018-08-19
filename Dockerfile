FROM ceshine/cuda-pytorch:0.4.1

MAINTAINER CeShine Lee <ceshine@ceshine.net>

RUN pip install --upgrade pip && \
  pip install -U jupyter h5py pandas==0.22.0 sklearn matplotlib seaborn plotly watermark \
                 pillow-simd joblib tqdm jupyter_contrib_nbextensions spacy cupy && \
  rm -rf ~/.cache/pip

RUN jupyter contrib nbextension install --user
RUN jupyter nbextension enable collapsible_headings/main

# TODO: create a separate group for 'docker' user
COPY --chown=docker:root fastai/ /home/docker/fastai
# COPY --chown=docker:root notebooks /home/docker/project/notebooks
COPY --chown=docker:root jupyter_notebook_config.json /home/docker/project/
COPY --chown=docker:root jupyter_notebook_config.py /home/docker/project/

RUN cd /home/docker/fastai && pip install -e  .

RUN pip uninstall -y opencv-python
RUN conda install -y opencv

WORKDIR /home/docker/project

# Jupyter
EXPOSE 8888
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --config=jupyter_notebook_config.json --no-browser
