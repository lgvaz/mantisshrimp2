# Base image

# FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

# Tested w/ (some) RTX & GTX cards
FROM nvcr.io/nvidia/pytorch:20.12-py3


# Metadata
LABEL maintainer=""
LABEL environment_type="Icevision Dev"


# ===================== STANDARD LINUX BOILERPLATE ================== #

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

# Run commands on initialising the container
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get clean

# Build Env
RUN apt-get install -y \
    apt-utils \
    build-essential \
    byobu \
    bzip2 \
    ca-certificates \
    curl \
    git-core \
    htop \
    libcurl4-openssl-dev \
    libssl-dev wget \
    pkg-config \
    sudo \
    tree \
    wget \
    unzip

RUN pip install --upgrade pip
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*


RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-get update

# ===================== TEXT EDITOR ==================== #

RUN add-apt-repository ppa:kelleyk/emacs
RUN apt update
RUN apt install emacs -y
RUN apt-get update

#TODO: vscode-server




# ============= INSTALL PYTHON PACKAGES VIA PIP ============= #

COPY requirements.txt ./
COPY no-dep-requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r no-dep-requirements.txt --no-dependencies
RUN pip uninstall torchtext -y

# =========================================================== #


# ============ JUPYTER SETUP - THEMES, EXTENSIONS =========== #

## Uncomment the code below for an IDE like Jupyter theme

# Custom Dark (Ocean) Theme

# WORKDIR /root/.jupyter/
# RUN mkdir custom
# WORKDIR /root/.jupyter/custom
# RUN wget https://raw.githubusercontent.com/rsomani95/jupyter-custom-theme/master/custom.css
# WORKDIR /root/.ipython/profile_default/startup
# RUN wget https://raw.githubusercontent.com/rsomani95/jupyter-custom-theme/master/startup.ipy
# WORKDIR /root

# Expose port for Jupyter - Necessary??
EXPOSE 8889


# Jupyter Extensions

RUN jupyter nbextension enable --py widgetsnbextension
RUN jupyter contrib nbextension install --user
RUN jupyter nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
RUN jupyter nbextension enable jupyter-black-master/jupyter-black

# =========================================================== #


# ========== DREADED MM-CV AND MMDETECTION INSTALLATION ========== #

COPY install-mm.sh ./
RUN bash install-mm.sh

# ================================================================ #




# ========================== ENTRYPOINT =========================== #

# There can only be one CMD instruction in a Dockerfile. If you list more 
#     than one CMD then only the last CMD will take effect.
# The main purpose of a CMD is to provide defaults for an executing container.

# You could start a bash shell and be inside it OR
# You could start a notebook server (in the background?)

COPY run-jupyter.sh /run-jupyter.sh
RUN chmod +x /run-jupyter.sh
#CMD ["/run-jupyter.sh"]
CMD ["/bin/bash"]