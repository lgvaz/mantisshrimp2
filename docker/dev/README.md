This repo provides a basic docker image with a full blown development environment using `icevision`.
To run docker, you need some prerequisites:
1. NVIDIA drivers for your card (CUDA >= 11.0 compatible)
2. Docker & [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Once you have the prerequisites, you can run `./docker-build.sh` to build the image. Then, run `docker-run.sh` to open up the container.

Once inside the docker image, you have the option to start up a Jupyter notebook server by running
```bash
/run-jupyter.sh
```

This starts a server on the port `8889`. Assuming you're running the image on a remote machine, you can connect to it using the following command:
```bash
ssh -N -f -L localhost:8889:localhost:8889 <USER>@<PUBLIC-IP-ADDRESS>
```
You will be prompted with a password on opening. By default, this is set to `ice-dev`. You can modify it in the `run-jupyter.sh` file.