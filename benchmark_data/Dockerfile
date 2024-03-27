FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip openjdk-21-jre postgresql-client curl zstd wget
RUN pip3 install jupyter

WORKDIR /benchmark

EXPOSE 8888
EXPOSE 4040
EXPOSE 4041-4050

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--allow-root", "--NotebookApp.token=my_secret_token2"]
