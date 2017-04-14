# Lab 04
Download inception-v3 model to master `/var/nfs/lab04`:
```sh
$ mkdir /var/nfs/lab04
$ curl -sSL  http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz | tar zx
$ cp inception-v3/* /var/nfs/lab04/
```

Run pv, pvc and deployment:
```sh
$ kubectl create -f pv.yml
$ kubectl get pv,pvc
$ kubectl create -f svc.yml deploy.yml
```

Finally, boot a docker to query image:
```sh
$ docker run --rm -it kairen/serving-base:0.5.1
root@5b9a89eeef5a$ curl "https://s-media-cache-ak0.pinimg.com/736x/32/00/3b/32003bd128bebe99cb8c655a9c0f00f5.jpg" --output rabbit.jpg
root@5b9a89eeef5a$ /opt/serving/bin/example/inception_client --server=localhost:9000 --image=rabbit.jpg
```
