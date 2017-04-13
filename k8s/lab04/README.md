# Lab 04
Run a Incepttion v3 serving:
- [Inception v3](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz")

Run a query:
```sh
$ curl "https://s-media-cache-ak0.pinimg.com/736x/32/00/3b/32003bd128bebe99cb8c655a9c0f00f5.jpg" --output rabbit.jpg
$ docker run --rm -it kairen/serving-base:0.5.1
root@5b9a89eeef5a$ bin/example/inception_client --server=localhost:9000 --image=rabbit.jpg
```
