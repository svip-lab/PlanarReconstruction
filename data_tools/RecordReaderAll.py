# modified from https://github.com/art-programmer/PlaneNet
import tensorflow as tf

HEIGHT=192
WIDTH=256
NUM_PLANES = 20
NUM_THREADS = 4


class RecordReaderAll:
    def __init__(self):
        return

    def getBatch(self, filename_queue, batchSize=1, min_after_dequeue=1000,
                 random=False, getLocal=False, getSegmentation=False, test=True):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'image_path': tf.FixedLenFeature([], tf.string),
                'num_planes': tf.FixedLenFeature([], tf.int64),
                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
                'segmentation_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
                'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
                'semantics_raw': tf.FixedLenFeature([], tf.string),                
                'boundary_raw': tf.FixedLenFeature([], tf.string),
                'info': tf.FixedLenFeature([4 * 4 + 4], tf.float32),                
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [HEIGHT, WIDTH, 3])

        depth = features['depth']
        depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])

        normal = features['normal']
        normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])

        semantics = tf.decode_raw(features['semantics_raw'], tf.uint8)
        semantics = tf.cast(tf.reshape(semantics, [HEIGHT, WIDTH]), tf.int32)

        numPlanes = tf.cast(features['num_planes'], tf.int32)

        planes = features['plane']
        planes = tf.reshape(planes, [NUM_PLANES, 3])
        
        boundary = tf.decode_raw(features['boundary_raw'], tf.uint8)
        boundary = tf.cast(tf.reshape(boundary, (HEIGHT, WIDTH, 2)), tf.float32)

        segmentation = tf.decode_raw(features['segmentation_raw'], tf.uint8)
        segmentation = tf.reshape(segmentation, [HEIGHT, WIDTH, 1])

        image_inp, plane_inp, depth_gt, normal_gt, semantics_gt, segmentation_gt, boundary_gt, num_planes_gt, image_path, info = \
            tf.train.batch([image, planes, depth, normal, semantics, segmentation, boundary, numPlanes, features['image_path'], features['info']], batch_size=batchSize, capacity=(NUM_THREADS + 2) * batchSize, num_threads=1)
        global_gt_dict = {'plane': plane_inp, 'depth': depth_gt, 'normal': normal_gt, 'semantics': semantics_gt,
                          'segmentation': segmentation_gt, 'boundary': boundary_gt, 'num_planes': num_planes_gt,
                          'image_path': image_path, 'info': info}
        return image_inp, global_gt_dict, {}
