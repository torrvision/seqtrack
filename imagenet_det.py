import csv
import functools
import os
import progressbar
import xmltodict

num_classes = 200


def load(ilsvrc_dir, set_name, cache_func=None):
    if cache_func is None:
        # Call function.
        cache_func = lambda f: f()
    metadata = cache_func(lambda: _load_metadata(ilsvrc_dir, set_name))
    if not 'image_labels' in metadata:
        assert set_name != 'train'
        metadata['image_labels'] = _labels_from_annotations(
            classes=metadata['classes'],
            images=metadata['images'],
            image_annotations=metadata['image_annotations'],
        )
    return Dataset(ilsvrc_dir, set_name,
        classes=metadata['classes'],
        images=metadata['images'],
        image_labels=metadata['image_labels'],
        image_annotations=metadata['image_annotations'])


def load_classes(ilsvrc_dir):
    fname = os.path.join(ilsvrc_dir, 'devkit', 'data', 'map_det.txt')
    fieldnames = ['wordnet_id', 'index', 'name']
    with open(fname, 'r') as f:
        rows = list(csv.DictReader(f, delimiter=' ', fieldnames=fieldnames))
    assert([int(r['index']) for r in rows] == range(1, num_classes+1))
    classes = [r['wordnet_id'] for r in rows]
    names = {r['wordnet_id']: r['name'] for r in rows}
    return classes, names


class Dataset:
    def __init__(self, ilsvrc_dir, set_name, classes, images, image_labels, image_annotations):
        self._ilsvrc_dir        = ilsvrc_dir
        self._set_name          = set_name
        self._classes           = classes
        self._images            = images
        self._image_labels      = image_labels
        self._image_annotations = image_annotations

    def image_file(self, im):
        parts = im.split('/')
        parts[-1] += '.JPEG'
        return os.path.join(self._ilsvrc_dir, 'Data', 'DET', self._set_name, *parts)

    def num_classes(self):
        return len(self._classes)

    def classes(self):
        return self._classes

    def images(self):
        return self._images

    def image_size(self, image):
        annotation = self._image_annotations[image]
        if annotation is None:
            raise ValueError('image does not have annotation (negative image)')
        size = annotation['size']
        return (int(size['width']), int(size['height']))

    def multi_class_label(self, image):
        return self._image_labels[image]

    def image_objects(self, image):
        annotation = self._image_annotations[image]
        if annotation is None:
            return []
        convert = lambda obj: _convert_object(obj, annotation['size'])
        return map(convert, annotation.get('object', []))


def _convert_object(obj, size):
    return obj['name'], {
        'xmin':  float(obj['bndbox']['xmin']) / float(size['width']),
        'ymin':  float(obj['bndbox']['ymin']) / float(size['height']),
        'xmax':  float(obj['bndbox']['xmax']) / float(size['width']),
        'ymax':  float(obj['bndbox']['ymax']) / float(size['height']),
    }


def _load_metadata(ilsvrc_dir, set_name):
    if set_name == 'train':
        return _load_metadata_train(ilsvrc_dir)
    elif set_name == 'val':
        return _load_metadata_val(ilsvrc_dir)
    else:
        return ValueError('unknown set: {}'.format(set_name))


def _load_metadata_train(ilsvrc_dir):
    '''Loads metadata of training set.

    metadata['classes'] -- List of class IDs.
    metadata['image_labels'] -- Dictionary that maps image ID to:
        dictionary that maps class to label in {-1, 0, 1}.
    metadata['images'] -- List of image IDs.
    metadata['image_annotations'] -- Dictionary that maps image ID to annotation.
    '''
    # Create lookup from class index to name.
    classes, class_names = load_classes(ilsvrc_dir)
    # Load supervision for each class.
    print 'load set of images for each class'
    class_images = {}
    bar = progressbar.ProgressBar()
    for i, name in enumerate(bar(classes)):
        class_images[name] = _load_image_set_train(ilsvrc_dir, i)
    # Need to get list of all images.
    # Since this requires going through images of each class,
    # we may as well construct map from image to class label.
    print 'invert index'
    image_labels = {}
    bar = progressbar.ProgressBar()
    for class_name in bar(classes):
        for example in class_images[class_name]:
            image = example['image']
            label = example['label']
            image_labels.setdefault(image, {})[class_name] = label
    print 'num images:', len(image_labels)
    images = sorted(image_labels.keys())
    # Load detailed annotation of all images.
    # Note that the annotation of an image which is not a positive example
    # for any class will be None.
    print 'load image annotations'
    image_annotations = {}
    bar = progressbar.ProgressBar()
    for i, im in enumerate(bar(images)):
        is_pos = any(map(lambda x: x == 1, image_labels[im].values()))
        image_annotations[im] = load_image_annot(ilsvrc_dir, 'train', im, must_exist=is_pos)
    return {
        'classes':           classes,
        'images':            images,
        'image_labels':      image_labels,
        'image_annotations': image_annotations,
    }


def _load_metadata_val(ilsvrc_dir):
    '''Loads metadata of validation set.

    metadata['classes'] -- List of class IDs.
    metadata['images'] -- List of image IDs.
    metadata['image_annotations'] -- Dictionary that maps image ID to annotation.
    '''
    # Create lookup from class index to name.
    classes, class_names = load_classes(ilsvrc_dir)
    # Load set of images with labels.
    images = _load_image_set_val(ilsvrc_dir)
    # Load detailed annotation of all images.
    print 'load image annotations'
    image_annotations = {}
    bar = progressbar.ProgressBar()
    for i, im in enumerate(bar(images)):
        image_annotations[im] = load_image_annot(ilsvrc_dir, 'val', im, must_exist=True)
    return {
        'classes':           classes,
        'images':            images,
        'image_annotations': image_annotations,
    }


def _labels_from_annotations(classes, images, image_annotations):
    # There is no such thing as a partial positive for the val set.
    # Set every image label to either +1 or -1.
    image_labels = {}
    for im in images:
        num = count_instances(image_annotations[im])
        image_labels[im] = {
            cls: 1 if num.get(cls, 0) > 0 else -1
            for cls in classes
        }
    return image_labels


def _load_image_set_train(ilsvrc_dir, class_index):
    '''Returns a list of dictionaries.'''
    set_dir = os.path.join(ilsvrc_dir, 'ImageSets', 'DET')
    fname = 'train_{:d}.txt'.format(class_index+1)
    fieldnames = ['image', 'label']
    types = {'image': str, 'label': int}
    with open(os.path.join(set_dir, fname), 'r') as f:
        rows = list(csv.DictReader(f, delimiter=' ', fieldnames=fieldnames))
    rows = map(functools.partial(_map_dict, fns=types), rows)
    return rows


def _load_image_set_val(ilsvrc_dir):
    '''Returns a list of strings.'''
    set_dir = os.path.join(ilsvrc_dir, 'ImageSets', 'DET')
    fname = 'val.txt'
    with open(os.path.join(set_dir, fname), 'r') as f:
        rows = list(csv.reader(f, delimiter=' '))
    return [im for im, ind in rows]


def load_image_annot(ilsvrc_dir, set_name, im, must_exist=True):
    fname = os.path.join(ilsvrc_dir, 'Annotations', 'DET', set_name, im+'.xml')
    if not must_exist and not os.path.exists(fname):
        return None
    with open(fname, 'r') as f:
        doc = xmltodict.parse(f.read(), force_list={'object'})
    return doc['annotation']


def count_instances(annotation):
    num = {}
    for obj in annotation.get('object', []):
        num[obj['name']] = num.get(obj['name'], 0) + 1
    return num


def _map_dict(x, fns):
    return {
        k: fns[k](v) if k in fns else v
        for k, v in x.iteritems()
    }
