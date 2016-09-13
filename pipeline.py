import numpy as np
import os
import cPickle

class Pipeline(object):
    """
    A Pipeline is an object representing the data transformations to make
    on the input data, finally outputting extracted features.
    gen_ictal: Whether ictal data generation should be used for this pipeline
    pipeline: List of transforms to apply one by one to the input data
    """
    def __init__(self, pipeline):
        self.transforms = pipeline
        names = [t.get_name() for t in self.transforms]
        self.name = 'empty' if len(names) == 0 else '_'.join(names)

    def get_name(self):
        return self.name

    def apply(self, data_gen):
        '''apply transforms to each item in generator'''
        features = []
        for data in data_gen:
            for transform in self.transforms:
                data = transform.apply(data)
            features.append(data)
        return np.array(features)

    def to_file(self, X, files, X_name, y=None, dest_dir='feature_vectors'):
        X = self.apply(X)
        fname = "%s_%s.pkl" % (X_name, self.name)
        outpath = os.path.join(dest_dir, fname)
        if y is None:
            data = [X, files]
        else:
            data = [X, y, files]
        with open(outpath, 'wb') as f:
            cPickle.dump(data, f)
