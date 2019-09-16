import tensorflow as tf
import numpy as np

class TfModel(object):

    def __init__(self, model_filepath):

        # The file path of model
        self.model_filepath = model_filepath
        # Initialize the model
        self.load_graph(model_filepath = self.model_filepath)

    def load_graph(self, model_filepath):
        '''
        Lode trained model.
        '''
        print('Loading tf model...')
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.InteractiveSession(graph = self.graph)

        with tf.io.gfile.GFile(model_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Define input tensor
        self.data = tf.compat.v1.placeholder(np.float32, shape = [None, 300, 300, 3], name='data')
        
        nodes = graph_def.node
        for node in nodes:
            if 'BatchNorm' in node.op:
                print(node)

        tf.import_graph_def(graph_def, {'data': self.data})

        print('Model loading complete!')

    def test(self, data):

        # Know your output node name
        conf_tensor = self.graph.get_tensor_by_name("import/data_bn/FusedBatchNorm:0")
        loc_tensor = self.graph.get_tensor_by_name("import/mbox_loc:0")
        output = self.sess.run([conf_tensor,loc_tensor], feed_dict = {self.data: data})

        return output