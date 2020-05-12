import argparse 
import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    # We use our "load_graph" function
    graph = load_graph(args.frozen_model_filename)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        #pass
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    x         = graph.get_tensor_by_name('prefix/input/x:0')
    keep_prob = graph.get_tensor_by_name('prefix/input/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/prediction_mask:0')

    print (x.shape)
    print (x.dtype)
    print (keep_prob.shape)
    print (keep_prob.dtype)
    print (y.shape)
    print (y.dtype)

    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 

        xin    = np.arange(120*257, dtype=np.float32).reshape((1,257,120))
        y_out  = sess.run(y, feed_dict = {x: xin, keep_prob: 1})
        print (y_out.shape)
        print (y_out.dtype)

        #y_out=  y_out.reshape((1,257,120,2))

        for i in range(120):
            for j in range(257):
                print ('i = ',i,' j = ',j, ' out = ',  y_out[0,j,i,0])

        #print (y_out.shape)
        #print (y_out.dtype)
        #np.save('mi_out_free', y_out)
