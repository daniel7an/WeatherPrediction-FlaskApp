from itertools import starmap
import os, sys
import json
import tensorflow as tf
import time

labels_path = '/home/daniielyan/Desktop/Computer Vision/WeatherPredictinApp/labels.txt'
graph_path = '/home/daniielyan/Desktop/Computer Vision/WeatherPredictinApp/retrained_graph.pb'



def predict(img_path):
    image_data = tf.io.gfile.GFile(img_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                    in tf.io.gfile.GFile(labels_path)]

    # Unpersists graph from file
    with tf.io.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
    with  tf.compat.v1.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        outputData = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            result = {
                "name": human_string,
                "score": "%.2f" % score
            }
            outputData.append(result) 

        
        #return json.dumps(outputData)

        return outputData[0]['name']


