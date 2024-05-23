# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the trained model once when the application starts
# model = None
# model_path = './rfc_model.pkl'
# try:
#     with open(model_path, 'rb') as file:
#         model = pickle.load(file)
# except FileNotFoundError:
#     print(f"Error: Model file not found at {model_path}")
# except EOFError:
#     print(f"Error: Model file is corrupted at {model_path}")

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     features = [
#         data['main_temp'],
#         data['visibility'],
#         data['wind_speed'],
#         data['pressure'],
#         data['humidity'],
#         966,  # grnd_level
#         1014  # sea_level
#     ]
#     prediction = model.predict(np.array(features).reshape(1, -1))[0]
#     return jsonify({'prediction': int(prediction)})

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)  # Set the port to 5000




# import numpy as np
# import pandas as pd
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the trained model
# model_path = './trainedmodels/keras/binary_model.keras'
# model = load_model(model_path)

# # Define the sequence length and the features used for prediction
# sequence_length = 50
# sequence_cols = ['id','cycle','setting1', 'setting2', 'setting3'] + ['s' + str(i) for i in range(1, 22)]
# print("abcd python1")
# @app.route('/predict', methods=['POST'])
# def predict():

#         print("abcd python2")
#         # Get data from request
#         data = request.get_json()
#         print(data)
#         print("abcd python3")
#         # Convert data to DataFrame
#         data_df = pd.DataFrame(data)
#         print("abcd python4")
        
#         # Normalize the cycle column
#         data_df['cycle'] = data_df['cycle']
#         print("abcd python5")
#         # Ensure the data is in the correct order of columns
#         data_df = data_df[sequence_cols]
#         print("abcd python6")
        
#         # As we have only one data point, we will create a dummy sequence of the required length
#         # Here we simply replicate the data point 50 times to form a sequence
#         data_sequence = np.tile(data_df.values, (sequence_length, 1)).reshape(1, sequence_length, len(sequence_cols))
#         print("abcd python7")
        
#         # Make a prediction
#         prediction = (model.predict(data_sequence) > 0.5).astype("int32")
#         print("abcd python8")
        
#         # Output the prediction
#         return jsonify({'prediction': prediction[0][0]})

    
# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn import preprocessing

app = Flask(__name__)

# Load the trained model
model_path = './trained models/keras/RNN_fwd.keras'
model = load_model(model_path)

# Define column names
cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5',
              's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    # Create a DataFrame with the specific data point
    data_point_df = pd.DataFrame(data, columns=cols_names)
    # Normalize the 's2' feature
    data_point_df['cycle_norm'] = data_point_df['cycle']
    cols_normalize = ['s2']  # Only normalize the 's2' feature
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(data_point_df[cols_normalize])
    norm_data_point_df = pd.DataFrame(min_max_scaler.transform(data_point_df[cols_normalize]), columns=cols_normalize, index=data_point_df.index)
    data_point_join_df = data_point_df[['id', 'cycle']].join(norm_data_point_df)
    data_point_df = data_point_join_df.reindex(columns=data_point_df.columns)
    # Replicate the row to create a sequence
    sequence_length = 50
    seq_cols = ['s2']  # Only use the 's2' feature
    replicated_df = pd.concat([data_point_df] * sequence_length, ignore_index=True)
    # Generate sequences
    def new_sequence_generator(feature_df, seq_length, seq_cols):
        feature_array = feature_df[seq_cols].values
        num_elements = feature_array.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
            yield feature_array[start:stop, :]
    new_seq_gen = list(new_sequence_generator(replicated_df, sequence_length, seq_cols))
    new_seq_set = np.array(new_seq_gen).astype(np.float32)
    # Make prediction
    new_predictions = model.predict(new_seq_set)
    # Interpret the predictions
    threshold = 0.5
    predicted_labels = (new_predictions > threshold).astype(int)
    # Output the prediction
    return jsonify({'prediction': predicted_labels.tolist()})


if __name__ == '__main__':
    app.run(port=5000, debug=True)
