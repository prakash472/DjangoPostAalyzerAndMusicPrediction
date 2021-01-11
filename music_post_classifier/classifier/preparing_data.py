import json
import os
import math
import librosa

DATASET_PATH = "../Data/genres_original"
JSON_PATH = "music_data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

"""
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data={
        "genres":[],
        "mfcc":[],
        "label":[]
    }

    samples_per_segment=int(SAMPLES_PER_TRACK/num_segments)
    num_mfcc_vectors_per_segment=math.ceil(samples_per_segment/hop_length)

    for i,(dir_path,dir_name,file_name) in enumerate(os.walk(DATASET_PATH)):
            adjusted_path= dir_path.replace(os.sep,"/")
            print("adjusted_path",adjusted_path)
            if adjusted_path is not DATASET_PATH:
                semantic_genre=adjusted_path.split("/")[-1]
                data["genres"].append(semantic_genre)
                print("\n Processing {} {}".format(i,semantic_genre))
                for f in file_name:
                    file_path= os.path.join(dir_path,f)
                    adjusted_file_path=file_path.replace(os.sep,"/")
                    signal,sample_rate=librosa.load(adjusted_file_path,sr=SAMPLE_RATE)
                    
                    for segment in range(num_segments):
                        start=samples_per_segment*segment
                        finish=start+samples_per_segment

                        mfcc=librosa.feature.mfcc(signal[start:finish],sample_rate,n_mfcc=13,n_fft=2048,hop_length=hop_length)
                        mfcc=mfcc.T

                        if len(mfcc)==num_mfcc_vectors_per_segment:
                            data["label"].append(i-1)
                            data["mfcc"].append(mfcc.tolist())
                            print("{},segment:{}".format(adjusted_file_path,segment+1))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
"""

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data={
        "genres":[],
        "mfcc":[],
        "label":[]
    }

    samples_per_segment=int(SAMPLES_PER_TRACK/num_segments)
    num_mfcc_vectors_per_segment=math.ceil(samples_per_segment/hop_length)

    #signal,sample_rate=librosa.load("Data/genres_original/jazz/jazz.00052.wav")
    for i,(dir_path,dir_name,file_name) in enumerate(os.walk(dataset_path)):
       if dir_path is not dataset_path:
         semantic_genre=dir_path.split("/")[-1]
         print("\n Processing {} {}".format(i,semantic_genre))
         data["genres"].append(semantic_genre)
         for f in file_name:
           file_path=os.path.join(dir_path,f)
           signal,sample_rate=librosa.load(file_path,sr=SAMPLE_RATE)

           for segment in range(num_segments):
             start=samples_per_segment*segment
             finish=start+samples_per_segment

             mfcc=librosa.feature.mfcc(signal[start:finish],sample_rate,n_mfcc=13,n_fft=2048,hop_length=hop_length)
             mfcc=mfcc.T

             if len(mfcc)==num_mfcc_vectors_per_segment:
               data["label"].append(i-1)
               data["mfcc"].append(mfcc.tolist())
               print("{},segment:{}".format(file_path,segment+1))
        
    with open(json_path, "w") as fp:
      json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)