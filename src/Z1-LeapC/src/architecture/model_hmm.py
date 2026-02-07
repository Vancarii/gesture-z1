import numpy as np
import os 
import traceback
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from hmmlearn.hmm import GMMHMM

#Use in conda environment with hmmlearn==0.28.0

GESTURE_SET = [
    "point up", "point down", "point forward", "point back",
    "point left", "point right",
    "move hand up", "move hand down",
    "pause", "clap",
    "swipe towards", "swipe back", 
]

STATIC_GESTURES = ["point_up", "point_down", "point_forward", "point_back",
                   "point_left", "point_right", "pause"]

DYNAMIC_GESTURES = ["move_hand_up", "move_hand_down", 
                    "swipe_back", "swipe_towards", "clap"]

os.makedirs("models", exist_ok=True)

def data_loader(labels):
    X_label = {}
    labelarr = []

    for label in labels:
        gesturename = label.replace(" ", "_")
        gesturedir = os.path.join("data", gesturename)

        samples = []
        for file in sorted(os.listdir(gesturedir)):
            if file.endswith(".npy"):
                path = os.path.join(gesturedir, file)
                sample = np.load(path)

                if sample.ndim == 2 and sample.shape[0] > 0 and sample.shape[1] > 0:
                    samples.append(sample)
                else:
                    print(f"Invalid sample found in {path}")
        
        if samples:
            X_label[gesturename] = samples
            labelarr.append(gesturename)
        
    return X_label, labelarr

def train_model():
    X_label, labelarr = data_loader(GESTURE_SET)

    if not labelarr:
        print("No data to train model")
        return
    
    groups = {}
    for label, samples in X_label.items():
        nfeatures = samples[0].shape[1]
        groups.setdefault(nfeatures, []).extend(samples)
    
    for label, samples in X_label.items():
        print(f"{label:20s} -> {samples[0].shape}")

    scalers = {}
    for nfeatures, samples in groups.items():
        flatten = np.vstack(samples)
        scaler = StandardScaler()
        scaler.fit(flatten)
        scalers[nfeatures] = scaler
        joblib.dump(scaler, os.path.join("models", f"scaler_{nfeatures}f.pkl"))

    #Scale data for training
    X_scaled = {}
    for label, samples in X_label.items():
        nfeatures = samples[0].shape[1]
        if nfeatures not in scalers:
            print(f"No scaler found for {label} with {nfeatures} features")
            continue
        scaler = scalers[nfeatures]
        X_scaled[label] = [scaler.transform(sample) for sample in samples]
        
    #Train HMM per gesture
    #Architecture: 3 hidden states, 3 guassian emissions per state, diagonal covariance matrix, 100 iterations
    models = {}
    staticfeatures = []
    staticlabels = []
    loglikelihoods = {}
    N_STATES_CLAP = 6
    N_MIXTURES = 1
    RANDOM_STATE = 42
    N_ITER = 100
    N_STATES_DYNAMIC = 3

    print("Labels in X_scaled:", list(X_scaled.keys()))

    for label, samples in X_scaled.items():
        print(f"Training {label} model")

        if label in DYNAMIC_GESTURES:
            # Check if we have enough samples
            if len(samples) < 2:
                print(f"Warning: {label} has only {len(samples)} sample(s). Need at least 2 samples for training.")
                continue
            
            X_concat = np.vstack(samples)
            lengths = [len(sample) for sample in samples]
            
            if np.any(np.isnan(X_concat)) or np.any(np.isinf(X_concat)):
                print(f"Warning: {label} data contains NaN or Inf values. Skipping.")
                continue
            
            feature_variance = np.var(X_concat, axis=0)
            if np.any(feature_variance < 1e-6):
                print(f"Warning: {label} has features with very low variance. This may cause training issues.")
            
            total_frames = sum(lengths)
            if total_frames < N_STATES_DYNAMIC * 2:
                print(f"Warning: {label} has only {total_frames} total frames. Need at least {N_STATES_DYNAMIC * 2} for {N_STATES_DYNAMIC} states.")
                continue

            if label == "clap": #Ergodic model
                model = GMMHMM(n_components = N_STATES_CLAP, n_mix = N_MIXTURES, covariance_type = "diag", n_iter = N_ITER, random_state = RANDOM_STATE, verbose = False, min_covar=0.001)
            else:
                #Top right topology
                startprob = np.zeros(N_STATES_DYNAMIC)
                startprob[0] = 1.0
                transmat = np.zeros((N_STATES_DYNAMIC, N_STATES_DYNAMIC))
                for i in range(N_STATES_DYNAMIC - 1):
                    transmat[i, i] = 0.5
                    transmat[i, i+1] = 0.5
                transmat[N_STATES_DYNAMIC - 1, N_STATES_DYNAMIC - 1] = 1.0

                model = GMMHMM(n_components = N_STATES_DYNAMIC, n_mix = N_MIXTURES, covariance_type = "spherical", n_iter = N_ITER, random_state = RANDOM_STATE, init_params="mcw", verbose = False, min_covar=0.001)
                print(f"Start probabilities: {startprob}")
                model.startprob_ = startprob
                model.transmat_ = transmat

            try:
                model.fit(X_concat, lengths)
                
                # Validate model after fitting
                if np.any(np.isnan(model.startprob_)) or np.any(np.isnan(model.transmat_)):
                    print(f"Warning: {label} model has NaN values after fitting. Model may not have converged.")
                    print(f"  Data stats: samples={len(samples)}, total_frames={total_frames}, feature_variance_range=[{feature_variance.min():.6f}, {feature_variance.max():.6f}]")
                    continue
                
                # Check for invalid probabilities
                if np.any(model.startprob_ < 0) or np.any(model.transmat_ < 0):
                    print(f"Warning: {label} model has negative probabilities. Skipping.")
                    continue
                
                if not np.isclose(model.startprob_.sum(), 1.0, atol=1e-6):
                    print(f"Warning: {label} model startprob_ does not sum to 1 (got {model.startprob_.sum()}). Re-normalizing.")
                    model.startprob_ = model.startprob_ / model.startprob_.sum()
                
                transmat_sums = model.transmat_.sum(axis=1)
                if not np.allclose(transmat_sums, 1.0, atol=1e-6):
                    print(f"Warning: {label} model transmat_ rows do not sum to 1. Re-normalizing.")
                    model.transmat_ = model.transmat_ / transmat_sums[:, np.newaxis]
                
                try:
                    avgloglikelihood = np.mean([model.score(sample) for sample in samples])
                    if np.isnan(avgloglikelihood) or np.isinf(avgloglikelihood):
                        print(f"Warning: {label} model produced invalid loglikelihood ({avgloglikelihood}). Skipping.")
                        continue
                except Exception as score_error:
                    print(f"Warning: {label} model failed to score samples: {score_error}. Skipping.")
                    continue
                
                models[label] = model
                loglikelihoods[label] = avgloglikelihood
                print(f"Trained {label} model with loglikelihood: {avgloglikelihood:.4f}")
            except Exception as e:
                print(f"Error training {label} model: {e}")
                traceback.print_exc()

        elif label in STATIC_GESTURES:
            for sample in samples:
                median = np.median(sample, axis=0)
                staticfeatures.append(median)
                staticlabels.append(label)
        else:
            print(f"Unknown gesture: {label}")
    
    if models:
        modelpath = os.path.join("models", "gesture_models.pkl")
        joblib.dump(models, modelpath)
        print(f"Trained {len(models)}/{len(DYNAMIC_GESTURES)} dynamic gesture models")
        print(f"Saved to: {modelpath}")
        print(f"Trained gestures: {list(models.keys())}")
        missing = set(DYNAMIC_GESTURES) - set(models.keys())
        if missing:
            print(f"Missing gestures: {list(missing)}")
    else:
        print("Failed to train any gesture models")
    
    if staticfeatures and staticlabels:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(staticfeatures, staticlabels)
        accuracy = knn.score(staticfeatures, staticlabels) * 100
        print(f"Trained static gesture KNN classifier")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Samples: {len(staticfeatures)}")
        knnpath = os.path.join("models", "static_gesture_knn.pkl")
        joblib.dump(knn, knnpath)
        print(f"Saved to: {knnpath}")
    else:
        print("No static features to train KNN model")

if __name__ == "__main__":
    train_model()
