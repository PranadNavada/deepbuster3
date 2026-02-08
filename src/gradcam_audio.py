import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

def compute_gradcam(model_path, spectrogram_path):
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    try:
        data = np.load(spectrogram_path, allow_pickle=True)
        spec = data["fake"][0] if "fake" in data else np.random.randn(96, 64)
    except:
        print("Warning: spectrograms.npz not found. Using dummy for visualization.")
        spec = np.random.randn(96, 64)
    
    s = np.array(spec)
    if s.ndim == 2:
        if s.shape[0] < s.shape[1] and s.shape[0] < 96: s = s.T
        s = s[:96, :64]
        if s.shape[0] < 96: s = np.pad(s, ((0, 96-s.shape[0]), (0, 0)), mode="constant")
        if s.shape[1] < 64: s = np.pad(s, ((0, 0), (0, 64-s.shape[1])), mode="constant")
        s = (s - np.mean(s)) / (np.std(s) + 1e-8)
    
    input_tensor = s[np.newaxis, ..., np.newaxis]
    
    last_conv_layer = model.get_layer("last_conv")
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(input_tensor)
        
        loss = preds[:, 0]
        
    grads = tape.gradient(loss, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_output = last_conv_output[0].numpy()
    heatmap = last_conv_output @ pooled_grads.numpy()[..., np.newaxis]
    heatmap = np.squeeze(np.maximum(heatmap, 0)) 
    heatmap /= (np.max(heatmap) + 1e-8)
    
    
    plt.figure(figsize=(10, 4))
    plt.imshow(s.T, aspect="auto", origin="lower", cmap="magma")
    
    
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (96, 64)).numpy()
    heatmap_resized = np.squeeze(heatmap_resized, axis=-1)
    plt.imshow(heatmap_resized.T, aspect="auto", origin="lower", cmap="jet", alpha=0.5)
    
    score = float(preds[0])
    label = "Fake" if score >= 0.5 else "Real"
    plt.title(f"Prediction: {label}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    
    plt.savefig("gradcam_audio_result.png")
    print("Result visualization saved as gradcam_audio_result.png")
    plt.show()

if __name__ == "__main__":
    m_path = sys.argv[1] if len(sys.argv) > 1 else "audio_detector_v2.h5" 
    s_path = sys.argv[2] if len(sys.argv) > 2 else "spectrograms.npz"
    compute_gradcam(m_path, s_path)
