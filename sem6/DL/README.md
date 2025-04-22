# Deep Learning Assignments: Theory Explained (Simplified)

This document provides a simple explanation of the core concepts behind each deep learning assignment.

---

## Assignment 1: Feedforward Neural Network (FNN)

*   **What is it?** The simplest type of artificial neural network. Information flows in one direction: from input, through hidden layers (if any), to the output. Think of it like a chain reaction.
*   **Goal:** Used for basic tasks like classifying data (e.g., identifying images) or predicting values (e.g., predicting house prices).
*   **Key Components:**
    *   **Input Layer:** Receives the initial data (features).
    *   **Hidden Layers:** Perform calculations. The "deep" in deep learning comes from having multiple hidden layers.
    *   **Output Layer:** Produces the final result (prediction or classification).
    *   **Neurons (Nodes):** Units within layers that perform calculations.
    *   **Weights & Biases:** Numbers the network learns during training. They determine the strength of connections between neurons.
    *   **Activation Function:** A function (like ReLU, Sigmoid, Tanh, Softmax) applied by neurons to introduce non-linearity, allowing the network to learn complex patterns.
    *   **Loss Function:** Measures how wrong the network's prediction is compared to the actual answer (e.g., MSE for regression, Cross-Entropy for classification).
    *   **Optimizer:** Algorithm (like Adam or SGD) that adjusts weights and biases to minimize the loss function (i.e., helps the network learn).
*   **How it Works (Simplified):**
    1.  **Forward Propagation:** Data goes through the network, layer by layer, calculations are performed, and an output is produced.
    2.  **Loss Calculation:** The output is compared to the true answer using the loss function.
    3.  **Backward Propagation (Backprop):** The error (loss) is sent backward through the network.
    4.  **Weight Update:** The optimizer uses the backward-propagated error to adjust the weights and biases slightly to make the network better next time.
*   **Challenge:** **Overfitting:** When the model learns the training data *too* well, including noise, and performs poorly on new, unseen data. Techniques like **Dropout** (randomly ignoring some neurons during training) help prevent this.

---

## Assignment 2: Multiclass Classification (OCR)

*   **What is it?** A classification task where the goal is to assign an input to one of **more than two** possible categories. Example: Recognizing letters A-Z (26 categories).
*   **Dataset:** OCR Letter Recognition - Uses features extracted from images of letters (like pixel counts, edges) to classify them.
*   **Key Concepts (Neural Network Specific):**
    *   **Softmax Activation:** Used in the *output layer*. It converts the network's raw scores into probabilities for each class, ensuring they all add up to 1. The class with the highest probability is the prediction.
    *   **Categorical Cross-Entropy Loss:** The loss function specifically designed for multiclass classification when labels are represented as one-hot vectors.
    *   **One-Hot Encoding:** A way to represent categorical labels. For 26 letters, 'A' might be `[1, 0, 0, ..., 0]`, 'B' might be `[0, 1, 0, ..., 0]`, etc. The network predicts a similar vector of probabilities.
*   **Evaluation:** Accuracy is common, but a **Confusion Matrix** helps see *which* classes are being confused with others.

---

## Assignment 3: Binary Classification (IMDB)

*   **What is it?** A classification task where the goal is to assign an input to one of **exactly two** possible categories. Example: Classifying movie reviews as 'Positive' or 'Negative'.
*   **Dataset:** IMDB Movie Reviews - Contains text reviews labeled as positive (1) or negative (0).
*   **Key Concepts (Text Processing):**
    *   **Tokenization:** Breaking text down into individual words or sub-words (tokens).
    *   **Padding/Truncation:** Making all text sequences the same length by adding padding tokens or cutting off long sequences. Neural networks usually need fixed-size inputs.
    *   **Feature Extraction:** Converting text into numbers the network can understand.
        *   **Bag-of-Words (BoW):** Simple method counting word occurrences (ignores order).
        *   **Word Embeddings:** Representing words as dense vectors (lists of numbers). Words with similar meanings get similar vectors. This captures relationships between words much better than BoW.
*   **Key Concepts (Neural Network Specific):**
    *   **Embedding Layer:** Often the first layer in text models. It learns the word embeddings (vectors) during training or uses pre-trained ones (like Word2Vec, GloVe).
    *   **Sigmoid Activation:** Used in the *output layer* for binary classification. It squashes the output to a probability between 0 and 1 (e.g., > 0.5 means Positive, <= 0.5 means Negative).
    *   **Binary Cross-Entropy Loss:** The loss function specifically designed for binary classification.

---

## Assignment 4: Digit Recognition (CNN on MNIST)

*   **What is it?** A classic image classification task: identifying handwritten digits (0-9) from small grayscale images.
*   **Dataset:** MNIST - A large collection of 28x28 pixel grayscale images of digits.
*   **Why CNNs (Convolutional Neural Networks)?** CNNs are specifically designed for grid-like data, especially images. They are excellent at recognizing spatial hierarchies (simple patterns combining into complex ones).
*   **Key CNN Components:**
    *   **Convolutional Layer:** Uses filters (kernels) that slide over the image to detect specific features (like edges, corners, textures). Different filters learn to detect different features. Generates "feature maps".
    *   **Pooling Layer (e.g., MaxPooling):** Reduces the size of the feature maps, making the network faster and more robust to small variations in the image. It takes the maximum value from a small window.
    *   **Flatten Layer:** Converts the 2D feature maps into a 1D vector so they can be fed into standard Dense layers.
    *   **Fully Connected (Dense) Layer:** Standard neural network layers used at the end for classification based on the learned features.
*   **Activation:** ReLU (Rectified Linear Unit) is very commonly used in CNN hidden layers.
*   **Output:** Softmax activation (for 10 digit classes).

---

## Assignment 5: Image Classification (CNN on CIFAR-10)

*   **What is it?** Classifying small (32x32 pixel) color images into 10 categories (airplane, car, bird, etc.). More challenging than MNIST.
*   **Dataset:** CIFAR-10 - Contains 60,000 color images across 10 classes.
*   **Key Concepts:**
    *   Uses the same CNN principles as Assignment 4 (Convolutional, Pooling, Dense layers).
    *   **Color Images:** Input has 3 channels (Red, Green, Blue) instead of 1 (grayscale). CNNs handle this naturally.
    *   **Deeper Networks:** Often required for more complex datasets like CIFAR-10 compared to MNIST. More layers can learn more complex features.
    *   **Parameter Sharing:** A key benefit of Convolutional layers. The *same* filter is used across the entire image, drastically reducing the number of parameters to learn compared to using only Dense layers.
*   **Challenges:** Lower image resolution and greater real-world variation make it harder than MNIST.

---

## Assignment 6: RNN-Based Sentiment Analysis

*   **What is it?** Analyzing text (like reviews) to determine the sentiment (Positive, Negative, Neutral) using Recurrent Neural Networks.
*   **Why RNNs?** Text is sequential data (word order matters). RNNs are designed to process sequences by maintaining a "memory" or **hidden state** that captures information from previous steps.
*   **Basic RNN Problem:** **Vanishing Gradients**. When sequences are long, it becomes difficult for basic RNNs to learn dependencies between distant words (the error signal gets too small during backpropagation).
*   **Solutions: LSTM & GRU:**
    *   **LSTM (Long Short-Term Memory):** A special type of RNN unit with internal "gates" (forget gate, input gate, output gate) that control what information to keep, what to discard, and what to output. Much better at learning long-range dependencies.
    *   **GRU (Gated Recurrent Unit):** A simpler variant of LSTM with fewer gates (update gate, reset gate). Often performs similarly to LSTM but is computationally slightly faster.
*   **Typical Model Structure:** Input Text -> **Embedding Layer** -> **LSTM or GRU Layer(s)** -> **Dense Layer(s)** -> **Output Layer (Sigmoid for binary, Softmax for multi-class)**.
*   **Advanced Techniques:**
    *   **Bidirectional RNNs:** Process the sequence both forwards and backwards, capturing context from both directions.
    *   **Attention Mechanisms:** Allow the model to dynamically focus on the most relevant parts of the input sequence when making a prediction.

---

## Assignment 7: Transfer Learning

*   **What is it?** Reusing a model that was already trained on a large dataset (like ImageNet, with millions of images and 1000s of classes) for a new, different, but related task (e.g., classifying medical images or specific types of objects).
*   **Why Use It?**
    *   Saves significant training time and computational resources.
    *   Requires less labeled data for the *new* task.
    *   Often achieves better performance than training from scratch, especially with limited data.
*   **How It Works:** Models trained on large datasets learn general features (edges, textures, shapes). These features are often useful for other tasks.
*   **Two Main Approaches:**
    1.  **Feature Extraction:**
        *   Load the pre-trained model (e.g., VGG16, ResNet) without its final classification layer (`include_top=False`).
        *   **Freeze** the weights of the pre-trained layers (so they don't change).
        *   Add your *own* new classification layers on top.
        *   Train *only* the new layers on your specific dataset.
        *   *Best for:* Small datasets or when the new task is very similar to the original task.
    2.  **Fine-Tuning:**
        *   Start like feature extraction (load pre-trained model, add new layers).
        *   Initially train *only* the new layers (like feature extraction).
        *   Then, **unfreeze** *some* of the top layers of the pre-trained model.
        *   Continue training the *entire* model (unfrozen pre-trained layers + new layers) with a **very low learning rate**.
        *   *Best for:* Larger datasets or when you want the pre-trained features to adapt more to the specifics of your new data.
*   **Important:** You usually need to preprocess your input data (e.g., image size, pixel scaling) exactly the way the original pre-trained model expects.

---

## Assignment 8: Time Series Forecasting (RNNs)

*   **What is it?** Predicting future values in a sequence based on historical data points observed over time. Examples: forecasting weather, stock prices, energy demand.
*   **Why RNNs/LSTM/GRU?** Time series data is sequential, and these models are good at capturing temporal dependencies (how past values influence future values). LSTMs/GRUs handle potential long-range dependencies better than basic RNNs.
*   **Typical Process:**
    1.  **Load Data:** Get the time series (e.g., temperature readings over time).
    2.  **Preprocess:**
        *   **Normalize/Scale:** Scale data usually to a range like [0, 1] or [-1, 1]. This helps the network train more stably.
        *   **Create Sequences:** Convert the flat time series into input-output pairs. Example: Use values from time `t-10` to `t-1` (input sequence) to predict the value at time `t` (output).
    3.  **Build Model:** Use LSTM or GRU layers. The input shape will be `(batch_size, sequence_length, num_features)`. The output layer typically has one neuron for predicting the next single value.
    4.  **Compile:** Use a regression loss function like **Mean Squared Error (MSE)**.
    5.  **Train:** Fit the model on the training sequences.
    6.  **Evaluate:** Make predictions on the test set and calculate metrics like **Root Mean Squared Error (RMSE)**. Remember to inverse the scaling to interpret the error in the original units.
    7.  **Visualize:** Plot the predicted values against the actual values.

---

## Assignment 9: Autoencoders

*   **What is it?** An **unsupervised** neural network trained to reconstruct its input. It learns efficient representations (encodings) of data without needing explicit labels.
*   **Structure:** Consists of two parts:
    *   **Encoder:** Maps the input data to a lower-dimensional **latent space** (bottleneck). This is the compressed representation.
    *   **Decoder:** Tries to reconstruct the original input data from the compressed latent space representation.
*   **Goal:** To minimize the **reconstruction error** (the difference between the original input and the reconstructed output).
*   **Applications:**
    *   **Data Compression:** The encoder compresses the data, and the decoder decompresses it (usually lossy compression).
    *   **Dimensionality Reduction:** The latent space provides a lower-dimensional view of the data, useful for visualization (like PCA, but can learn non-linear relationships).
    *   **Image Denoising:** Train the autoencoder by feeding it **noisy** images as input and making it reconstruct the original **clean** images. It learns to ignore the noise.
    *   **Feature Learning:** The encoder learns meaningful features from the data in an unsupervised way.
*   **Types:**
    *   **Basic:** Uses Dense (fully connected) layers.
    *   **Convolutional (CAE):** Uses Convolutional/Pooling layers in the encoder and Transposed Convolutional layers in the decoder (good for images).
    *   **Denoising (DAE):** Specifically trained for noise removal.
    *   **Variational (VAE):** A more advanced, generative type that learns a probability distribution in the latent space, allowing generation of new data.

---

## Assignment 10: Generative Adversarial Networks (GANs)

*   **What is it?** A type of **generative** model that learns to create *new* data samples that resemble the training data (e.g., creating realistic-looking faces or handwritten digits).
*   **Structure (The "Adversarial" Game):** Consists of two networks trained simultaneously in opposition:
    *   **Generator (G):** Takes random noise as input and tries to generate fake data samples that look real. Think of it as a counterfeiter.
    *   **Discriminator (D):** Takes both real data samples (from the training set) and fake samples (from the Generator) and tries to classify them correctly as "real" or "fake". Think of it as a police detective.
*   **How Training Works:**
    1.  **Train Discriminator:** Show D a mix of real images (labeled real) and fake images from G (labeled fake). D learns to get better at telling them apart.
    2.  **Train Generator:** Generate fake images with G. Feed them to D. Tell G to adjust its weights to make D classify these fake images as "real". G learns by getting feedback *through* the (temporarily frozen) Discriminator.
*   **Goal:** Reach an **equilibrium** where the Generator produces fakes so realistic that the Discriminator can only guess randomly (50% accuracy).
*   **Challenges:**
    *   **Training Instability:** Can be hard to balance the training of G and D. One might overpower the other.
    *   **Mode Collapse:** The Generator might learn to produce only a few types of realistic outputs, failing to capture the full diversity of the training data.
*   **Types:** Basic (Dense layers), DCGAN (uses Convolutional layers, more stable for images), WGAN, Conditional GAN (cGAN - generate specific types of data based on a label).

---

## Assignment 11: Q-Learning

*   **What is it?** A fundamental **Reinforcement Learning (RL)** algorithm. It's used to teach an **agent** how to make optimal decisions in an **environment** to maximize a cumulative **reward**. It's **model-free**, meaning the agent learns directly from experience without needing a perfect understanding of the environment's rules.
*   **Key Concepts:**
    *   **Agent:** The learner/decision-maker (e.g., a game character).
    *   **Environment:** The world the agent interacts with (e.g., the game level, a maze).
    *   **State (s):** The current situation or configuration of the environment (e.g., agent's position).
    *   **Action (a):** A possible move the agent can take in a state (e.g., move left, right, up, down).
    *   **Reward (R):** Numerical feedback from the environment after taking an action (e.g., +1 for reaching goal, -1 for falling in hole, 0 otherwise).
    *   **Policy (π):** The agent's strategy for choosing actions in different states. Q-learning aims to find the optimal policy.
    *   **Q-Value (Q(s, a)):** The **expected future cumulative reward** the agent can get by taking action `a` in state `s`, and then following the optimal policy thereafter. It represents the "quality" of taking that action in that state.
    *   **Q-Table:** A table storing the Q-values for all possible state-action pairs.
*   **How Q-Learning Works (Simplified):**
    1.  **Initialize:** Create the Q-table, usually with all zeros.
    2.  **Loop (Episodes):** For many episodes (attempts from start to finish):
        a.  Start in the initial state `s`.
        b.  **Choose Action:** Select an action `a`. Use **epsilon-greedy** strategy:
            *   With probability `epsilon` (exploration rate): choose a random action (Explore).
            *   With probability `1-epsilon`: choose the action with the highest Q-value for state `s` from the Q-table (Exploit).
        c.  **Take Action:** Perform action `a` in the environment.
        d.  **Observe:** Get the reward `R` and the new state `s'`.
        e.  **Update Q-Table:** Adjust the Q-value for the *previous* state-action pair `(s, a)` using the observed reward and the *estimated best future value* from the new state `s'`:
            `New Q(s, a) = Old Q(s, a) + learning_rate * [Reward + discount_factor * max(Q(s', all_actions)) - Old Q(s, a)]`
            *   `learning_rate` (alpha): Controls how much the new information overrides old information.
            *   `discount_factor` (gamma): Makes future rewards less valuable than immediate rewards (0 <= gamma < 1).
        f.  Update the current state: `s = s'`.
        g.  Repeat b-f until the episode ends (e.g., goal reached, agent falls in hole).
    3.  **Decay Epsilon:** Gradually decrease `epsilon` over episodes, shifting from exploration to exploitation as the agent learns more.
*   **Goal:** After enough episodes, the Q-table converges, and the agent can act optimally by always choosing the action with the highest Q-value in its current state.
*   **Challenge:** The Q-table becomes impractically large for environments with many states (**curse of dimensionality**). This leads to **Deep Q-Learning (DQN)**, which uses a neural network to approximate Q-values instead of a table.

---