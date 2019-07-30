import tf from '@tensorflow/tfjs';

// Optional Load the binding:
// Use '@tensorflow/tfjs-node-gpu' if running with GPU.
import '@tensorflow/tfjs-node';

export default function main() {

    // Train a simple model:
    const model = tf.sequential();

    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [10] }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    const xs = tf.randomNormal([100, 10]);
    const ys = tf.randomNormal([100, 1]);

    model.fit(xs, ys, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        }
    });

    // Saving the model
    model.save('file://./model/segment');

}
