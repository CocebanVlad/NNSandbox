namespace NNSandbox.NN
{
    internal abstract class TrainingAlgorithm
    {
        public abstract void Train(NeuralNetwork network, TrainingDataSet dataSet);
    }
}
