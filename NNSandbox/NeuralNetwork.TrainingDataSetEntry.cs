namespace NNSandbox;

internal partial class NeuralNetwork
{
    public record TrainingDataSetEntry
    {
        public TrainingDataSetEntry(float[] inputs, float[] expectedOutputs)
        {
            Inputs = inputs;
            ExpectedOutputs = expectedOutputs;
        }

        public float[] Inputs { get; }
        public float[] ExpectedOutputs { get; }
    }
}
