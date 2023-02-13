using Newtonsoft.Json;

namespace NNSandbox.NN.TrainingAlgorithms
{
    internal class BacktrackingTrainingAlgorithm : TrainingAlgorithm
    {
        [JsonProperty("rate")]
        public float LearningRate { get; set; } = 0.8f;

        [JsonProperty("decay")]
        public float WeightDecay { get; set; } = 0.001f;

        public override void Train(NeuralNetwork network, TrainingDataSet dataSet)
        {
            var biasesSmudge =
                NeuralNetworkHelpers.CreateBiasesArray(
                    network.Structure, () => 0f);

            var weightsSmudge =
                NeuralNetworkHelpers.CreateWeightsArray(
                    network.Structure, () => 0f);

            // cumulate smudges
            for (var i = 0; i < dataSet.Count; i++)
            {
                var entry = dataSet[i];

                // test network
                var values = network.Test(entry.Inputs);

                if (/* expected */ values[^1].Length !=
                    /* provided */ entry.ExpectedOutputs.Length)
                    throw new ArgumentException(
                        $"Provided expected outputs do not fit the output layer (expected: {values[0].Length}; provided: {entry.ExpectedOutputs.Length})");

                var expectedValues = NeuralNetworkHelpers.CloneValuesArray(values);

                // set expected outputs on the output layer
                for (var n = 0; n < entry.ExpectedOutputs.Length; n++) expectedValues[^1][n] = entry.ExpectedOutputs[n];

                // starting from the last layer
                for (var l = values.Length - 1; l >= 1; l--)
                {
                    // foreach neuron
                    for (var n = 0; n < values[l].Length; n++)
                    {
                        // 1. calculate bias smudge
                        var biasSmudge = expectedValues[l][n] - values[l][n];

                        // 2. multiply the bias smudge by the value derivative, if present
                        if (network.ActivationFunction != null)
                            biasSmudge *= network.ActivationFunction.CalculateDerivative(values[l][n]);

                        // 3. collect the bias smudge for the current entry
                        biasesSmudge[l][n] += biasSmudge;

                        // foreach neuron on the previous layer
                        for (var p = 0; p < values[l - 1].Length; p++)
                        {
                            // 4. calculate the weight smudge
                            var weightSmudge = values[l - 1][p] * biasSmudge;

                            // 5. collect the weight smudge for the current entry
                            weightsSmudge[l - 1][p][n] += weightSmudge;

                            // 6. calculate the value smudge
                            var valueSmudge = network.Weights[l - 1][p][n] * biasSmudge;

                            // 7. adjust expected value
                            expectedValues[l - 1][p] += valueSmudge;
                        }
                    }
                }
            }

            // apply changes to the network
            for (var l = network.Structure.Length - 1; l >= 1; l--)
            {
                for (var n = 0; n < network.Structure[l]; n++)
                {
                    network.Biases[l][n] += biasesSmudge[l][n] * LearningRate;
                    network.Biases[l][n] *= 1 - WeightDecay;

                    for (var p = 0; p < network.Structure[l - 1]; p++)
                    {
                        network.Weights[l - 1][p][n] += weightsSmudge[l - 1][p][n] * LearningRate;
                        network.Weights[l - 1][p][n] *= 1 - WeightDecay;
                    }
                }
            }
        }
    }
}
