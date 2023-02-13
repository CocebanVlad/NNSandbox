using Newtonsoft.Json;

namespace NNSandbox.NN
{
    internal class NeuralNetwork
    {
        [JsonProperty("struct")]
        public int[] Structure { get; set; }

        [JsonProperty("biases")]
        public float[][] Biases { get; set; }

        [JsonProperty("weights")]
        public float[][][] Weights { get; set; }

        [JsonProperty("activation", TypeNameHandling = TypeNameHandling.All)]
        public ActivationFunction ActivationFunction { get; set; }

        [JsonProperty("training", TypeNameHandling = TypeNameHandling.All)]
        public TrainingAlgorithm TrainingAlgorithm { get; set; }

        public void Initialize(int[] structure)
        {
            Structure = structure;

            var r = new Random();

            // create a randomized biases array
            // with values within the range of -0.5 to 0.5
            var biasesArray =
                NeuralNetworkHelpers.CreateBiasesArray(
                    Structure,
                    () => Helpers.GetRandomFloat(r, -0.5f, 0.5f)
                    );

            Biases = biasesArray;

            // create a randomized weights array
            // with values within the range of -0.5 to 0.5
            var weightsArray =
                NeuralNetworkHelpers.CreateWeightsArray(
                    Structure,
                    () => Helpers.GetRandomFloat(r, -0.5f, 0.5f)
                    );

            Weights = weightsArray;
        }

        public float[][] Test(float[] inputs)
        {
            var values = NeuralNetworkHelpers.CreateValuesArray(Structure, () => 0f);

            if (/* expected */ values[0].Length !=
                /* provided */ inputs.Length)
                throw new ArgumentException(
                    $"Provided inputs do not fit the input layer (expected: {values[0].Length}; provided: {inputs.Length})");

            // set inputs on the input layer
            for (var n = 0; n < inputs.Length; n++) values[0][n] = inputs[n];

            // starting from the second layer
            for (var l = 1; l < values.Length; l++)
            {
                // foreach neuron
                for (var n = 0; n < values[l].Length; n++)
                {
                    // 1. calculate the sum of all values weighted (from the previous layer)
                    var weightedValuesSum =
                        values[l - 1].Select((v, p) => v * Weights[l - 1][p][n])
                                     .Sum();

                    // 2. calculate current neuron value
                    var value = weightedValuesSum + Biases[l][n];

                    // 3. apply the activation function, if present
                    if (ActivationFunction != null)
                        value = ActivationFunction.Calculate(value);

                    // 4. set value for the current neuron
                    values[l][n] = value;
                }
            }

            return values;
        }

        public void Train(TrainingDataSet dataSet) => TrainingAlgorithm?.Train(this, dataSet);

        public async Task SaveAsync(string fileName)
        {
            using var file = File.Open(fileName, FileMode.Create, FileAccess.Write);
            using var writer = new StreamWriter(file);
            var json = JsonConvert.SerializeObject(this, Formatting.None);
            await writer.WriteAsync(json);
        }

        public static async Task<T> LoadAsync<T>(string fileName)
            where T : NeuralNetwork
        {
            using var file = File.Open(fileName, FileMode.Open, FileAccess.Read);
            using var reader = new StreamReader(file);
            var json = await reader.ReadToEndAsync();
            return JsonConvert.DeserializeObject<T>(json);
        }
    }
}
