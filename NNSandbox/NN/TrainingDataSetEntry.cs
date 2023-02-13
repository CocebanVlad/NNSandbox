using Newtonsoft.Json;

namespace NNSandbox.NN
{
    internal class TrainingDataSetEntry
    {
        [JsonProperty("inputs")]
        public float[] Inputs { get; set; }

        [JsonProperty("expectedOutputs")]
        public float[] ExpectedOutputs { get; set; }
    }
}
