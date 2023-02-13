namespace NNSandbox.NN
{
    internal static class NeuralNetworkHelpers
    {
        public static float[][] CreateBiasesArray(int[] structure, Func<float> factory) =>
            structure.Select(n => Helpers.CreateArray(n, factory))
                     .ToArray();

        public static float[][][] CreateWeightsArray(int[] structure, Func<float> factory) =>
            structure.Take(structure.Length - 1)
                     .Select((n, idx) => Helpers.CreateArray(n, () => Helpers.CreateArray(structure[idx + 1], factory)))
                     .ToArray();

        public static float[][] CreateValuesArray(int[] structure, Func<float> factory) =>
            structure.Select(n => Helpers.CreateArray(n, factory))
                     .ToArray();

        public static float[][] CloneValuesArray(float[][] values) =>
            values.Select(l => l.Select(v => v).ToArray())
                  .ToArray();
    }
}
