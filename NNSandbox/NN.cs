namespace NNSandbox;

internal class NN
{
    private readonly int[] _structure;
    private readonly float[][] _biases;
    private readonly float[][][] _weights;

    public NN(int[] structure)
    {
        _structure = structure;
        _biases = CreateBiasesArray(_structure);
        _weights = CreateWeightsArray(_structure);
    }

    public float[][] Test(float[] inputs)
    {
        var values = CreateValuesArray(_structure);
        SetInputs(values, inputs);

        for (var l = 1; l < values.Length; l++)
        {
            for (var n = 0; n < values[l].Length; n++)
            {
                values[l][n] = Sigmoid(values[l - 1].Select((v, i) => v * _weights[l - 1][i][n]).Sum() + _biases[l][n]);
            }
        }

        return values;
    }

    public void Train(float[] inputs, float[] outputs)
    {
        var values = Test(inputs);
        SetOutputs(values, outputs);

        for (var l = values.Length - 1; l >= 1; l++)
        {
            for (var n = 0; n < values[l].Length; n++)
            {
                var b = SigmoidDerivative(values[l][n]);
            }
        }
    }

    private static T[] Array<T>(int size, Func<T> value) => Enumerable.Range(0, size).Select(_ => value()).ToArray();

    private static float[] Array(int size, float value) => Array(size, () => value);

    private static float RandomFloat(float min, float max) => min + (max - min) * new Random().NextSingle();

    private static float[] Array(int size, float min, float max) => Array(size, () => RandomFloat(min, max));

    private static float[][] CreateBiasesArray(IEnumerable<int> structure) => structure.Select(n => Array(n, -0.5f, 0.5f)).ToArray();

    private static float[][][] CreateWeightsArray(IReadOnlyList<int> structure) =>
        structure.Take(structure.Count - 1)
              .Select((n, i) => Array(n, () => Array(structure[i + 1], -0.5f, 0.5f)))
              .ToArray();

    private static float[][] CreateValuesArray(IEnumerable<int> structure) => structure.Select(n => Array(n, 0)).ToArray();

    private static void SetInputs(IReadOnlyList<float[]> values, IReadOnlyList<float> inputs)
    {
        for (var i = 0; i < inputs.Count; i++)
        {
            values[0][i] = inputs[i];
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

    private static void SetOutputs(IReadOnlyList<float[]> values, IReadOnlyList<float> outputs)
    {
        for (var i = 0; i < outputs.Count; i++)
        {
            values[^1][i] = outputs[i];
        }
    }

    private static float SigmoidDerivative(float x) => x * (1 - x);
}
