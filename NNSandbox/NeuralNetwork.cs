namespace NNSandbox;

internal partial class NeuralNetwork
{
    private static readonly Random _random = new();

    private readonly int[] _structure;
    private readonly float[][] _biases;
    private readonly float[][][] _weights;
    private readonly float _learningRate;
    private readonly float _weightDecay;

    public NeuralNetwork(int[] structure, float learningRate = 1f, float weightDecay = 0.001f)
    {
        _structure = structure;
        _biases = CreateBiasesArray(_structure);
        _weights = CreateWeightsArray(_structure);
        _learningRate = learningRate;
        _weightDecay = weightDecay;
    }

    public float[][] Test(float[] inputs)
    {
        var values = CreateValuesArray(_structure);
        SetInputs(values, inputs);

        for (var l = 1; l < values.Length; l++)
        {
            for (var n = 0; n < values[l].Length; n++)
            {
                values[l][n] = Sigmoid(values[l - 1].Select((v, p) => v * _weights[l - 1][p][n]).Sum() + _biases[l][n]);
            }
        }

        return values;
    }

    public void Train(TrainingDataSet dataSet)
    {
        var biasesSmudge = CreateBiasesSmudgeArray(_biases);
        var weightsSmudge = CreateWeightsSmudgeArray(_weights);

        for (var i = 0; i < dataSet.Count; i++)
        {
            var values = Test(dataSet[i].Inputs);

            var expectedValues = CloneValuesArray(values);
            SetOutputs(expectedValues, dataSet[i].ExpectedOutputs);

            for (var l = values.Length - 1; l >= 1; l--)
            {
                for (var n = 0; n < values[l].Length; n++)
                {
                    var biasSmudge = SigmoidDerivative(values[l][n]) * (expectedValues[l][n] - values[l][n]);
                    biasesSmudge[l][n] += biasSmudge;

                    for (var p = 0; p < values[l - 1].Length; p++)
                    {
                        var weightSmudge = values[l - 1][p] * biasSmudge;
                        weightsSmudge[l - 1][p][n] += weightSmudge;

                        var valueSmudge = _weights[l - 1][p][n] * biasSmudge;
                        expectedValues[l - 1][p] += valueSmudge;
                    }
                }
            }
        }

        for (var l = _structure.Length - 1; l >= 1; l--)
        {
            for (var n = 0; n < _structure[l]; n++)
            {
                _biases[l][n] += biasesSmudge[l][n] * _learningRate;
                _biases[l][n] *= 1 - _weightDecay;

                for (var p = 0; p < _structure[l - 1]; p++)
                {
                    _weights[l - 1][p][n] += weightsSmudge[l - 1][p][n] * _learningRate;
                    _weights[l - 1][p][n] *= 1 - _weightDecay;
                }
            }
        }
    }

    private static T[] Array<T>(int size, Func<T> value) => Enumerable.Range(0, size).Select(_ => value()).ToArray();

    private static float[] Array(int size, float value) => Array(size, () => value);

    private static float RandomFloat(float min, float max) => min + (max - min) * _random.NextSingle();

    private static float[] Array(int size, float min, float max) => Array(size, () => RandomFloat(min, max));

    private static float[][] CreateBiasesArray(int[] structure) => structure.Select(n => Array(n, -0.5f, 0.5f)).ToArray();

    private static float[][][] CreateWeightsArray(int[] structure) =>
        structure.Take(structure.Length - 1)
              .Select((n, idx) => Array(n, () => Array(structure[idx + 1], -0.5f, 0.5f)))
              .ToArray();

    private static float[][] CreateValuesArray(int[] structure) => structure.Select(n => Array(n, 0f)).ToArray();

    private static void SetInputs(float[][] values, float[] inputs)
    {
        if (values[0].Length != inputs.Length)
        {
            throw new ArgumentException(
                "Provided inputs dimension doesnt match Network input dimension\n" +
                "\tExpected: " + values[0].Length + "\n" +
                "\tProvided: " + inputs.Length
                );
        }

        for (var n = 0; n < inputs.Length; n++)
        {
            values[0][n] = inputs[n];
        }
    }

    private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

    private static void SetOutputs(float[][] values, float[] outputs)
    {
        if (values[^1].Length != outputs.Length)
        {
            throw new ArgumentException(
                "Provided outputs dimension doesnt match Network outputs dimension\n" +
                "\tExpected: " + values[^1].Length + "\n" +
                "\tProvided: " + outputs.Length
                );
        }

        for (var n = 0; n < outputs.Length; n++)
        {
            values[^1][n] = outputs[n];
        }
    }

    private static float SigmoidDerivative(float x) => x * (1 - x);

    private static float[][] CreateBiasesSmudgeArray(float[][] biases) => biases.Select(l => l.Select(b => 0f).ToArray()).ToArray();

    private static float[][][] CreateWeightsSmudgeArray(float[][][] weights) => weights.Select(l => l.Select(n => n.Select(w => 0f).ToArray()).ToArray()).ToArray();

    private static float[][] CloneValuesArray(float[][] values) => values.Select(l => l.Select(v => v).ToArray()).ToArray();
}
