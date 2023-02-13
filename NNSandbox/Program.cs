using NNSandbox;
using NNSandbox.NN;
using NNSandbox.NN.ActivationFunctions;
using NNSandbox.NN.TrainingAlgorithms;

var bit = new float[] { 0.1f, 1.0f };

var trainingDataSet = new TrainingDataSet();

// populate the training data-set
for (var i = 0; i < 1; i++)
{
    // 0 or 0 must be 0
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[0], bit[0], bit[0] },
        ExpectedOutputs = new[] { bit[0] }
    });

    // 0 or 1 must be 1
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[0], bit[1], bit[0] },
        ExpectedOutputs = new[] { bit[1] }
    });

    // 1 or 0 must be 1
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[1], bit[0], bit[0] },
        ExpectedOutputs = new[] { bit[1] }
    });

    // 1 or 1 must be 1
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[1], bit[1], bit[0] },
        ExpectedOutputs = new[] { bit[1] }
    });

    // 0 and 0 must be 0
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[0], bit[0], bit[1] },
        ExpectedOutputs = new[] { bit[0] }
    });

    // 0 and 1 must be 0
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[0], bit[1], bit[1] },
        ExpectedOutputs = new[] { bit[0] }
    });

    // 1 and 0 must be 0
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[1], bit[0], bit[1] },
        ExpectedOutputs = new[] { bit[0] }
    });

    // 1 and 1 must be 1
    trainingDataSet.Add(new()
    {
        Inputs = new[] { bit[1], bit[1], bit[1] },
        ExpectedOutputs = new[] { bit[1] }
    });
}

var nn = new NeuralNetwork()
{
    ActivationFunction = new SigmoidActivationFunction(),
    TrainingAlgorithm = new BacktrackingTrainingAlgorithm()
};

nn.Initialize(new[] { 3, 6, 9, 6, 1 });

// train
for (var i = 1; i <= 10000; i++)
{
    nn.Train(trainingDataSet);

    if (i % 1000 != 0)
    {
        continue;
    }

    Console.WriteLine("Epoch " + i);

    Console.WriteLine("0 or 0");
    Utils.PrintArray(nn.Test(new float[] { bit[0], bit[0], bit[0] })[^1]);

    Console.WriteLine("0 or 1");
    Utils.PrintArray(nn.Test(new float[] { bit[0], bit[1], bit[0] })[^1]);

    Console.WriteLine("1 or 1");
    Utils.PrintArray(nn.Test(new float[] { bit[1], bit[1], bit[0] })[^1]);

    Console.WriteLine("0 and 0");
    Utils.PrintArray(nn.Test(new float[] { bit[0], bit[0], bit[1] })[^1]);

    Console.WriteLine("0 and 1");
    Utils.PrintArray(nn.Test(new float[] { bit[0], bit[1], bit[1] })[^1]);

    Console.WriteLine("1 and 1");
    Utils.PrintArray(nn.Test(new float[] { bit[1], bit[1], bit[1] })[^1]);

    Console.WriteLine(new string('-', 10));
}

await nn.SaveAsync("bitorand.nn");

Console.WriteLine();